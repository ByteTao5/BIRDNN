
import time
import logging
import random

import torch
import numpy as np
import copy as cp
from scipy.spatial.distance import cdist
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import onnx
import onnxruntime as ort
import h5py



class REPAIR:
    """
    A class for the repair of a neural network

        Attributes:
            properties (Property): Safety properties of the neural network
            delta_neigh (Property): delta neighood of pre-condition of given domain
            neg_num (int): Maximal negative samples from the pre-condition input
            pos_num (int): Maximal positive samples from the neighbourhood
            torch_model (Pytorch): Network model in Pytorch
            data (DATA): data to train, validate and test the network

        Methods:
            sample_negative_data():
                Sample a set of negative data pairs (x,y)s
            sample_postive_neigh_data():
                Sample a set of positive neighbourhood sample (x,y)s
            generate_data():
                Generate random data for training, validating and testing the network model.
            purify_data():
                Remove the negative data from the data
            falsify_data():
                judge a negative input sample
            correct_negative_data():
                Approximate the closest positive data for the negative date.
            response_neuron_localization_sample():
                Find the responsibility of each neuron for the faulty behaviours for sample-wise repair.
            response_neuron_localization_domain():
                Find the responsibility of each neuron for the faulty behaviours for domain-wise repair.

            compute_deviation(model):
                Compute the parameter deviation
            compute_accuracy(model):
                Compute the accuracy of the network model on the test data.
            repair_model_regular(optimizer, loss_fun, alpha, beta, savepath, iters=100, batch_size=200, epochs=200):
                Repair the network model for regression
            repair_model_classification(self, optimizer, loss_fun, alpha, beta, savepath, iters=100, batch_size=2000, epochs=200):
                Repair the network model for classification
    """

    def __init__(self, torch_model, properties_repair, neg_num, pos_num, data=None):
        """
        Constructs all the necessary attributes for the Repair object

        Parameters:
            torch_model (Pytorch): A network model
            properties_repair (list): Safety properties and functions to correct unsafe elements.
            data (list): Data for training, validating and testing the network model
            neg_num (int): Maximal negative samples from the pre-condition input
            pos_num (int): Maximal positive samples from the neighbourhood
        """

        self.properties = [item[0] for item in properties_repair]
        self.neg_num = neg_num
        self.pos_num = pos_num
        self.safe_num = 0
        self.unsafe_num = 0

        self.torch_model = torch_model
        if data is not None:
            self.data = data
        else:
            self.data = None
            # self.data = self.generate_data()
            # self.test()

    def get_data(self, N=0):
        data_path = "../benchmarks/acas_N" + str(N) + "/data/drawndown_test.h5"
        f = h5py.File(data_path, 'r')
        for key in f.keys(): print(f[key].name)
        self.safe_num = f[key].shape[0]
        safe_in = torch.tensor(f[key][:], dtype=torch.float32)
        with torch.no_grad():
            safe_out = self.torch_model(safe_in)

        data_path = "../benchmarks/acas_N" + str(N) + "/data/drawndown.h5"
        f = h5py.File(data_path, 'r')
        for key in f.keys(): print(f[key].name)
        self.pos_num = f[key].shape[0]
        pos_in = torch.tensor(f[key][:], dtype=torch.float32)
        with torch.no_grad():
            pos_out = self.torch_model(pos_in)

        data_path = "../benchmarks/acas_N" + str(N) + "/data/counterexample_test.h5"
        f = h5py.File(data_path, 'r')
        for key in f.keys(): print(f[key].name)
        self.unsafe_num = f[key].shape[0]
        unsafe_in = torch.tensor(f[key][:], dtype=torch.float32)
        with torch.no_grad():
            unsafe_out = self.torch_model(unsafe_in)

        data_path = "../benchmarks/acas_N" + str(N) + "/data/counterexample.h5"
        f = h5py.File(data_path, 'r')
        for key in f.keys(): print(f[key].name)
        self.neg_num = f[key].shape[0]
        neg_in = torch.tensor(f[key][:], dtype=torch.float32)
        with torch.no_grad():
            neg_out = self.torch_model(neg_in)

        # neg_in = neg_in[:1000]
        # neg_out = neg_out[:1000]

        self.t0 = time.time()

        [cor_in, cor_out] = self.correct_negative_data([neg_in, neg_out], [pos_in, pos_out])

        #10000个数据，按8:2分成训练集和验证集，再按9:1分别分配正样本和负样本，故正训:负训:正验:负验 = 7200 :800 :1800 :200
        # 1:9
        train_data = [torch.cat([pos_in[:800], cor_in[:7200]], dim=0), torch.cat([pos_out[:800], cor_out[:7200]], dim=0)]

        valid_data = [torch.cat([pos_in[800:1000], cor_in[7200:9000]], dim=0), torch.cat([pos_out[800:1000], cor_out[7200:9000]], dim=0)]

        test_data = [torch.cat([safe_in[:4500], unsafe_in[:500]], dim=0), torch.cat([safe_out[:4500], unsafe_out[:500]], dim=0)]
        self.data = DATA(train_data, valid_data, test_data)


    def sample_postive_neigh_data(self):
        """
        Generate positive neighbourhood input samples in the delta-neighood of the negative domains
        
        Parameters:
            None

        Returns:
            pos_data_set: (x,y)s
        """
        pnd_x = []
        pnd_y = []
        while len(pnd_x) < self.pos_num:
            lbs = self.properties[0].input_ranges[0]
            ubs = self.properties[0].input_ranges[1]
            train_x = torch.tensor([np.random.uniform(lbs[i], ubs[i], 10000).tolist() for i in range(len(lbs))]).T
            with torch.no_grad():
                train_y = self.torch_model(train_x)
            pos_set = self.purify_data([train_x, train_y])
            if len(pos_set[0]) != 0:
                pnd_x = pos_set[0] if len(pnd_x) == 0 else torch.cat((pnd_x, pos_set[0]), axis=0)
                pnd_y = pos_set[1] if len(pnd_y) == 0 else torch.cat((pnd_y, pos_set[1]), axis=0)
        return [pnd_x[0:self.pos_num], pnd_y[0:self.pos_num]]

    def generate_data(self, num=10000):
        """
        Generate random data for training, validating and testing the network model.

        Parameters:
            num (int): Total number of the data

        Returns:
            data (DATA): Data for training, validating and testing the network model
        """
        lbs = self.properties[0].input_ranges[0]
        ubs = self.properties[0].input_ranges[1]

        train_x = torch.tensor([np.random.uniform(lbs[i], ubs[i], num).tolist() for i in range(len(lbs))]).T
        with torch.no_grad():
            train_y = self.torch_model(train_x)
        train_data = self.purify_data([train_x, train_y])

        valid_x = torch.tensor([np.random.uniform(lbs[i], ubs[i], int(num * 0.5)).tolist() for i in range(len(lbs))]).T
        with torch.no_grad():
            valid_y = self.torch_model(valid_x)
        valid_data = self.purify_data([valid_x, valid_y])

        test_x = torch.tensor([np.random.uniform(lbs[i], ubs[i], int(num * 0.5)).tolist() for i in range(len(lbs))]).T
        with torch.no_grad():
            test_y = self.torch_model(test_x)
        test_data = self.purify_data([test_x, test_y])

        return DATA(train_data, valid_data, test_data)
    

    def purify_data(self, data):
        """
        Remove the unsafe data from the data

        Parameters:
            data (np.ndarray): Data pairs (x,y)s

        Returns:
            data_x (np.ndarray): Safe data x on all safety properties
            data_y (np.ndarray): safe data y on all safety properties
        """
        data_x = data[0]
        data_y = data[1]
        for p in self.properties:
            lb, ub = p.lbs, p.ubs
            for ufd in p.unsafe_domains:
                M, vec = torch.tensor(ufd[0], dtype=torch.float32), torch.tensor(ufd[1],dtype=torch.float32)
                bools = torch.ones(len(data_x), dtype=torch.bool)
                for n in range(len(lb)):
                    lbx, ubx = lb[n], ub[n]
                    x = data_x[:, n]
                    bools = (x > lbx) & (x < ubx) & bools

                if not torch.any(bools):
                    continue
                outs = torch.mm(M, data_y.T) + vec
                out_bools = torch.all(outs<=0, dim=0) & bools
                if not torch.any(out_bools):
                    continue

                safe_indx = torch.nonzero(~out_bools)[:,0]
                data_x = data_x[safe_indx]
                data_y = data_y[safe_indx]

        return [data_x, data_y]

    def correct_negative_data(self, neg_data, pos_data, NEG_NUM=50):
        """
        Approximate the closest positive data for the negative date.

        Parameters:
            neg_data (np.ndarray): Data pairs (x,y)s, unsafe data
            pos_data (np.ndarray): Data pairs (x,y)s, safe data around the unsafe data

        Returns:
            cor_data (np.ndarray): Data pairs (x,y)s, correct data
        """
        neg_data_y = neg_data[1]
        pos_data_y = pos_data[1]
        dist = torch.tensor(cdist(neg_data_y, pos_data_y)) # calculate the min dist
        min_n_values, min_n_indexes = torch.topk(dist, k=NEG_NUM, dim=1, largest=False, sorted=True)
        cor_data_y = []

        # for i in range(len(neg_data_y)):
        #     avg_y = torch.tensor([np.average(pos_data_y[min_n_indexes[i]], axis=0)]) # TODO: is this right?
        #     tv_avg_unsafe = False
        #
        #     for p in self.properties:
        #         for ufd in p.unsafe_domains:
        #             M, vec = torch.tensor(ufd[0], dtype=torch.float32), torch.tensor(ufd[1], dtype=torch.float32)
        #             bools = torch.ones(len(avg_y), dtype=torch.bool)
        #             outs = torch.mm(M, avg_y.T) + vec
        #             out_bools = torch.all(outs <= 0, dim=0) & bools
        #             if not torch.any(out_bools):
        #                 continue
        #             tv_avg_unsafe = True
        #     if tv_avg_unsafe:
        #         cor_data_y.append(pos_data_y[min_n_indexes[i][0]])
        #     else:
        #         cor_data_y.append(torch.squeeze(avg_y, dim=0))

        for i in range(len(neg_data_y)):
            cor_data_y.append(pos_data_y[min_n_indexes[i][0]])

        return [neg_data[0], torch.stack(cor_data_y, dim=0)]
    

    def compute_deviation(self, model):
        """
        Compute the parameter deviation

        Parameters:
            model (Pytorch): A network model

        Returns:
            rls (float): Parameter deviation
        """

        model.eval()
        with torch.no_grad():
            predicts = model(self.data.test_data[0]) # The minimum is the predication
        actl_ys = self.data.test_data[1]
        rls = torch.sqrt(torch.sum(torch.square(predicts-actl_ys)))/ torch.sqrt(torch.sum(torch.square(actl_ys)))
        logging.info(f'Output deviation on the test data: {rls :.2f} ')
        return rls


    def compute_accuracy(self, model):
        """
        Compute the accuracy of the network model on the test data.

        Parameters:
            model (Pytorch): A network model

        Returns:
            accuracy (float): Accuracy
        """

        model.eval()
        with torch.no_grad():
            predicts = model(self.data.test_data[0])
        # pred_actions = torch.argmax(predicts, dim=1)
        # actl_actions = torch.argmax(self.data.test_data[1] * (-1), dim=1)
        # actions_times = torch.tensor([len(torch.nonzero(actl_actions==n)[:,0]) for n in range(predicts.shape[1])])
        # self.optimal_dim = torch.argmax(actions_times)
        # accuracy = len(torch.nonzero(pred_actions == actl_actions)) / len(predicts)

        purify_test = self.purify_data([self.data.test_data[0], predicts])
        accuracy = len(purify_test[0]) / len(predicts)

        logging.info(f'Accuracy on the test data: {accuracy * 100 :.2f}% ')
        return accuracy

    # def test(self, num=10000):
    #     lbs = self.properties[0].input_ranges[0]
    #     ubs = self.properties[0].input_ranges[1]
    #
    #     train_x = torch.tensor([np.random.uniform(lbs[i], ubs[i], num).tolist() for i in range(len(lbs))]).T
    #     with torch.no_grad():
    #         train_y = self.torch_model(train_x)
    #     safe_data = self.purify_data([train_x, train_y])
    #     unsafe_data = self.falsify_data([train_x, train_y])
    #
    #     if len(unsafe_data[0]) != 0:
    #         cor_data = self.correct_negative_data(unsafe_data, safe_data)
    #     return True

    
    def repair_model_regular(self, optimizer, loss_fun, alpha, beta, savepath, iters=100, batch_size=200, epochs=200):
        """
        Repair the network model for regression

        Parameters:
            optimizer (optimizer): Optimizer for the training of the model
            loss_fun (function): Loss function for the training of the model
            alpha (float): Weight of the distance between safe domains and unsafe domains in the loss function
            beta (float): Wight of the loss value on training data in the loss function
            savepath (str): Path to save the repaired network
            iters (int): Number of the iterations
            batch_size (int): Batch size
            epochs (int): Number of epochs

        """

        t0 = time.time()
        all_test_deviation = []
        repaired = False
        for num in range(iters):
            logging.info(f'Iteration of repair: {num}')
            # deviation = self.compute_deviation(self.torch_model)
            # all_test_deviation.append(deviation)

            # Compute unsafe domain of the network and construct corrected safe training data
            tt0 = time.time()

            neg_sample = self.sample_negative_data()

            if len(neg_sample[0]) < self.neg_num:
                repaired = True
                if len(neg_sample[0]) == 0:
                    break

            pos_sample = self.sample_postive_neigh_data()
            [train_adv_x, train_adv_y] = self.correct_negative_data(neg_sample, pos_sample, NEG_NUM=10)
            training_dataset_adv = TensorDataset(train_adv_x, train_adv_y)
            train_loader_adv = DataLoader(training_dataset_adv, batch_size, shuffle=True)

            training_dataset_train = TensorDataset(self.data.train_data[0], self.data.train_data[1])
            train_loader_train = DataLoader(training_dataset_train, batch_size, shuffle=True)

            logging.info('Start retraining for the repair...')
            self.torch_model.train()
            for e in range(epochs):
                # print('\rEpoch: '+str(e)+'/'+str(epochs),end='')
                for batch_idx, data in enumerate(zip(train_loader_adv, train_loader_train)):
                    datax, datay = data[0][0], data[0][1]
                    datax_train, datay_train = data[1][0], data[1][1]
                    optimizer.zero_grad()

                    predicts_adv = self.torch_model(datax)
                    loss_adv = loss_fun(datay, predicts_adv)
                    predicts_train = self.torch_model(datax_train)
                    loss_train = loss_fun(datay_train, predicts_train)
                    loss = alpha*loss_adv + beta*loss_train
                    loss.backward()
                    optimizer.step()
            self.torch_model.cpu()
            logging.info('The retraining is done\n')
            if num % 1 == 0:
                torch.save(self.torch_model, savepath + "/epoch" + str(num) + ".pt")

            if repaired is True:
                break

        if not repaired:
            logging.info('The accurate and safe candidate model is found? False')
            logging.info(f'Total running time: {time.time() - t0 :.2f} sec')
            torch.save(self.torch_model, savepath + "/unrepaired_model.pt")


    def repair_model_classification(self, optimizer, loss_fun, alpha, beta, savepath, iters=100, batch_size=2000, epochs=200):
        """
        Repair the network model for classification

        Parameters:
            optimizer (optimizer): Optimizer for the training of the model
            loss_fun (function): Loss function for the training of the model
            alpha (float): Weight of the distance between safe domains and unsafe domains in the loss function
            beta (float): Wight of the loss value on training data in the loss function
            savepath (str): Path to save the repaired network
            iters (int): Number of the iterations
            batch_size (int): Batch size
            epochs (int): Number of epochs

        """
        all_test_accuracy = []
        accuracy_old = 1.0
        candidate_old = cp.deepcopy(self.torch_model)
        reset_flag = False
        repaired = False

        terminal_num = 0


        neg_sample_all = [[], []]
        for num in range(iters):
            logging.info(f'Iteration of repair: {num}')
            accuracy_new = self.compute_accuracy(self.torch_model)
            if accuracy_new >= 0.99:
                terminal_num += 1
            else:
                terminal_num = 0

            if accuracy_new >= 0.99 and num == 0:
                break


            # Restore the previous model if there is a large drop of accuracy in the current model
            if accuracy_old - accuracy_new > 0.1:
                logging.info('A large drop of accuracy!')
                self.torch_model = cp.deepcopy(candidate_old)
                # Decrease the learning rate to reduce the accuracy degradation
                lr = optimizer.param_groups[0]['lr'] * 0.8
                logging.info(f'Current lr: {lr}')
                optimizer = optim.SGD(self.torch_model.parameters(), lr=lr, momentum=0.9)
                reset_flag = True
                continue

            # Compute unsafe domain of the network and construct corrected safe training data
            if not reset_flag:
                candidate_old = cp.deepcopy(self.torch_model)
                accuracy_old = accuracy_new
                all_test_accuracy.append(accuracy_new)

                # neg_sample = self.sample_negative_data()
                #
                # if len(neg_sample[0]) < self.neg_num:
                #     neg_sample_all[0] = neg_sample_all[0] if len(neg_sample[0]) == 0 else torch.cat((neg_sample_all[0], neg_sample[0]), axis=0)
                #     neg_sample_all[1] = neg_sample_all[1] if len(neg_sample[1]) == 0 else torch.cat((neg_sample_all[1], neg_sample[1]), axis=0)
                #     neg_index = random.sample(range(len(neg_sample_all[0])), self.neg_num)
                #     neg_sample = [neg_sample_all[0][neg_index], neg_sample_all[1][neg_index]]
                # else:
                #     neg_sample_all = neg_sample
                #
                # pos_sample = self.sample_postive_neigh_data()
                # [train_adv_x, train_adv_y] = self.correct_negative_data(neg_sample, pos_sample, NEG_NUM=10)
                # training_dataset_adv = TensorDataset(train_adv_x, train_adv_y)
                # train_loader_adv = DataLoader(training_dataset_adv, batch_size, shuffle=True)
                #
                # training_dataset_train = TensorDataset(self.data.train_data[0], self.data.train_data[1])
                # train_loader_train = DataLoader(training_dataset_train, batch_size, shuffle=True)

            training_dataset_train = TensorDataset(self.data.train_data[0], self.data.train_data[1])
            train_loader_train = DataLoader(training_dataset_train, batch_size, shuffle=True)
            reset_flag = False
            logging.info('Start retraining for the repair...')
            self.torch_model.train()
            for e in range(epochs):
                # print('\rEpoch: '+str(e)+'/'+str(epochs),end='')
                for batch_idx, data in enumerate(train_loader_train):
                    datax, datay = data[0][0], data[0][1]
                    datax_train, datay_train = data[1][0], data[1][1]
                    optimizer.zero_grad()

                    predicts_adv = self.torch_model(datax)
                    loss_adv = loss_fun(datay, predicts_adv)
                    predicts_train = self.torch_model(datax_train)
                    loss_train = loss_fun(datay_train, predicts_train)
                    loss = alpha*loss_adv + beta*loss_train
                    loss.backward()
                    optimizer.step()

            self.torch_model.cpu()
            logging.info('The retraining is done\n')
            if num % 1 == 0:
                torch.save(self.torch_model, savepath + "/acasxu_epoch" + str(num) + ".pt")

            if terminal_num >= 10:
                break

        if not repaired:
            # logging.info('The accurate and safe candidate model is found? False')
            logging.info(f'Total running time: {time.time() - self.t0 :.2f} sec')
            # torch.save(self.torch_model, savepath + "/unrepaired_model.pt")



class DATA:
    """
    A class for data

    Attributes:
        train_data (np.ndarray): Training data
        valid_data (np.ndarray): Validation data
        test_data (np.ndarray): Test data
    """
    def __init__(self, train_data, valid_data, test_data):
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data


    

    