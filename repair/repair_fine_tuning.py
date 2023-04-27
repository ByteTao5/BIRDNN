import pyswarms as ps
from itertools import zip_longest
import h5py
import time
import torch
import numpy as np
from tqdm import tqdm

def relu(x):
    return np.maximum(x, 0)

def leaky_relu(x):
    return np.where(x < 0, 0.5 * x, x)

def tanh(x):
    return np.tanh(x)

def elu(x):
    return np.where(x <= 0, 0.5 * (np.exp(x) - 1), x)

def setdiff2d_list(arr1, arr2):
    # test = []
    # for x in arr1:
    #     if np.isin(arr1, vector).all(axis=1).any()
    #         test.append(x)
    # return np.array(test)
    return np.array([x for x in arr1 if not np.isin(arr2, x).all(axis=1).any()])


class REPAIR_FT:

    def __init__(self, model, properties_repair, nn_path, safe_num, unsafe_num, neg_num, pos_num, repair_num, alpha, data=None):
        
        """
        Constructs all the necessary attributes for the Repair object

        Parameters:
            model (Pytorch): A network model
            properties_repair (list): Safety properties and functions to correct unsafe elements.
            data (list): Data for training, validating and testing the network model
            unsafe_num (int): Maximal negative samples from the pre-condition input
            safe_num (int): Maximal positive samples from the neighbourhood
        """

        self.properties = [item[0] for item in properties_repair]
        self.safe_num = safe_num
        self.unsafe_num = unsafe_num
        self.neg_num = neg_num
        self.pos_num = pos_num
        self.model = model
        self.repair_num = repair_num
        self.alpha = alpha
        self.safe_data = []
        self.unsafe_data = []
        self.pos_data = []
        self.neg_data = []
        self.nn_path = nn_path

    def get_data(self, N=0, sample_data=True):
        if N in {19, 29, 33} and not sample_data:
            data_path = "../benchmarks/acas_N" + str(N) + "/data/drawndown_test.h5"
            f = h5py.File(data_path, 'r')
            for key in f.keys(): print(f[key].name)
            self.safe_num = f[key].shape[0]
            safe_in = np.array(f[key][:])
            safe_out = self.get_output(safe_in.T).T

            data_path = "../benchmarks/acas_N" + str(N) + "/data/drawndown.h5"
            f = h5py.File(data_path, 'r')
            for key in f.keys(): print(f[key].name)
            self.pos_num = f[key].shape[0]
            pos_in = np.array(f[key][:])
            pos_out = self.get_output(pos_in.T).T

            data_path = "../benchmarks/acas_N" + str(N) + "/data/counterexample_test.h5"
            f = h5py.File(data_path, 'r')
            for key in f.keys(): print(f[key].name)
            self.unsafe_num = f[key].shape[0]
            unsafe_in = np.array(f[key][:])
            unsafe_out = self.get_output(unsafe_in.T).T

            data_path = "../benchmarks/acas_N" + str(N) + "/data/counterexample.h5"
            f = h5py.File(data_path, 'r')
            for key in f.keys(): print(f[key].name)
            self.neg_num = f[key].shape[0]
            neg_in = np.array(f[key][:])
            neg_out = self.get_output(neg_in.T).T

            self.safe_data = [safe_in, safe_out]
            self.unsafe_data = [unsafe_in, unsafe_out]

            self.pos_data = [pos_in, pos_out]
            self.neg_data = [neg_in, neg_out]
        else:
            self.pos_data, self.neg_data = self.sample_data(self.pos_num, self.neg_num)

            self.safe_data, self.unsafe_data = self.sample_data(self.safe_num, self.unsafe_num)




    def sample_data(self, pos_num, neg_num):
        """
        Generate positive  input samples of the negative domains

        Parameters:
            None

        Returns:
            pos_data_set: (x,y)s
        """
        pos_x = []
        pos_y = []
        neg_x = []
        neg_y = []
        while len(pos_x) < pos_num or len(neg_x) < neg_num:
            print(len(pos_x), len(neg_x))
            lbs = self.properties[0].input_ranges[0]
            ubs = self.properties[0].input_ranges[1]
            train_x = np.array([np.random.uniform(lbs[i], ubs[i], 10000).tolist() for i in range(len(lbs))])
            # with torch.no_grad():

            train_y = self.get_output(train_x)

            train_x = train_x.T
            train_y = train_y.T
            pos_set = self.purify_data([train_x, train_y])
            if len(pos_set[0]) != 0 and len(pos_x) < pos_num:
                pos_x = pos_set[0] if len(pos_x) == 0 else np.vstack((pos_x, pos_set[0]))
                pos_y = pos_set[1] if len(pos_y) == 0 else np.vstack((pos_y, pos_set[1]))
            neg_set = [setdiff2d_list(train_x, pos_set[0]), setdiff2d_list(train_y, pos_set[1])]
            if len(neg_set[0]) != 0 and len(neg_x) < neg_num:
                neg_x = neg_set[0] if len(neg_x) == 0 else np.vstack((neg_x, neg_set[0]))
                neg_y = neg_set[1] if len(neg_y) == 0 else np.vstack((neg_y, neg_set[1]))
        return [pos_x[0:pos_num], pos_y[0:pos_num]], [neg_x[0:neg_num], neg_y[0:neg_num]]

    def get_output(self, input):
        num_layers = self.model['W'].shape[1]
        for i in range(num_layers):
            if (i < num_layers - 1):
                weights = self.model['W'][0, i]
                biases = self.model['b'][0, i]
                # print(weights.shape)
                # print(biases.shape)
                # print(input.shape)
                output = relu(np.matmul(weights, input) + biases)
                # print("output shape", i, output.shape)
                input = output
            else:
                weights = self.model['W'][0, i]
                biases = self.model['b'][0, i]
                output = np.matmul(weights, input) + biases
                return output

    def sample_postive_neigh_data(self, num):
        """
        Generate positive neighbourhood input samples in the delta-neighood of the negative domains
        
        Parameters:
            None

        Returns:
            pos_data_set: (x,y)s
        """
        pnd_x = []
        pnd_y = []
        while len(pnd_x) < num:
            lbs = self.properties[0].input_ranges[0]
            ubs = self.properties[0].input_ranges[1]
            train_x = np.array([np.random.uniform(lbs[i], ubs[i], 10000).tolist() for i in range(len(lbs))])
            # with torch.no_grad():

            train_y = self.get_output(train_x)

            train_x = train_x.T
            train_y = train_y.T

            pos_set = self.purify_data([train_x, train_y])
            if len(pos_set[0]) != 0:
                pnd_x = pos_set[0] if len(pnd_x) == 0 else np.vstack((pnd_x, pos_set[0]))
                pnd_y = pos_set[1] if len(pnd_y) == 0 else np.vstack((pnd_y, pos_set[1]))
        return [pnd_x[0:num], pnd_y[0:num]]

    def purify_data(self, data):
        """
        Remove the unsafe data from the data

        Parameters:
            data (np.ndarray): Data pairs (x,y)s

        Returns:
            data_x (np.ndarray): Safe data x on all safety properties
            data_y (np.ndarray): safe data y on all safety properties
        """
        data_x = torch.tensor(data[0], dtype=torch.float32)
        data_y = torch.tensor(data[1], dtype=torch.float32)

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

        return [data_x.numpy(), data_y.numpy()]

    def response_neuron_localization(self, neg_data, pos_data):
        self.overall_starttime = time.time()

        response = {}
        nn_model = self.model
        num_layers = nn_model['W'].shape[1]

        neg_input = neg_data[0].T
        neg_beh = {}

        pos_input = pos_data[0].T
        pos_beh = {}

        for i in range(num_layers):

            if i < num_layers - 1:
                weights = nn_model['W'][0, i]
                biases = nn_model['b'][0, i]
                neg_output = relu(np.matmul(weights, neg_input) + biases)
                pos_output = relu(np.matmul(weights, pos_input) + biases)

                neg_beh[i] = neg_output
                pos_beh[i] = pos_output

                neg_input = neg_output
                pos_input = pos_output

            else:
                weights = nn_model['W'][0, i]
                biases = nn_model['b'][0, i]
                neg_output = np.matmul(weights, neg_input) + biases
                pos_output = np.matmul(weights, pos_input) + biases
                neg_beh[i] = neg_output
                pos_beh[i] = pos_output

        for i in range(num_layers):
            beh_diff = np.sum(pos_beh[i].T, axis=0) - np.sum(neg_beh[i].T, axis=0)
            response[i] = np.abs(beh_diff)
        return response

    def response_neuron_localization_sample(self, neg_sample, pos_data):
        response = {}
        nn_model = self.model
        num_layers = nn_model['W'].shape[1]

        neg_input = neg_sample
        neg_beh = {}

        pos_input = pos_data[0].T
        pos_beh = {}

        for i in range(num_layers):

            if i < num_layers - 1:
                weights = nn_model['W'][0, i]
                biases = nn_model['b'][0, i]
                neg_output = relu(np.matmul(weights, neg_input) + np.ravel(biases))
                pos_output = relu(np.matmul(weights, pos_input) + biases)

                neg_beh[i] = neg_output
                pos_beh[i] = pos_output

                neg_input = neg_output
                pos_input = pos_output

            else:
                weights = nn_model['W'][0, i]
                biases = nn_model['b'][0, i]
                neg_output = np.matmul(weights, neg_input) + np.ravel(biases)
                pos_output = np.matmul(weights, pos_input) + biases
                neg_beh[i] = neg_output
                pos_beh[i] = pos_output

        for i in range(num_layers):
            # print(pos_beh[i].T.shape)
            # print(np.ravel(neg_beh[i]).T.shape)
            beh_diff = np.abs(pos_beh[i].T - np.ravel(neg_beh[i]).T)
            response[i] = np.sum(beh_diff, axis=0)
        return response

    def response_neuron_localization_domain(self):
        response = {}
        for neg_index in tqdm(range(len(self.neg_data[0]))):
            response_temp = self.response_neuron_localization_sample(self.neg_data[0][neg_index], self.pos_data)
            if response:
                for i in range(len(response_temp)):
                    response[i] = response[i] + response_temp[i]
            else:
                for i in range(len(response_temp)):
                    response[i] = response_temp[i]
        return response

    def model_repair(self, x, r_weight):
        nn_model = self.model
        num_layers = nn_model['W'].shape[1]

        self.sort_index.sort(key=lambda x: x[0])

        index_sort = 0
        input = x.T
        y = []
        for i in range(num_layers):

            if i < num_layers - 1:
                weights = nn_model['W'][0, i]
                biases = nn_model['b'][0, i]
                output = relu(np.matmul(weights, input) + biases)

                if index_sort < np.shape(self.sort_index)[0]:
                    while self.sort_index[index_sort][0] == i:
                        output[self.sort_index[index_sort][1]] = output[self.sort_index[index_sort][1]] * (
                                    1 + r_weight[index_sort])
                        index_sort += 1
                        if index_sort >= np.shape(self.sort_index)[0]:
                            break

                input = output

            else:
                weights = nn_model['W'][0, i]
                biases = nn_model['b'][0, i]
                output = np.matmul(weights, input) + biases

                if index_sort < np.shape(self.sort_index)[0]:
                    while self.sort_index[index_sort][0] == i:
                        output[self.sort_index[index_sort][1]] = output[self.sort_index[index_sort][1]] * (
                                    1 + r_weight[index_sort])
                        index_sort += 1
                        if index_sort >= np.shape(self.sort_index)[0]:
                            break

                y = output

        return y.T

    def solve_safety(self, response_dict):
        response = [[], [], [], [], [], [], []]
        for i in range(len(response_dict)):
            response[i] = response_dict[i]

        row = len(response)
        col = 0
        for sub_res in response:
            col = max(col, np.shape(sub_res)[0])

        beh_diff_unsort = np.zeros([row, col])
        
        for i in range(0, np.shape(beh_diff_unsort)[0]):
            beh_diff_unsort[i] = [sum(x) for x in zip_longest(beh_diff_unsort[i], response[i].flatten(), fillvalue=0)]

        matrix_with_indices = [(beh_diff_unsort[i][j], i, j) for i in range(len(beh_diff_unsort)) for j in range(len(beh_diff_unsort[0]))]
        sorted_matrix = sorted(matrix_with_indices, key=lambda x: -x[0])
        sort_index = [[x[1], x[2]] for x in sorted_matrix[:self.repair_num]]

        # self.r_layer = []
        # self.r_neuron = []
        # for i in range(0, self.repair_num):
        #     self.r_layer.append(sort_index[i][0])
        #     self.r_neuron.append(sort_index[i][1])

        self.sort_index = sort_index
        # x = np.array([[1.1, 1.2, 1.3, 1.4, 1.5],[2,3,4,5,6]], dtype=np.float32)
        # y = self.model_repair(x, [0.1, 0.2, 0.3])

        fault_loc_time = time.time() - self.overall_starttime

        print('Repair:')

        # print('\nRepair layer: {}'.format(self.r_layer))
        # print('Repair neuron: {}'.format(self.r_neuron))

        # start repair
        best_pos = self.repair()

        # verify prob diff and model accuracy after repair , weight, layer, neuron
        r_safety, r_acc = self.net_accuracy_test(self.safe_data[0], self.unsafe_data[0], best_pos)


        print('Percentage of discriminatory instance after repair: {}'.format(r_safety))
        print('Network Accuracy after repair: {}'.format(r_acc))

        ori_safety, ori_accuracy = self.net_accuracy_test(self.safe_data[0], self.unsafe_data[0], [])
        self.acc_datalen = 100

        print('Percentage of discriminatory instance before repair: {}'.format(ori_safety))
        print('Network Accuracy before repair: {}'.format(ori_accuracy))
        print('Fault localization time(s): {}'.format(fault_loc_time))
        print('Total execution time(s): {}'.format(time.time() - self.overall_starttime))

        return True

    # pso find a repair range

    def repair(self):
        # repair
        print('Start reparing...')
        # self.mode = 'repair'
        options = {'c1': 0.41, 'c2': 0.41, 'w': 0.8}   #parameter tuning
        #'''# original
        optimizer = ps.single.GlobalBestPSO(n_particles=20, dimensions=self.repair_num, options=options,
                                            bounds=([[-1.0] * self.repair_num, [1.0] * self.repair_num]),
                                            init_pos=np.zeros((20, self.repair_num), dtype=float), ftol=1e-3,
                                            ftol_iter=10)
        #'''

        # Perform optimization
        best_cost, best_pos = optimizer.optimize(self.pso_fitness_func, iters=100)
        # print(best_pos)

        return best_pos



    def pso_fitness_func(self, weight):

        result = []
        for i in range (0, int(len(weight))):
            r_weight =  weight[i]

            safety, accuracy = self.net_accuracy_test(self.pos_data[0], self.neg_data[0], r_weight)

            _result = (1.0 - self.alpha) * safety + self.alpha * (1.0 - accuracy)
            # _result = (1.0 - self.alpha) * accuracy + self.alpha * (1.0 - safety)


            result.append(_result)

        return result

    # def repair_one_layer(self, input, r_weight, index_sort, j, nn_model):


    def net_accuracy_test(self, right_data, wrong_data, r_weight=[]):
        
        REPAIR_flag = (len(r_weight) != 0)

        input_s = right_data
        input_us = wrong_data
        if REPAIR_flag:
            output_s = self.model_repair(input_s, r_weight)
            output_us = self.model_repair(input_us, r_weight)
        else:
            output_s = self.get_output(input_s.T).T
            output_us = self.get_output(input_us.T).T


        purify_pos_data = self.purify_data([input_s, output_s])
        purify_neg_data = self.purify_data([input_us, output_us])

        acc = np.shape(purify_pos_data[0])[0] / np.shape(input_s)[0]
        safety = (np.shape(input_us)[0] - np.shape(purify_neg_data[0])[0]) / np.shape(input_us)[0]

        return safety, acc
