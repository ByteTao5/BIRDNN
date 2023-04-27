import os
import logging
import multiprocessing
from repair_retraining import REPAIR
from include.utils.load_onnx import load_ffnn_onnx
from acasxu_repair_list import *
import argparse
import torch.optim as optim
import torch.nn as nn


if __name__ == '__main__':
    savepath = './logs'
    if not os.path.isdir(savepath):
        os.mkdir(savepath)

    # Creating and Configuring Logger
    logger = logging.getLogger()
    Log_Format = logging.Formatter('%(levelname)s %(asctime)s - %(message)s')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler('./include/logs/neural_network_repair.log', 'w+')
    file_handler.setFormatter(Log_Format)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(Log_Format)
    logger.addHandler(console_handler)

    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=int, default=19)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--repair_neurons_num', type=int, default=20)
    parser.add_argument('--epoch', type=int, default=200)
    args = parser.parse_args()

    networks = repair_list
    num_processors = multiprocessing.cpu_count()
    alpha = args.alpha
    beta = args.beta
    lr = args.lr
    epochs = args.epoch
    for n in range(len(networks)):

        item = networks[n]
        i, j = item[0][0], item[0][1]

        if i * 10 + j != args.network:
            continue

        if (i==1 and j ==9) or (i==2 and j ==9):
            # output_limit = 10
            lr = 0.001
        else:
            output_limit = 100

        logging.info(f'Neural Network {i} {j}')
        properties_repair = item[1]
        nn_path = "../nets/ACASXU_run2a_" + str(i) + "_" + str(j) + "_batch_2000.onnx"
        torch_model = load_ffnn_onnx(nn_path)

        rp = REPAIR(torch_model, properties_repair, neg_num=20, pos_num=10000)
        rp.get_data(N=i * 10 + j)
        optimizer = optim.SGD(torch_model.parameters(), lr=lr, momentum=0.9)
        criterion = nn.MSELoss()

        filepath = savepath + '/nnet'+str(i)+str(j)+'_lr'+str(lr)+'_epochs'+str(epochs)+'_alpha'+str(alpha)+'_beta'+str(beta)
        if not os.path.isdir(filepath):
            os.mkdir(filepath)

        rp.repair_model_classification(optimizer, criterion, alpha, beta, filepath, epochs=epochs)
        logging.info('\n****************************************************************\n')



