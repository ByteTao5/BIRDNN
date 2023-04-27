from repair_fine_tuning import REPAIR_FT
import os
import logging
import scipy.io as sio
import multiprocessing
from acasxu_repair_list import *
import argparse

if __name__ == '__main__':
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    savepath = './logs'
    if not os.path.isdir(savepath):
        os.mkdir(savepath)

    # Creating and Configuring Logger
    logger = logging.getLogger()
    Log_Format = logging.Formatter('%(levelname)s %(asctime)s - %(message)s')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler('./logs/neural_network_repair.log', 'w+')
    file_handler.setFormatter(Log_Format)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(Log_Format)
    logger.addHandler(console_handler)

    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=int, default=19)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--repair_neurons_num', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()

    networks = repair_list
    num_processors = multiprocessing.cpu_count()
    alpha = args.alpha
    lr = args.lr
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

        logging.info(f'Neural Network {i} {j}, alpha = {alpha}')
        properties_repair = item[1]

        nn_path = "../networks/nnv_format/ACASXU_run2a_" + str(i) + "_" + str(j) + "_batch_2000"
        model = sio.loadmat(nn_path)


        rp = REPAIR_FT(model, properties_repair, nn_path, safe_num=9000, unsafe_num=1000, neg_num=9000, pos_num=1000, repair_num=args.repair_neurons_num, alpha=alpha)

        rp.get_data(N=i * 10 + j, sample_data=False)

        response = rp.response_neuron_localization(rp.neg_data, rp.pos_data)

        rp.solve_safety(response)

        logging.info('\n****************************************************************\n')



