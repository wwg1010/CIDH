import time
from tqdm import tqdm
import numpy as np
import scipy.io as scio
import torch

def get_precision_recall_by_Hamming_Radius(database_output, database_labels, query_output, query_labels, radius=2):
    # signed_query_output = np.sign(query_output)
    # signed_database_output = np.sign(database_output)
    signed_query_output = query_output
    signed_database_output = database_output
    bit_n = signed_query_output.shape[1]

    ips = np.dot(signed_query_output, signed_database_output.T)
    ips = (bit_n - ips) / 2

    # start_time = time.time()
    ids = np.argsort(ips, 1)
    # end_time = time.time()
    # sort_time = end_time - start_time
    # print("total query: {:d}, sorting time: {:.3f}".format(ips.shape[0], sort_time))

    precX = []
    recX = []
    mAPX = []
    matchX = []
    allX = []
    zero_count = 0
    for i in range(ips.shape[0]):
        if i % 100 == 0:
            tmp_time = time.time()
            end_time = tmp_time
        label = query_labels[i, :]
        label[label == 0] = -1
        idx = np.reshape(np.argwhere(ips[i, :] <= radius), (-1))
        all_num = len(idx)
        if all_num != 0:
            imatch = np.sum(database_labels[idx[:], :] == label, 1) > 0
            match_num = np.sum(imatch)
            precX.append(np.float(match_num) / all_num)
            matchX.append(match_num)
            allX.append(all_num)
            all_sim_num = np.sum(
                np.sum(database_labels[:, :] == label, 1) > 0)
            recX.append(np.float(match_num) / all_sim_num)
            if radius < 10:
                ips_trad = np.dot(
                    query_output[i, :], database_output[ids[i, 0:all_num], :].T)
                ids_trad = np.argsort(-ips_trad, axis=0)
                db_labels = database_labels[ids[i, 0:all_num], :]

                rel = match_num
                imatch = np.sum(db_labels[ids_trad, :] == label, 1) > 0
                Lx = np.cumsum(imatch)
                Px = Lx.astype(float) / np.arange(1, all_num + 1, 1)
                if rel != 0:
                    mAPX.append(np.sum(Px * imatch) / rel)
            else:
                mAPX.append(np.float(match_num) / all_num)

        else:
            print('zero: %d, no return' % zero_count)
            zero_count += 1
            precX.append(np.float(0.0))
            recX.append(np.float(0.0))
            mAPX.append(np.float(0.0))
            matchX.append(0.0)
            allX.append(0.0)

    return np.mean(np.array(precX)), np.mean(np.array(recX)), np.mean(np.array(mAPX))


qb = np.load('G:\\遥感图像哈希检索\\DeepHash-pytorch-master\\save\\DBDH\\nuswide_21_16bits_0.7931965406174993\\tst_binary.npy')
ql = np.load('G:\\遥感图像哈希检索\\DeepHash-pytorch-master\\save\\DBDH\\nuswide_21_16bits_0.7931965406174993\\tst_label.npy')
rb = np.load('G:\\遥感图像哈希检索\\DeepHash-pytorch-master\\save\\DBDH\\nuswide_21_16bits_0.7931965406174993\\trn_binary.npy')
rl = np.load('G:\\遥感图像哈希检索\\DeepHash-pytorch-master\\save\\DBDH\\nuswide_21_16bits_0.7931965406174993\\trn_label.npy')
qb = torch.from_numpy(qb)
ql = torch.from_numpy(ql)
rb = torch.from_numpy(rb)
rl = torch.from_numpy(rl)
#
x,y,z = get_precision_recall_by_Hamming_Radius(qb,  ql,rb, rl)
print(x)