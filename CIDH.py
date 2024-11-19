from utils.tools import *
from network import *
import torch
import torch.optim as optim
import time
import numpy as np
torch.cuda.set_device(3)
from scipy.linalg import hadamard  # direct import  hadamrd matrix from scipy
import random
import torch.backends.cudnn as cudnn
torch.multiprocessing.set_sharing_strategy('file_system')
from loss.triplet_loss import batch_all_triplet_loss,batch_hard_triplet_loss,corrective_triplet_loss


def get_config():
    config = {
        "alpha": 100,
        "lambda": 0.01,
        "num_samples": 2000,
        "optimizer": {"type": optim.RMSprop, "optim_params": {"lr": 1e-5, "weight_decay": 10 ** -5}},
        "info": "[CIDH]",
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 128,
        "net": ExtractFeature,
        "dataset": "UCMD",
        # "dataset": "WHURS",
        # "dataset": "AID",
        # "dataset": "NWPU-45",
        "n_class":21,
        "save_path": "save/CIDH",
        "epoch": 120,
        "test_map": 1,
        "device": torch.device("cuda:0"),
        "bit_list": [16,32,64,128],
    }
    config = config_dataset(config)
    return config

class CentralLoss(torch.nn.Module):
    def __init__(self, config, bit):
        super(CentralLoss, self).__init__()
        self.hash_targets = self.get_hash_targets(config["n_class"], bit).to(config["device"])
        self.criterion = torch.nn.BCELoss().to(config["device"])

    def forward(self, u, y, ind, config):
        u = u.tanh()
        hash_center = self.label2center(y)
        center_loss = self.criterion(0.5 * (u + 1), 0.5 * (hash_center + 1))

        Q_loss = (u.abs() - 1).cosh().log().mean()
        return center_loss + config["lambda"] * Q_loss   #default 0.0001
    def label2center(self, y):
        hash_center = self.hash_targets[y.argmax(axis=1)]
        return hash_center

    # use algorithm 1 to generate hash centers
    def get_hash_targets(self, n_class, bit):
        H_K = hadamard(bit)
        H_2K = np.concatenate((H_K, -H_K), 0)
        hash_targets = torch.from_numpy(H_2K[:n_class]).float()

        if H_2K.shape[0] < n_class:
            hash_targets.resize_(n_class, bit)
            for k in range(20):
                for index in range(H_2K.shape[0], n_class):
                    ones = torch.ones(bit)
                    # Bernouli distribution
                    sa = random.sample(list(range(bit)), bit // 2)
                    ones[sa] = -1
                    hash_targets[index] = ones
                # to find average/min  pairwise distance
                c = []
                for i in range(n_class):
                    for j in range(n_class):
                        if i < j:
                            TF = sum(hash_targets[i] != hash_targets[j])
                            c.append(TF)
                c = np.array(c)

                if c.min() > bit / 4 and c.mean() >= bit / 2:
                    print(c.min(), c.mean())
                    break
        return hash_targets

class IntensiveLoss(torch.nn.Module):
    def __init__(self, config, bit):
        super(IntensiveLoss, self).__init__()
        self.U = torch.zeros(config["num_train"], bit).float().to(config["device"])
        self.Y = torch.zeros(config["num_train"], config["n_class"]).float().to(config["device"])

    def forward(self, u, y, ind):
        self.U[ind, :] = u.data
        self.Y[ind, :] = y.float()

        inner_product = u @ self.U.t() * 0.5
        likelihood_loss = (1 + (-(inner_product.abs())).exp()).log() + inner_product.clamp(min=0)
        likelihood_loss = likelihood_loss.mean()
        return likelihood_loss

def calc_sim(database_label, train_label):
    S = (database_label @ train_label.t() > 0).float()
    # soft constraint
    r = S.sum() / (1 - S).sum()
    S = S * (1 + r) - r
    return S

def train_val(config, bit):
    device = config["device"]
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(config)

    config["num_train"] = num_train
    net = config["net"](bit).to(device)

    optimizer = config["optimizer"]["type"](net.parameters(), **(config["optimizer"]["optim_params"]))
    criterion_central = CentralLoss(config, bit)
    n_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('numberts of parameters:',n_parameters)

    Best_mAP = 0
    for epoch in range(config["epoch"]):
        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))
        print("%s[%2d/%2d][%s] bit:%d, dataset:%s, training...." % (
            config["info"], epoch + 1, config["epoch"], current_time, bit, config["dataset"]), end="")
        net.train()
        train_loss = 0
        for image, label, ind in train_loader:
            image = image.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            u = net(image)
            loss_central = criterion_central(u, label.float(), ind, config)
            criterion_pair = IntensiveLoss(config, bit)
            loss_intensive = criterion_pair(u, label.float(), ind)
            train_loss = loss_central  + config["alpha"]* loss_intensive
            train_loss.backward()
            optimizer.step()
        train_loss = train_loss / len(train_loader)
        print("\b\b\b\b\b\b\b loss:%.3f" % (train_loss))
        if (epoch + 1) % config["test_map"] == 0:
            Best_mAP = validate(config, Best_mAP, test_loader, dataset_loader, net, bit, epoch, num_dataset)


if __name__ == "__main__":
    config = get_config()
    print(config)

    seed = 58
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = False

    for bit in config["bit_list"]:
        config["pr_curve_path"] = f"log/CIDH_{config['dataset']}_{bit}.json"
        train_val(config, bit)