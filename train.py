import os
import argparse
import json
import pickle
import random
import numpy as np
import torch
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import SGD, Adam
from data_process import eedi_a, eedi_b, ednet
from dkt import DKT
from sakt import SAKT
from saint import SAINT
from atkt import ATKT
from dkt_plus import DKTPlus
from kqn import KQN
from utils import collate_fn

from sklearn.model_selection import train_test_split
device = "cuda:0"

seed = 42
random.seed(seed)
np.random.seed(seed)


def main(model_name,option, dataset_name, method_name):
    if not os.path.isdir("ckpts"):
        os.mkdir("ckpts")

    ckpt_path = os.path.join("ckpts", model_name)
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)

    ckpt_path = os.path.join(ckpt_path, dataset_name)
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)

    with open("config.json") as f:
        config = json.load(f)
        model_config = config[model_name]
        train_config = config["train_config"]

    batch_size = train_config["batch_size"]
    num_epochs = train_config["num_epochs"]
    train_ratio = train_config["train_ratio"]
    learning_rate = train_config["learning_rate"]
    optimizer = train_config["optimizer"]  # can be [sgd, adam]
    seq_len = train_config["seq_len"]

    if dataset_name == "eedi_a":
        dataset = eedi_a(seq_len, method_name)
    elif dataset_name == "eedi_b":
        dataset = eedi_b(seq_len, method_name)
    elif dataset_name == "ednet":
        dataset = ednet(seq_len, method_name)

    q_matrix = torch.tensor(dataset.q_matrix).to(device)

    with open(os.path.join(ckpt_path, "model_config.json"), "w") as f:
        json.dump(model_config, f, indent=4)
    with open(os.path.join(ckpt_path, "train_config.json"), "w") as f:
        json.dump(train_config, f, indent=4)

    if model_name == "dkt":
        model = DKT(dataset.num_q, q_matrix,option, **model_config).to(device)
    elif model_name == "dkvmn":
        model = DKVMN(dataset.num_q, q_matrix,option, **model_config).to(device)
    elif model_name == "sakt":
        model = SAKT(dataset.num_q, q_matrix,option, **model_config).to(device)
    elif model_name == "saint":
        model = SAINT(dataset.num_q, q_matrix,option, **model_config).to(device)
    elif model_name == "dkt+":
        model = DKTPlus(dataset.num_q, q_matrix,option, **model_config).to(device)
    elif model_name == "kqn":
        model = KQN(dataset.num_q, q_matrix,option, 30).to(device)
    elif model_name == "atkt":
        model = ATKT(dataset.num_q, q_matrix,option, **model_config).to(device)

    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, generator=torch.Generator(device='cuda:0')
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, generator=torch.Generator(device='cuda:0')
    )

    opt = Adam(model.parameters(), learning_rate)

    aucs, loss_means= \
        model.train_model(
            train_loader, test_loader, num_epochs, opt, ckpt_path
        )

    with open(os.path.join(ckpt_path, "aucs.pkl"), "wb") as f:
        pickle.dump(aucs, f)
    with open(os.path.join(ckpt_path, "loss_means.pkl"), "wb") as f:
        pickle.dump(loss_means, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="dkt",
        help="The name of the model to train. \
            The possible models are in [dkt, dkt+, kqn, sakt, saint, atkt]. \
            The default model is dkt."
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        default="eedi_a",
        help="The name of the dataset to use in training. \
            The possible datasets are in \
            [eedi_a, eedi_b, ednet]. \
            The default dataset is eedi_a."
    )

    parser.add_argument(
        "--method_name",
        type=str,
        default="method_1",
        help="The name of the method to use in training. \
            The possible methods are in \
            [method_1, method_2]. \
            The default method is method_1."
    )

    parser.add_argument(
        "--option",
        type=str,
        default="no",
        help="The name of the option to use in training. \
            The possible options are in \
            [no, no_kc, no_option, yes]. \
            The default option is no."
    )
    args = parser.parse_args()

    main(args.model_name, args.option ,args.dataset_name, args.method_name)
