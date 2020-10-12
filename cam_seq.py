#!/usr/bin/env python
# coding: utf-8

"""
Protein - ligand binding residue prediction from sequence 
using Class Activation Maps

This script allows the user to train a neural network to predict binding 
classes for a protein sequence from scratch or load a network from disk. 
Then Class Activation Maps are calculated to also predict for each residue of 
the input protein sequence which classes it binds.

The CAM calculations are based on https://github.com/zhoubolei/CAM 
(the implementation of the paper http://cnnlocalization.csail.mit.edu/)

If you want to use this tool with own data, a path to a folder is required as 
argument, which contains all input files like in the example_input_data folder.

This script requires that all libraries listed in the requirements.txt are 
installed within the Python environment you are running this script in.

"""

import numpy as np
import os
import os.path
import sys
import datetime
import torch
import torch.nn.init as init
from torch import nn
from torch import optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from Bio import SeqIO
import argparse
from collections.abc import Iterable


def aa_to_onehot(data):
    """Converts an array of aminoacids with length len into a onehot encoded
    matrix of format (len, 20)

    Parameters
    ----------
    data : array
            The aminoacid sequence of length len to be converted

    Returns
    -------
    onehot
            a matrix with format (len,20) containing only zeros and ones
    """
    seq = data.seq
    id_ = data.id
    alphabet = [
        "A",
        "R",
        "N",
        "D",
        "C",
        "E",
        "Q",
        "G",
        "H",
        "I",
        "L",
        "K",
        "M",
        "F",
        "P",
        "S",
        "T",
        "W",
        "Y",
        "V",
        "X",
        "U",
    ]
    int_dict = {alphabet[i]: i for i in range(0, 22)}
    int_seq = np.array([int_dict[aa] for aa in seq])
    onehot = list()
    for value in int_seq:
        letter = [0 for _ in range(len(alphabet))]
        letter[value] = 1
        onehot.append(letter)
    onehot = np.array(onehot)
    onehot = onehot[:, 0:20]  # remove U and X
    return onehot


def train_valid_split(protein_ids, x_splits, y_splits, valid_split):
    """Makes x_train, x_valid, y_train, y_valid arrays from dictionaries with
    5 splits and a chosen valid_split

    Parameters
    ----------
    protein_ids : array
            array containing an array of the order of the protein ids for each
            of the 5 splits
    x_splits : array
            array containing an array of the onehot-encoded protein sequence
            for each of the 5 splits with 215 proteins each
    y_splits : array
            array containing an array of the onehot-encoded binding classes
            for each of the 5 splits with 215 proteins each
    valid_split : int
            The split chosen to be the valid data, has to be in [0,1,2,3,4]

    Returns
    -------
    data
            a dictionary with six entries ids_train, ids_valid, x_train,
            x_valid, y_train, y_valid (valid containing data for 215 proteins
            and train for 860 proteins)
    """
    y_train = []
    x_train = []
    ids_train = []
    y_valid = []
    x_valid = []
    ids_valid = []
    for split in range(0, valid_split):
        y_train.extend(y_splits[split])
        x_train.extend(x_splits[split])
        ids_train.extend(protein_ids[split])
    y_valid.extend(y_splits[valid_split])
    x_valid.extend(x_splits[valid_split])
    ids_valid.extend(protein_ids[valid_split])
    for split in range(valid_split + 1, 5):
        y_train.extend(y_splits[split])
        x_train.extend(x_splits[split])
        ids_train.extend(protein_ids[split])
    print(
        "Split",
        valid_split,
        "train (",
        len(x_train),
        ") and valid (",
        len(x_valid),
        ") data was loaded sucessfully.",
    )
    return {
        "ids_train": ids_train,
        "ids_valid": ids_valid,
        "x_train": np.array(x_train),
        "x_valid": np.array(x_valid),
        "y_train": np.array(y_train),
        "y_valid": np.array(y_valid),
    }


def load_train_valid_data(data_path):
    """Load 5 data splits from path for train and valid

    Parameters
    ----------
    data_path : Path
            location of the input data file containing a mapping of protein_id
            to the binding classes, a file containing a mapping of protein_id
            to the amino acid sequence and the files containing a list of
            protein_id for each of the four classes

    Returns
    -------
    protein_ids
            array containing an array of the order of the protein ids for each
            of the 5 splits
    x_splits
            array containing an array of the onehot-encoded protein sequence
            for each of the 5 splits with 215 proteins each
    y_splits
            array containing an array of the onehot-encoded binding classes
            for each of the 5 splits with 215 proteins each
    """
    y_file = os.path.join(data_path, "binary_binding_train.txt")
    txt_y = open(y_file)
    y_dict = {}
    for line in txt_y:
        [id_, y] = line.split()
        y_new = np.array([0, 0, 0, 0])
        if "m" in y:
            y_new += [1, 0, 0, 0]
        if "s" in y:
            y_new += [0, 1, 0, 0]
        if "n" in y:
            y_new += [0, 0, 1, 0]
        if "p" in y:
            y_new += [0, 0, 0, 1]
        y_dict[id_] = y_new
    protein_ids = [[] for i in range(0, 5)]
    y_splits = [[] for i in range(0, 5)]
    x_splits = [[] for i in range(0, 5)]
    input_file = os.path.join(data_path, "uniprot_train.fasta")
    seq_dict = SeqIO.to_dict(SeqIO.parse(input_file, "fasta"))
    for split in range(0, 5):
        split_file = os.path.join(data_path, ("ids_split" + str(split + 1) + ".txt"))
        txt_split = open(split_file)
        for i, txt_line in enumerate(txt_split):
            id_ = txt_line.rstrip("\n")
            protein_ids[split].append(id_)
            x_splits[split].append(aa_to_onehot(seq_dict[id_]))
            y_splits[split].append(y_dict[id_])
        txt_split.close()
    return protein_ids, x_splits, y_splits


def pad_set(set_, max_length):
    """Pad an array to max_length and code the padding in a new extra column

    Parameters
    ----------
    set_ : array
            Array containing one-hot encoded protein sequences to be padded
            in format (len, 20)
    max_length : int
            The length to be padded to, has to be bigger than the longest
            protein sequence

    Returns
    -------
    padded_set
            array of one-hot encoded padded protein sequences in format
            (max_length, 21) with padding column (1 = data and 0 = padding)
    """
    aa_len = 20
    padded_set = np.empty([len(set_), max_length, aa_len + 1], dtype=np.float16)
    for i, protein in enumerate(set_):
        diff = max_length - len(protein)
        new_set = np.append(protein, np.zeros((diff, aa_len))).reshape(
            max_length, aa_len
        )  # now (max_length, 20)
        pad_mask = np.append(np.ones(len(protein)), np.zeros(diff))
        padded_set[i] = np.hstack(
            (new_set, pad_mask[:, np.newaxis])
        )  # now (max_length, 21)
    return padded_set


def load_testset(data_path, max_length, classes):
    """Load x_test, y_test, x_seq_dict, y_test_bs test data from path

    Parameters
    ----------
    data_path : Path
            location of the input data file containing a mapping of protein_id
            to the binding classes, a file containing a mapping of protein_id
            to the amino acid sequence and the files containing a mapping of
            protein_id to the binding residues for each of the four classes
    max_length : int
            length to which the input sequences should be padded
    classes : array
            the classes that should be predicted

    Returns
    -------
    ids_test
            array containing an array of the order of the protein ids
    x_test
            array containing an array of the onehot-encoded padded protein
            sequence
    y_test
            array containing an array of the onehot-encoded binding classes
    x_seq_dict
            dictionary containing an array of the protein sequence for a
            protein_id
    y_test_bs
            dictionary containing a dictionary of arrays of onehot-encoded
            binding residues per protein_id per class where 1 means binding
    """
    # load y_test in form [[0,1,1,0]]
    y_file = os.path.join(data_path, "binary_binding_test.txt")
    txt_y = open(y_file, "r")
    y_test = {}
    for idx, line in enumerate(txt_y):
        [protein_id, y] = line.split()
        y_new = np.array([0, 0, 0, 0])
        if "m" in y:
            y_new += [1, 0, 0, 0]
        if "s" in y:
            y_new += [0, 1, 0, 0]
        if "n" in y:
            y_new += [0, 0, 1, 0]
        if "p" in y:
            y_new += [0, 0, 0, 1]
        y_test[protein_id] = y_new
    # load x_test in form [[0,0,1,0,...,0,1]]
    input_file = os.path.join(data_path, "uniprot_test.fasta")
    seq_dict = SeqIO.to_dict(SeqIO.parse(input_file, "fasta"))
    x_test = [0] * len(seq_dict)
    for idx, protein_id in enumerate(y_test):
        x_test[idx] = pad_set([aa_to_onehot(seq_dict[protein_id])], max_length).reshape(
            (1000, 21)
        )
    # load y_test_bs in form {{[0,1,...,0]}}
    y_test_bs = {}
    for class_idx in range(len(classes)):
        label = open(
            os.path.join(data_path, f"binding_residues_{classes[class_idx]}.txt")
        )
        bs_labels = {}
        for line in label:
            line = line.rstrip("\n")
            protein_id, bs = line.split("\t")
            bs_labels[protein_id] = bs
        for idx, protein_id in enumerate(
            seq_dict
        ):  # only use protein_ids for which we have sequence
            x_AA_seq = seq_dict[protein_id][0]
            y_test_bs[idx] = y_test_bs.get(idx, {})
            if protein_id in bs_labels:
                bs = bs_labels[protein_id]
                bs = [int(b) if b != "" else None for b in bs.split(",")]
                new_bs = np.zeros(len(x_AA_seq))
                new_bs = [1 if i in bs else 0 for i in range(len(new_bs))]
                y_test_bs[idx][class_idx] = y_test_bs[idx].get(class_idx, [])
                y_test_bs[idx][class_idx] = np.array(new_bs)
            else:
                y_test_bs[idx][class_idx] = y_test_bs[idx].get(class_idx, [])
                y_test_bs[idx][class_idx] = np.zeros(len(x_AA_seq), dtype=np.int)
    ids_test = list(seq_dict.keys())
    return ids_test, x_test, list(y_test.values()), seq_dict, y_test_bs


# # Training


def calc_weights(k_data, device):
    """Calculate the class weights to better handle class imbalance

    Parameters
    ----------
    k_data : array
            Array containing a dictionary of x_train, x_valid, y_train,
            y_valid per split
    device : torch device
            Device to send torch Tensor to, cuda or cpu

    Returns
    -------
    weight
            array containing a weight for each class
    """
    y_train = k_data[0]["y_train"]
    P = y_train.sum(axis=0)
    N = y_train.shape[0] - P
    sum_P = P.sum(axis=0)
    weight = torch.Tensor(np.array(sum_P / P)).to(device)
    return weight


def calc_pos_weights(k_data, device):
    """Calculate weights to better handle imbalance between positive and
    negatives within each class

    Parameters
    ----------
    k_data : array
            Array containing a dictionary of x_train, x_valid, y_train,
            y_valid per split
    device : torch device
            Device to send torch Tensor to, cuda or cpu

    Returns
    -------
    weight
            array containing a weight for each class
    """
    y_train = k_data[0]["y_train"]
    P = y_train.sum(axis=0)
    N = y_train.shape[0] - P
    weight = N / P
    weight = torch.Tensor(weight).to(device)
    return weight


class MyDataset(Dataset):
    """Dataset class to easily iterate over x and y data"""

    def __init__(self, protein_ids, x_data, y_data):
        self.protein_ids = protein_ids
        self.x_data = x_data
        self.y_data = y_data

    def __len__(self):
        return len(self.x_data)

    """
    Parameters
    ----------
    idx : int 
        index for the item to get from x and y data
        
    Returns
    -------
    sample
        array containing x in format (21, 1, max_length) and y in format (4,)
    """

    def __getitem__(self, idx):
        sample = [
            self.protein_ids[idx],
            torch.from_numpy(self.x_data[idx].T).float().unsqueeze(1),
            self.y_data[idx].astype(float),
        ]
        return sample


def train(train_loader, model, criterion, optimizer, device, weights):
    """Trains the model with the given training data for one epoch and
    calculates loss with given class weights

    Parameters
    ----------
    train_loader : DataLoader
            train data
    model : Module object
            architecture to train
    criterion : _WeightedLoss object
            criterion to calculate the loss
    optimizer : Optimizer object
            optimizer for updating parameters based on the computed gradients
    device : torch device
            device to send torch Tensor to, cuda or cpu
    weights : array
            array containing a weight for each class

    Returns
    -------
    loss_avg
            The average loss for all proteins in the trainset
    """
    loss_sum = 0
    all_y_pred = {}
    all_y = {}
    for (protein_id, x, y) in train_loader:
        protein_id = protein_id[0]
        all_y[protein_id] = y.numpy()
        y = y.to(device).float()
        x = x.to(device)
        model.train()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss = (loss * weights).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_sum += loss.item()
        sigmoid = nn.Sigmoid()
        y_pred_sigmoid = sigmoid(y_pred).squeeze().tolist()
        all_y_pred[protein_id] = y_pred_sigmoid
    loss_avg = loss_sum / len(train_loader)
    return loss_avg


def valid(valid_loader, model, criterion, device, weights):
    """Evaluates the model with the given validation data for one epoch and
    calculates loss with given class weights

    Parameters
    ----------
    valid_loader : DataLoader
            valid data
    model : Module object
            architecture to calculate validation performance on
    criterion : _WeightedLoss object
            criterion to calculate the loss
    device : torch device
            device to send torch Tensor to, cuda or cpu
    weights : array
            array containing a weight for each class

    Returns
    -------
    loss_avg
            The average loss for all proteins in the validset
    """
    loss_sum = 0
    all_y_pred = {}
    all_y = {}
    with torch.no_grad():
        for (protein_id, x, y) in valid_loader:
            protein_id = protein_id[0]
            all_y[protein_id] = y.numpy()
            y = y.to(device).float()
            x = x.to(device)
            model.eval()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss = (loss * weights).mean()
            loss_sum += loss.item()
            sigmoid = nn.Sigmoid()
            y_pred_sigmoid = sigmoid(y_pred).squeeze().tolist()
            all_y_pred[protein_id] = y_pred_sigmoid
    loss_avg = loss_sum / len(valid_loader)
    return loss_avg


class SaveFeatures:
    """Register forward hook to save activation features, called for final
    convolutional layer for Class Activation Mapping"""

    act_features = None

    def __init__(self, layer):
        self.hook = layer.register_forward_hook(self.save_features)

    def save_features(self, module, input, output):
        self.act_features = ((output.cpu()).data).numpy()

    def remove_hook(self):
        self.hook.remove()


def calc_CAMs(x, model, classes):
    """Calculate Class Activation Maps (CAMs) for the model for input data x

    Parameters
    ----------
    x : array
            input data in shape (1, 21, 1, max_length)
    model : Module object
            architecture to calculate CAM with
    classes : array
            the classes that should be predicted

    Returns
    -------
    y_pred
            array containing an array of the predicted onehot-encoded protein
            binding classes (before sigmoid)
    nonnorms
            array of non-normalized CAMs directly from network
    norms
            array of normalized CAMs with minimum 0 and maximum 1
    """
    model.eval()
    with torch.no_grad():
        final_layer = model._modules.get("conv")  # last convolutional layer
        activated_features = SaveFeatures(final_layer)
        y_pred = model(x)
        activated_features.remove_hook()
        weight_softmax_params = list(
            model._modules.get("fc").parameters()
        )  # out parameters after average pooling
        weight_softmax = np.squeeze(weight_softmax_params[0].cpu().data.numpy())
        norms = []
        nonnorms = []
        len_ = int(
            x[0][20].cpu().sum().numpy()
        )  # residues where padding column 1 = length of non-padded protein
        feature_conv = activated_features.act_features
        for class_idx in range(len(classes)):
            _, nc, h, w = feature_conv.shape
            nonnorm_CAM = np.dot(
                weight_softmax[class_idx], feature_conv.reshape((nc, h * w))
            )
            nonnorm_CAM = nonnorm_CAM[:len_]
            cam = nonnorm_CAM - np.min(nonnorm_CAM)
            norm_CAM = cam / np.max(cam)
            norms.append(norm_CAM)
            nonnorms.append(nonnorm_CAM)
        return y_pred, nonnorms, norms


def test(test_loader, model, criterion, device, weights, classes):
    """Evaluates the model with the given test data for one epoch and
    calculates loss with given class weights

    Parameters
    ----------
    test_loader : DataLoader
            test data
    model : Module object
            architecture to calculate test performance on
    criterion : _WeightedLoss object
            criterion to calculate the loss
    device : torch device
            device to send torch Tensor to, cuda or cpu
    weights : array
            array containing a weight for each class
    classes : array
            the classes that should be predicted

    Returns
    -------
    all_y_pred
            dictionary containing an array of the predicted onehot-encoded
            protein binding classes per protein_id
    y_pred_bs_both
            array of residue-level predictions (multiplication of normalized
            residue level prediction from CAM with protein level prediction)
    loss_avg
            The average loss for all proteins in the testset
    """
    loss_sum = 0
    all_y_pred = {}
    all_y = {}
    y_pred_bs = {}
    y_pred_bs_norm = {}
    y_pred_bs_both = {}
    with torch.no_grad():
        for (protein_id, x, y) in test_loader:
            protein_id = protein_id[0]
            all_y[protein_id] = y.numpy()
            y = y.to(device).float()
            x = x.to(device)
            model.eval()
            (
                y_pred,
                y_pred_bs[protein_id],
                y_pred_bs_norm[protein_id],
            ) = calc_CAMs(x, model, classes)
            loss = criterion(y_pred, y)
            loss = (loss * weights).mean()
            loss_sum += loss.item()
            sigmoid = nn.Sigmoid()
            y_pred_sigmoid = sigmoid(y_pred).squeeze().tolist()
            all_y_pred[protein_id] = y_pred_sigmoid
            y_pred_bs_both[protein_id] = [
                y_pred_bs_norm[protein_id][class_id] * y_pred_sigmoid[class_id]
                for class_id, class_ in enumerate(classes)
            ]
    loss_avg = loss_sum / len(test_loader)
    return all_y_pred, y_pred_bs_both, loss_avg


def write_to_file(path, name, current_time, comments, preds, classes):
    """Write model predictions to file with a seperate line for protein_id
    metal small nuclear peptide values each

    Parameters
    ----------
    path : Path
            location where the predictions should be saved
    name : Module object
            architecture to calculate CAM with
    current_time : string
            current time to name file
    comments : string
            information about the model parameters to write into file
    preds : array
            array containing an array of the predicted onehot-encoded protein
            binding classes
    classes : array
            the classes that should be predicted
    """
    file = os.path.join(path, (f"{current_time}_{name}.txt"))
    if not os.path.exists(file):
        with open(file, "a") as f:
            f.write(
                (
                    f"{comments}, format: protein_id "
                    f'{" ".join(str(c) for c in classes)}\n'
                )
            )
    with open(file, "a") as f:
        for protein_id, protein_preds in preds.items():
            f.write(f"{protein_id}")
            for class_id, class_ in enumerate(classes):
                if isinstance(protein_preds[class_id], Iterable):
                    data = ",".join(str(val) for val in protein_preds[class_id])
                else:
                    data = protein_preds[class_id]
                f.write(f"\n{data}")
            f.write("\n")


class MyArchitecture(nn.Module):
    """Network architecture with 3 convolutional layers (kernelsize 5 and
    nr_hidden channels 250) using ELU and Dropout, 1 Global Average Pooling
    layer and 1 Fully Connected layer"""

    def __init__(self):
        super().__init__()
        nr_channels, hidden, output = 21, 250, 4
        kernel = (1, 5)
        pad = (0, 2)
        self.conv_1 = nn.Sequential(
            nn.Conv2d(nr_channels, hidden, kernel_size=kernel, stride=1, padding=pad),
            nn.ELU(),
            nn.Dropout2d(),
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=kernel, stride=1, padding=pad),
            nn.ELU(),
            nn.Dropout2d(),
        )
        self.conv = nn.Conv2d(hidden, hidden, kernel_size=kernel, stride=1, padding=pad)
        self.aavg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(hidden, output)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv(x)
        x = self.aavg(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def main(logging, log_path, pretrained, model_path, input_path, batchsize):
    """Train a model on protein sequence data and predict whether the protein
    is binding for the classes metal, small, nuclear, peptide. Use the
    generated Class Activation Map to extract residue binding prediction.

    Parameters
    ----------
    logging : boolean
            whether to save model and predictions to files
    log_path : Path
            given logging is True: the folder where to save model and
            predictions
    pretrained : boolean
            whether to load a model from a file or to train a model from
            scratch using input data
    model_path : Path
            given pretrained is True: the folder from which the model should
            be loaded
    """

    classes = ["metal", "small", "nuclear", "peptide"]
    max_length = 1000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr = 0.001

    if pretrained:
        print("Load model from disk.")
        model = MyArchitecture()
        checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        nr_hyperparams = sum(
            [
                np.prod(p.size())
                for p in filter(lambda p: p.requires_grad, model.parameters())
            ]
        )
        weights = torch.Tensor([2.9191, 2.0739, 11.0989, 11.7442]).to(device)
        pos_weight = torch.Tensor([1.4855, 0.7659, 8.4505, 9.0000]).to(device)
        criterion = torch.nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epochs = checkpoint["epoch"]
        dt = str(datetime.datetime.now())
        curr_time = f"{dt[5:10]}-{str(int(dt[11:13])+2)}-{dt[14:16]}-{dt[17:19]}"
        comments = (
            f"{curr_time}: lr={lr}, weights={weights.cpu()},"
            f" {epochs} epochs, {nr_hyperparams} hyperparams"
        )
    else:
        k_data = {}
        protein_ids, x_splits, y_splits = load_train_valid_data(input_path)
        for k in range(0, 5):
            k_data[k] = train_valid_split(protein_ids, x_splits, y_splits, k)
            k_data[k]["x_train"] = pad_set(k_data[k]["x_train"], max_length)
            k_data[k]["x_valid"] = pad_set(k_data[k]["x_valid"], max_length)

        weights = calc_weights(k_data, device)
        pos_weight = calc_pos_weights(k_data, device)
        batch_size_train = batchsize if batchsize else 215
        batch_size_valid = batchsize if batchsize else 215
        epochs = 350

        crossvalidation = False
        if not crossvalidation:
            k_data = [k_data[4]]
        print("Train network from scratch.")
        for k in range(len(k_data)):
            criterion = torch.nn.BCEWithLogitsLoss(
                reduction="none", pos_weight=pos_weight
            )
            model = MyArchitecture()
            nr_hyperparams = sum(
                [
                    np.prod(p.size())
                    for p in filter(lambda p: p.requires_grad, model.parameters())
                ]
            )
            optimizer = optim.Adam(model.parameters(), lr=lr)
            model.to(device)
            dt = str(datetime.datetime.now())
            curr_time = f"{dt[5:10]}-{str(int(dt[11:13])+2)}-{dt[14:16]}-{dt[17:19]}"
            comments = (
                f"{curr_time}: lr={lr}, bs(T/V)={batch_size_train}"
                f"/{batch_size_valid}, weights={weights.cpu()}, {epochs}"
                f" epochs, {nr_hyperparams} hyperparams"
            )
            if logging:
                print(comments)
            train_data = MyDataset(
                k_data[k]["ids_train"],
                k_data[k]["x_train"],
                k_data[k]["y_train"],
            )
            train_loader = DataLoader(train_data, batch_size=batch_size_train)
            valid_data = MyDataset(
                k_data[k]["ids_valid"],
                k_data[k]["x_valid"],
                k_data[k]["y_valid"],
            )
            valid_loader = DataLoader(valid_data, batch_size=batch_size_valid)
            for epoch in range(1, epochs + 1):
                train_loss = train(
                    train_loader, model, criterion, optimizer, device, weights
                )
                valid_loss = valid(valid_loader, model, criterion, device, weights)
            if logging:
                state = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }
                torch.save(state, os.path.join(log_path, (f"{str(curr_time)}_model")))

    print("Calculate CAM on test data.")
    batch_size_test = 1
    ids_test, x_test, y_test, seq_dict, y_bs_test = load_testset(
        input_path, max_length, classes
    )
    test_data = MyDataset(ids_test, x_test, y_test)
    test_loader = DataLoader(test_data, batch_size=batch_size_test)
    y_pred_test, y_pred_bs_test, test_l = test(
        test_loader, model, criterion, device, weights, classes
    )
    if logging:
        write_to_file(
            log_path,
            "test_protein_predictions",
            curr_time,
            comments,
            y_pred_test,
            classes,
        )
        write_to_file(
            log_path,
            "test_residue_predictions",
            curr_time,
            comments,
            y_pred_bs_test,
            classes,
        )
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        help="path to the input data folder (default: /example_input_data)",
    )
    parser.add_argument("-m", "--model", help="load pretrained model from file")
    parser.add_argument("-l", "--log", help="log model and predictions to folder")
    parser.add_argument(
        "-b", "--batchsize", help="set batchsize for training and validation"
    )
    args = parser.parse_args()

    if args.log:
        logging = True
        log_path = args.log
        if not os.path.exists(log_path):
            os.makedirs(log_path)
    else:
        logging = False
        log_path = ""

    if args.input:
        input_path = args.input
    else:
        path = os.path.abspath(os.path.dirname(__file__))
        input_path = os.path.join(path, "example_input_data")
    needed_input_files = [
        "binary_binding_test.txt",
        "binary_binding_train.txt",
        "binding_residues_metal.txt",
        "binding_residues_nuclear.txt",
        "binding_residues_peptide.txt",
        "binding_residues_small.txt",
        "ids_split1.txt",
        "ids_split2.txt",
        "ids_split3.txt",
        "ids_split4.txt",
        "ids_split5.txt",
        "uniprot_test.fasta",
        "uniprot_train.fasta",
    ]
    if not os.path.exists(input_path):
        print("Error: The input folder couldn't be found.")
        exit()
    elif not set(needed_input_files).issubset(os.listdir(path=input_path)):
        print("Error: Not all necessary input files were found in the folder.")
        print(
            (
                f"Necessary files are: "
                f"{', '.join(str(val) for val in needed_input_files)}"
            )
        )
        exit()

    if args.model:
        pretrained = True
        model_path = args.model
        if not os.path.exists(log_path):
            print("Error: The model file couldn't be found.")
            exit()
    else:
        pretrained = False
        model_path = ""
    if args.batchsize:
        batchsize = int(args.batchsize)
    else:
        batchsize = args.batchsize
    main(logging, log_path, pretrained, model_path, input_path, batchsize)
