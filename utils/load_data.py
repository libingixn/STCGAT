# -*- coding: utf-8 -*-
# @Project: model_X
# @Author  : shiqiZhang
import os
import numpy as np
import argparse
import torch
from torch.utils.data import DataLoader


def load_st_dataset(dataset):
    # output B, N, D
    if dataset == 'PEMSD4':
        data_path = os.path.join('./datasets/PEMSD4/PEMS04.npz')
        data = np.load(data_path)['data'][:, :, 0]  # onley the first dimension, traffic flow data
    elif dataset == 'PEMSD8':
        data_path = os.path.join('./datasets/PEMSD8/PEMS08.npz')
        data = np.load(data_path)['data'][:, :, 0]  # onley the first dimension, traffic flow data
    else:
        raise ValueError
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
    print('Load %s Dataset shaped: ' % dataset, data.shape, data.max(), data.min(), data.mean(), np.median(data))

    return data.transpose(1, 0, 2)


def pre_process_data(data, norm_dim):
    """
    :param data: np.array, original traffic data without normalization.
    :param norm_dim: int, normalization dimension.
    :return:
        norm_base: list, [max_data, min_data], data of normalization base.
        norm_data: np.array, normalized traffic data.
    """
    norm_base = normalize_base(data, norm_dim)  # find the normalize base
    norm_data = normalize_data(norm_base[0], norm_base[1], data)  # normalize data

    return norm_base, norm_data


def normalize_base(data, norm_dim):
    """
    :param data: np.array, original traffic data without normalization.
    :param norm_dim: int, normalization dimension.
    :return:
        max_data: np.array
        min_data: np.array
    """
    max_data = np.max(data, norm_dim, keepdims=True)  # [N, T, D] , norm_dim=1, [N, 1, D]
    min_data = np.min(data, norm_dim, keepdims=True)
    return max_data, min_data


def normalize_data(max_data, min_data, data):
    """
    :param max_data: np.array, max data.
    :param min_data: np.array, min data.
    :param data: np.array, original traffic data without normalization.
    :return:
        np.array, normalized traffic data.
    """
    mid = min_data
    base = max_data - min_data
    normalized_data = (data - mid) / base

    return normalized_data


def recover_data(max_data, min_data, data):
    """
    :param max_data: np.array, max data.
    :param min_data: np.array, min data.
    :param data: np.array, normalized data.
    :return:
        recovered_data: np.array, recovered data.
    """
    mid = min_data
    base = max_data - min_data

    recovered_data = data * base + mid

    return recovered_data


def to_tensor(data):
    return torch.tensor(data, dtype=torch.float)


def split_data_by_ratio(data, val_ratio, test_ratio):
    data_len = data.shape[0]
    test_data = data[-int(data_len*test_ratio):]
    val_data = data[-int(data_len*(test_ratio+val_ratio)):-int(data_len*test_ratio)]
    train_data = data[:-int(data_len*(test_ratio+val_ratio))]
    return train_data, val_data, test_data


def Add_Window_Horizon(data, window, horizon, single=False):
    '''
    :param data: shape [B, ...]
    :param window:
    :param horizon:
    :return: X is [B, W, ...], Y is [B, H, ...]
    '''
    length = len(data)
    end_index = length - horizon - window + 1
    X = []      #windows
    Y = []      #horizon
    index = 0
    if single:
        while index < end_index:
            X.append(data[index:index+window])
            Y.append(data[index+window+horizon-1:index+window+horizon])
            index = index + 1
    else:
        while index < end_index:
            X.append(data[index:index+window])
            Y.append(data[index+window:index+window+horizon])
            index = index + 1
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

def data_loader(X, Y):

    x_ = to_tensor(X)  # [N, H, D]
    y_ = to_tensor(Y)  # [N, 1]

    dataloader = torch.utils.data.TensorDataset(x_.transpose(1, 2), y_.transpose(1, 2))

    return dataloader


def get_dataloader(dataset, split_ratio, lag, horizon):
    # load raw st dataset
    data_Set = load_st_dataset(dataset)

    flow_norm, flow_data = pre_process_data(data=data_Set, norm_dim=1)
    data = flow_data.transpose(1, 0, 2)

    data_train, data_val, data_test = split_data_by_ratio(data, split_ratio[2], split_ratio[1])


    x_tra, y_tra = Add_Window_Horizon(data_train, lag, horizon, single=False)
    x_val, y_val = Add_Window_Horizon(data_val, lag, horizon, single=False)
    x_test, y_test = Add_Window_Horizon(data_test, lag, horizon, single=False)

    print("train_x:{}, train_y:{}".format(x_tra.shape, y_tra.shape))
    print("val_x:{}, val_y:{}".format(x_val.shape, y_val.shape))
    print("test_x:{}, test_y:{}".format(x_test.shape, y_test.shape))
    train_data = data_loader(x_tra, y_tra)

    val_data = data_loader(x_val, y_val)

    test_data = data_loader(x_test, y_test)

    return train_data, val_data, test_data, flow_norm



if __name__ == '__main__':
    # MetrLA 207; BikeNYC 128; SIGIR_solar 137; SIGIR_electric 321
    DATASET = 'SIGIR_electric'
    if DATASET == 'MetrLA':
        NODE_NUM = 207
    elif DATASET == 'BikeNYC':
        NODE_NUM = 128
    elif DATASET == 'SIGIR_solar':
        NODE_NUM = 137
    elif DATASET == 'SIGIR_electric':
        NODE_NUM = 321
    parser = argparse.ArgumentParser(description='PyTorch dataloader')
    parser.add_argument('--dataset', default=DATASET, type=str)
    parser.add_argument('--num_nodes', default=NODE_NUM, type=int)
    parser.add_argument('--val_ratio', default=0.2, type=float)
    parser.add_argument('--test_ratio', default=0.2, type=float)
    parser.add_argument('--lag', default=12, type=int)
    parser.add_argument('--horizon', default=12, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    args = parser.parse_args()

    train_data, val_data, test_data = get_dataloader(dataset="PEMSD4",
                   split_ratio=[0.6, 0.2, 0.2], lag=12,
                   horizon=12)

    train_loader = DataLoader(train_data, batch_size=64, shuffle=False, drop_last=False, num_workers=2)

    for data in train_loader:
        print("checkpoint:", data[0].shape)

        print("y:", data[1].shape)