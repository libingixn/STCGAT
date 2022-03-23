# -*- coding: utf-8 -*-
# @Project: STCGAT
# @Author  : shiqiZhang
import os
import csv
import torch
import numpy as np
from torch.utils.data import Dataset
import scipy.sparse as sp


def get_adjacent_matrix(distance_file: str, num_nodes: int, id_file: str = None, graph_type="distance")->np.array:

    """
    :param distance_file: str, path of csv file to save the distances between nodes.
    :param num_nodes: int, number of nodes in the graph
    :param id_file: str, path of txt file to save the order of the nodes.
    :param graph_type: str, ["connect", "distance"]
    :return:
        np.array(N, N)
    """
    data_path = None
    if distance_file == 'PEMSD4':
        data_path = os.path.join('./datasets/PEMSD4/PEMS04.csv')
    elif distance_file == 'PEMSD8':
        data_path = os.path.join('./datasets/PEMSD8/PEMS08.csv')

    A = np.zeros([int(num_nodes), int(num_nodes)])

    if id_file:
        with open(id_file, "r") as f_id:
            node_id_dict = {int(node_id): idx for idx, node_id in enumerate(f_id.read().strip().split("\n"))}

            with open(data_path, "r") as f_d:
                f_d.readline()
                reader = csv.reader(f_d)
                for item in reader:
                    if len(item) != 3:
                        continue
                    i, j, distance = int(item[0]), int(item[1]), float(item[2])
                    if graph_type == "connect":
                        A[node_id_dict[i], node_id_dict[j]] = 1.
                        A[node_id_dict[j], node_id_dict[i]] = 1.
                    elif graph_type == "distance":
                        A[node_id_dict[i], node_id_dict[j]] = 1. / distance
                        A[node_id_dict[j], node_id_dict[i]] = 1. / distance
                    else:
                        raise ValueError("graph type is not correct (connect or distance)")
        return A

    with open(data_path, "r") as f_d:
        f_d.readline()  # 跳过第一行
        reader = csv.reader(f_d)
        for item in reader:
            if len(item) != 3:
                continue
            i, j, distance = int(item[0]), int(item[1]), float(item[2])

            if graph_type == "connect":
                A[i, j], A[j, i] = 1., 1.
            elif graph_type == "distance":
                A[i, j] = 1. / distance
                A[j, i] = 1. / distance
            else:
                raise ValueError("graph type is not correct (connect or distance)")

    return A


def preprocess_grap_A(adj):
    # _A = A + I
    _adj = adj + sp.eye(adj.shape[0])
    # _dsqp
    _dsep = _adj.sum(1).A1

    _D_half = sp.diags(np.power(_dsep, -0.5))

    adj_normalized = _D_half @ _adj @ _D_half
    return adj_normalized


def preprocess_grap(adj):
    N = adj.size(0)
    matrix_i = torch.eye(N, dtype=torch.float)
    adj += matrix_i  # [N, N]   A+I

    degree_matrix = torch.sum(adj, dim=1, keepdim=False)  # [N],
    degree_matrix = degree_matrix.pow(-1)
    degree_matrix[degree_matrix == float("inf")] = 0.

    degree_matrix = torch.diag(degree_matrix)  # Convert to diagonal matrix

    return torch.mm(degree_matrix, adj)  # \hat A=D^(-1) * A ,This is equivalent to \hat A = D_{-1/2}*A*D_{-1/2}


def get_flow_data(flow_file: str) -> np.array:
    """
    :param flow_file: str, path of .npz file to save the traffic flow data
    :return:
        np.array(N, T, D)
    """
    data = np.load(flow_file)

    flow_data = data['data'].transpose([1, 0, 2])[:, :, 0][:, :, np.newaxis]

    return flow_data


class LoadData(Dataset):
    def __init__(self, data_path, Split_rate, history_length, predict_length, train_mode):
        """
        :param data_path: list, ["graph file name" , "flow data file name"], path to save the data file names.
        :param num_nodes: int, number of nodes.
        :param divide_days: list, [ days of train data, days of test data], list to divide the original data.
        :param time_interval: int, time interval between two traffic data records (mins).
        :param history_length: int, length of history data to be used.
        :param train_mode: list, ["train", "test"].
        """

        self.data_path = data_path
        self.train_mode = train_mode
        self.train_rate = Split_rate[0]
        self.test_rate = Split_rate[1]
        self.history_length = history_length
        self.predict_length = predict_length
        self.flow_norm, self.flow_data = self.pre_process_data(data=get_flow_data(data_path), norm_dim=1)

    def __len__(self):
        """
        :return: length of dataset (number of samples).
        """
        if self.train_mode == "train":
            return int(self.flow_data.shape[1] * self.train_rate) - self.history_length - self.predict_length + 1
        elif self.train_mode == "val":
            return int(self.flow_data.shape[1] * self.test_rate) - self.history_length - self.predict_length + 1
        elif self.train_mode == "test":
            return int(self.flow_data.shape[1]) - int(self.flow_data.shape[1] * (self.train_rate + self.test_rate)) - self.history_length - self.predict_length + 1
        else:
            raise ValueError("train mode: [{}] is not defined".format(self.train_mode))

    def __getitem__(self, index):  # (x, y), index = [0, L1 - 1]
        """
        :param index: int, range between [0, length - 1].
        :return:
            graph: torch.tensor, [N, N].
            data_x: torch.tensor, [N, H, D].
            data_y: torch.tensor, [N, 1, D].
        """
        if self.train_mode == "train":
            index = index
        elif self.train_mode == "val":
            index += int(self.flow_data.shape[1] * self.train_rate)
        elif self.train_mode == "test":
            index += (int(self.flow_data.shape[1] * (self.train_rate + self.test_rate)))
        else:
            raise ValueError("train mode: [{}] is not defined".format(self.train_mode))

        data_x, data_y = LoadData.slice_data(self.flow_data, self.history_length, self.predict_length, index)

        data_x = LoadData.to_tensor(data_x)  # [N, H, D]
        data_y = LoadData.to_tensor(data_y)  # [N, 1]

        # 处理邻接矩阵 graph
        # graph = preprocess_grap_A(get_adjacent_matrix(distance_file=self.data_path[0], num_nodes=data_x.shape[0]))
        # return {"graph": LoadData.to_tensor(graph), "flow_x": data_x, "flow_y": data_y}

        # graph = preprocess_grap_B(LoadData.to_tensor(
        #     get_adjacent_matrix(distance_file=self.data_path[0], num_nodes=data_x.shape[0])))
        # return {"graph": graph, "flow_x": data_x, "flow_y": data_y}

        return {"flow_x": data_x, "flow_y": data_y}

    @staticmethod
    def slice_data(data, history_length, predict_length, index):
        """
        :param data: np.array, normalized traffic data.
        :param history_length: int, length of history data to be used.
        :param index: int, index on temporal axis.
        :param train_mode: str, ["train", "test"].
        :return:
            data_x: np.array, [N, H, D].
            data_y: np.array [N, D].
        """

        start_index = index
        end_index = index + history_length + predict_length

        data_w = data[:, start_index: end_index]
        data_x, data_y = np.split(data_w, 2, axis=1)

        return data_x, data_y

    @staticmethod
    def pre_process_data(data, norm_dim):
        """
        :param data: np.array, original traffic data without normalization.
        :param norm_dim: int, normalization dimension.
        :return:
            norm_base: list, [max_data, min_data], data of normalization base.
            norm_data: np.array, normalized traffic data.
        """
        norm_base = LoadData.normalize_base(data, norm_dim)  # find the normalize base
        norm_data = LoadData.normalize_data(norm_base[0], norm_base[1], data)  # normalize data

        return norm_base, norm_data

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def to_tensor(data):
        return torch.tensor(data, dtype=torch.float)


if __name__ == '__main__':
    train_data = LoadData(data_path=["../datasets/PEMS04/PEMS04.csv", "../datasets/PEMS04/PEMS04.npz"], Split_rate=[0.6, 0.2],
                          history_length=12, predict_length=12, train_mode="train")

    val_data = LoadData(data_path=["../datasets/PEMS04/PEMS04.csv", "../datasets/PEMS04/PEMS04.npz"], Split_rate=[0.6, 0.2],
                        history_length=12, predict_length=12, train_mode="val")

    test_data = LoadData(data_path=["../datasets/PEMS04/PEMS04.csv", "../datasets/PEMS04/PEMS04.npz"], Split_rate=[0.6, 0.2],
                         history_length=12, predict_length=12, train_mode="test")
    print("------------------------------------------------------")
    print(len(train_data))
    print(train_data[0]["flow_x"].size())
    print(train_data[0]["flow_y"].size())

    print(len(val_data))
    print(val_data[0]["flow_x"].size())
    print(val_data[0]["flow_y"].size())

    print(len(test_data))
    print(test_data[0]["flow_x"].size())
    print(test_data[0]["flow_y"].size())
