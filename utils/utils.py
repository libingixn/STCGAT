# -*- coding: utf-8 -*-
# @Project: trafficPrediction
# @Author  : shiqiZhang
import os
import matplotlib.pyplot as plt
import numpy as np
import h5py
import torch
import xlwt
from utils.metrics import r2


def create_dir_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)
        # os.mkdir(path)


class Evaluation(object):
    def __init__(self):
        pass

    @staticmethod
    def mae_(target, output):
        return np.mean(np.abs(target - output))

    @staticmethod
    def mape_(target, output):
        return np.mean(np.abs(target - output) / (target + 5))

    @staticmethod
    def rmse_(target, output):
        return np.sqrt(np.mean(np.power(target - output, 2)))

    @staticmethod
    def mse_(target, output):
        return np.sum(((target - output)**2)/len(target))

    @staticmethod
    def smape(y_true, y_pred):
        return 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))) * 100

    @staticmethod
    def total(target, output):
        mae = Evaluation.mae_(target, output)
        # mape = Evaluation.mape_(target, output)
        smape = Evaluation.smape(target, output)
        rmse = Evaluation.rmse_(target, output)

        mse = Evaluation.mse_(target, output)
        var = np.var(target)
        R2 = 1 - mse/var

        return mae, smape, rmse, R2

    @staticmethod
    def total_3(label, pred):
        # mae = Evaluation.mask_mae_(pred, label, 0)
        # rmse = np.square(mae)

        with np.errstate(divide='ignore', invalid='ignore'):
            mask = np.not_equal(label, 0)
            mask = mask.astype(np.float32)
            mask /= np.mean(mask)
            mae = np.abs(np.subtract(pred, label)).astype(np.float32)
            # smape = np.abs(label - pred) / (np.abs(pred) + np.abs(label))
            rmse = np.square(mae)
            mape = np.divide(mae, label)
            mae = np.nan_to_num(mae * mask)

            # smape = np.nan_to_num(smape * mask)

            # wape = np.divide(np.sum(mae), np.sum(label))
            mae = np.mean(mae)

            # smape = 2.0 * np.mean(smape) * 100

            rmse = np.nan_to_num(rmse * mask)
            rmse = np.sqrt(np.mean(rmse))
            mape = np.nan_to_num(mape * mask)
            mape = np.mean(mape)
            r2_new = r2(label, pred)
        return mae, rmse, mape, r2_new


def visualize_result(h5_file, nodes_id, time_se, visualize_file):
    file_obj = h5py.File(h5_file, "r") # 获得文件对象，这个文件对象有两个keys："predict"和"target"
    prediction = file_obj["predict"][:][:, :, 0]  # [N, T],切片，最后一维取第0列，所以变成二维了，要是[:, :, :1]那么维度不会缩减
    target = file_obj["target"][:][:, :, 0]  # [N, T],同上
    file_obj.close()

    plot_prediction = prediction[nodes_id][time_se[0]: time_se[1]]  # [T1]，将指定节点的，指定时间的数据拿出来
    plot_target = target[nodes_id][time_se[0]: time_se[1]]  # [T1]，同上

    # workbook = xlwt.Workbook(encoding='utf-8')
    # worksheet = workbook.add_sheet('My Worksheet')
    # worksheet.write(0, 0, "target")
    # worksheet.write(0, 1, "prediction")
    # for i in range(plot_target.shape[0]):
    #     worksheet.write(i+1, 0, label=plot_target[i])
    #     worksheet.write(i+1, 1, label=plot_prediction[i])
    #
    # # 保存
    # # workbook.save('PeMS04_single_head.xls')
    # workbook.save('STCGAT_no_Tcn_single_lstm_y_Ls.xls')

    plt.figure(figsize=(15,5))
    plt.grid(True, linestyle="-.", linewidth=0.5)
    plt.plot(np.array([t for t in range(time_se[1] - time_se[0])]), plot_prediction, ls="-", marker=" ", color="r")
    plt.plot(np.array([t for t in range(time_se[1] - time_se[0])]), plot_target, ls="-", marker=" ", color="b")

    plt.legend(["prediction", "target"], loc="upper right", fontsize=9)

    plt.xlabel("Time", fontsize=15)
    plt.ylabel("Traffic speed", fontsize=15)

    plt.axis([0, time_se[1] - time_se[0],
              np.min(np.array([np.min(plot_prediction), np.min(plot_target)])),
              np.max(np.array([np.max(plot_prediction), np.max(plot_target)]))])

    plt.savefig(visualize_file + ".svg")

