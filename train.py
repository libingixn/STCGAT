# -*- coding: utf-8 -*-
# @Project: STCGAT
# @Author  : shiqiZhang
# @Time    : 2022/2/26 14:32
# @Phrase: 人生不如意十之八九，Bug是一二。
# -*- coding: utf-8 -*-
# @Project: trafficPrediction
# @Author  : shiqiZhang
# @Time    : 2021/11/24 21:19
# @Phrase: 人生不如意十之八九，Bug是一二。
import glob
import os
import time
import h5py
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
from utils.SoftDTW import SoftDTW
from models.STCGAT import STCGAT
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.prepareData import LoadData, preprocess_grap_A, preprocess_grap_B, get_adjacent_matrix
from utils.utils import Evaluation, visualize_result

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=0, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=8, help='Random seed.')
parser.add_argument('--epochs', type=int, default=12, help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=64, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=6, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=4, help='Number of head attentions.')
parser.add_argument('--history_length', type=int, default=12, help='history_length')
parser.add_argument('--predict_length', type=int, default=12, help='predict_length')
parser.add_argument('--dropout', type=int, default=0.2, help='history_length')
parser.add_argument('--gatOut', type=int, default=20, help='history_length')

parser.add_argument('--alpha', type=int, default=0.2, help='history_length')
parser.add_argument('--hidden_size', type=int, default=64, help='history_length')
parser.add_argument('--num_layers', type=int, default=1, help='history_length')
parser.add_argument('--tcn_hidden', type=int, default=64, help='history_length')
parser.add_argument('--d', type=int, default=9, help='history_length')
parser.add_argument('--nhead', type=int, default=6, help='history_length')

parser.add_argument('--kernel_size', type=int, default=2, help='history_length')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 配置GPU,因为可能有多个GPU，这里用了第0号GPU

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 定义设备

if __name__ == "__main__":

    train_data = LoadData(data_path="datasets/PEMS04/PEMS04.npz",
                          Split_rate=[0.6, 0.2],
                          history_length=args.history_length, predict_length=args.predict_length,
                          train_mode="train")
    print(len(train_data))
    nodes_num = train_data.flow_data.shape[0]
    print("nodes_num:", nodes_num)
    # TODO 比较A方式和B方式

    graph = preprocess_grap_B(LoadData.to_tensor(
        get_adjacent_matrix(distance_file="datasets/PEMS04/PEMS04.csv", num_nodes=nodes_num))).to(device)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=2)

    val_data = LoadData(data_path="datasets/PEMS04/PEMS04.npz",
                        Split_rate=[0.6, 0.2],
                        history_length=args.history_length, predict_length=args.predict_length,
                        train_mode="val")
    print(len(val_data))
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=2)

    test_data = LoadData(data_path="datasets/PEMS04/PEMS04.npz",
                         Split_rate=[0.6, 0.2],
                         history_length=args.history_length, predict_length=args.predict_length,
                         train_mode="test")
    print(len(test_data))
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=2)

    model = STCGAT(device=device, nfeat=args.history_length, nhid=args.hidden, gatOut=args.gatOut, nheads=args.nb_heads,
                   dropout=args.dropout, predict_length=args.predict_length, hidden_size=args.hidden_size,
                   num_layers=args.num_layers, alpha=args.alpha, tcn_hidden=args.tcn_hidden, d=args.d,
                   kernel_size=args.kernel_size).to(device)

    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr)
    criterion = nn.MSELoss()  # 均方损失函数

    criterion_SoftDTW = SoftDTW(use_cuda=False, gamma=0.1)


    def res(model, test_loader):
        model.eval()  # 评估模式, 这会关闭dropout

        pred = []
        label = []

        with torch.no_grad():
            pbar = tqdm(test_loader)

            for data in pbar:
                flow = data["flow_x"].to(device)
                B = flow.shape[0]
                T = flow.shape[1]
                predict_value = model(flow, graph).to(
                    torch.device("cpu")).view(B, T, -1)

                prediction = LoadData.recover_data(test_loader.dataset.flow_norm[0], test_loader.dataset.flow_norm[1],
                                                   predict_value.transpose(0, 1).numpy())
                target = LoadData.recover_data(test_loader.dataset.flow_norm[0], test_loader.dataset.flow_norm[1],
                                               data["flow_y"].view(B, T, -1).transpose(0, 1).numpy())

                # (307, 260, 12)

                pbar.set_postfix({'loss': '{:02.4f}'.format(1)})  # 输入一个字典，显示实验指标
                pbar.set_description("Test")

                p = np.swapaxes(prediction, 0, 1)
                q = np.swapaxes(target, 0, 1)
                # p: (260, 307, 12)
                pred.append(p)
                label.append(q)

        pred = np.concatenate(pred, axis=0)
        label = np.concatenate(label, axis=0)

        T = pred.shape[2]
        maes = []
        rmses = []
        mapes = []
        wapes = []

        for i in range(T):
            mae, rmse, mape, r2_new = Evaluation.total_3(np.round(label[:, :, i]), np.round(pred[:, :, i]))
            maes.append(mae)
            rmses.append(rmse)
            mapes.append(mape)
            wapes.append(r2_new)
            print('step %d, mae: %.4f, rmse: %.4f, mape: %.4f, R2: %.4f' % (i + 1, mae, rmse, mape, r2_new))
        return np.mean(maes), np.mean(rmses), np.mean(mapes), np.mean(wapes)


    def test(model, test_loader):
        model.eval()  # 评估模式, 这会关闭dropout

        pred = []
        label = []

        with torch.no_grad():
            pbar = tqdm(test_loader)

            for data in pbar:
                flow = data["flow_x"].to(device)
                B = flow.shape[0]
                T = flow.shape[1]
                predict_value = model(flow, graph).to(
                    torch.device("cpu")).view(B, T, -1)

                prediction = LoadData.recover_data(test_loader.dataset.flow_norm[0], test_loader.dataset.flow_norm[1],
                                                   predict_value.transpose(0, 1).numpy())
                target = LoadData.recover_data(test_loader.dataset.flow_norm[0], test_loader.dataset.flow_norm[1],
                                               data["flow_y"].view(B, T, -1).transpose(0, 1).numpy())

                pbar.set_postfix({'loss': '{:02.4f}'.format(1)})  # 输入一个字典，显示实验指标
                pbar.set_description("Test")

                p = np.swapaxes(prediction, 0, 1)
                q = np.swapaxes(target, 0, 1)

                pred.append(p)
                label.append(q)

        pred = np.concatenate(pred, axis=0)
        label = np.concatenate(label, axis=0)

        T = pred.shape[2]
        maes = []
        rmses = []
        mapes = []
        wapes = []

        result_file = "result/pems04/data_result/result.h5"
        file_obj = h5py.File(result_file, "w")
        file_obj["predict"] = pred.reshape(nodes_num, -1)[:, :, np.newaxis]  # [N, T, D]
        file_obj["target"] = label.reshape(nodes_num, -1)[:, :, np.newaxis]  # [N, T, D]

        file_obj.close()

        for i in range(T):
            label_1 = label[:, :, i].reshape(nodes_num, -1)
            pred_1 = pred[:, :, i].reshape(nodes_num, -1)

            mae, rmse, mape, r2_new = Evaluation.total_3(label_1, pred_1)
            maes.append(mae)
            rmses.append(rmse)
            mapes.append(mape)
            wapes.append(r2_new)
            print('step %d, mae: %.4f, rmse: %.4f, mape: %.4f, R2: %.4f' % (i + 1, mae, rmse, mape, r2_new))
        return np.mean(maes), np.mean(rmses), np.mean(mapes), np.mean(wapes)


    RESUME = False
    start_epoch = -1

    if RESUME:
        path_checkpoint = "saveModels/pems04/checkpoint/ckpt_best_459.pth"  # 断点路径
        checkpoint = torch.load(path_checkpoint)  # 加载断点

        model.load_state_dict(checkpoint['net'])  # 加载模型可学习参数

        optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
        start_epoch = checkpoint['epoch']  # 设置开始的epoch

    # Train model
    t_total = time.time()
    loss_values = []
    bad_counter = 0
    best = args.epochs + 1
    best_epoch = 0
    train_loss_all = []
    for epoch in range(start_epoch + 1, args.epochs):
        model.train()
        total_train_loss = 0.
        start_time = time.time()
        pbar = tqdm(train_loader)
        num = 0
        for data in pbar:
            optimizer.zero_grad()
            flow = data["flow_x"].to(device)
            labels_value = data["flow_y"]

            predict_value = model(flow, graph).to(
                torch.device("cpu"))

            # loss_train = criterion(predict_value, labels_value)
            # loss_train.backward()
            # optimizer.step()

            B = predict_value.shape[0]
            T = predict_value.shape[2]
            _y = predict_value.transpose(1, 2).view(B, T, -1)
            y = data["flow_y"].transpose(1, 2).view(B, T, -1)
            loss_SoftDTW = criterion_SoftDTW(y, _y)
            loss_SoftDTW.mean().backward()
            optimizer.step()

            loss_train = criterion(predict_value, labels_value)

            pbar.set_postfix({'loss': '{:02.4f}'.format(loss_train.item())})  # 输入一个字典，显示实验指标
            pbar.set_description("Trainer")

            total_train_loss += loss_train.item()
            num += 1
        train_loss_all.append(total_train_loss / num)

        end_time = time.time()
        checkpoint = {
            "net": model.state_dict(),
            'optimizer': optimizer.state_dict(),
            "epoch": epoch
        }
        torch.save(checkpoint, 'saveModels/pems04/checkpoint/ckpt_best_%s.pth' % (str(epoch)))

        print("Epoch: {:04d}, loss_train:{:.4f}, Time: {:02.2f} mins".format(epoch + 1,
                                                                             1000 * total_train_loss / len(train_data),
                                                                             (end_time - start_time) / 60))
        if epoch % 10 == 0 and epoch != 0:
            mae, rmse, mape, r2 = res(model, val_loader)
            print('average, mae: %.4f, rmse: %.4f, mape: %.4f, R2: %.4f' % (mae, rmse, mape, r2))

        loss_values.append(total_train_loss)

        torch.save(model.state_dict(), '{}.pkl'.format(epoch))

        if loss_values[-1] < best:
            best = loss_values[-1]
            best_epoch = epoch
            bad_counter = 0
        else:
            bad_counter += 1

        files = glob.glob('*.pkl')
        for file in files:
            epoch_nb = int(file.split('.')[0])
            if epoch_nb < best_epoch:
                os.remove(file)

    files = glob.glob('*.pkl')
    for file in files:
        epoch_nb = int(file.split('.')[0])
        if epoch_nb > best_epoch:
            os.remove(file)

    with open("train_loss.txt", "w") as f:
        f.write(str(train_loss_all))

    # Restore best model
    print('Loading {}th epoch'.format(best_epoch))
    model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))

    # Testing
    mae, rmse, mape, r2 = test(model, test_loader)
    print('average, mae: %.4f, rmse: %.4f, mape: %.4f, R2: %.4f' % (mae, rmse, mape, r2))

    visualize_result(h5_file="result/pems04/data_result/result.h5",
                     nodes_id=1, time_se=[0, 12 * 24],  # 是节点的时间范围
                     visualize_file="result/pems04/picture_result/{}_node".format(150))
