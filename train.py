# -*- coding: utf-8 -*-
# @Project: STCGAT
# @Author  : shiqiZhang
import glob
import os
import time
import h5py
import torch
import random
import argparse
import configparser
import numpy as np
import torch.nn as nn
import torch.optim as optim
from utils.SoftDTW import SoftDTW
from models.STCGAT import STCGAT
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.metrics_1 import All_Metrics
from utils.prepareData import LoadData, preprocess_grap, get_adjacent_matrix
from utils.utils import visualize_result, create_dir_not_exist
from utils.load_data import get_dataloader, recover_data


Mode = 'Train'  # Train or Test
DATASET = 'PEMSD4'
MODEL = 'STCGAT'

#get configuration
config_file = './config/{}_{}.conf'.format(DATASET, MODEL)
config = configparser.ConfigParser()
config.read(config_file)

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=config['train']['seed'], help='Random seed.')
parser.add_argument('--epochs', type=int, default=config['train']['epochs'], help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=config['train']['batch_size'], help='Batch Size.')
parser.add_argument('--lr', type=float, default=config['train']['lr'], help='Initial learning rate.')
parser.add_argument('--gamma', type=float, default=config['train']['gamma'], help='soft-DTW gamma parameters.')
parser.add_argument('--alpha', type=float, default=config['train']['alpha'], help='alpha.')
parser.add_argument('--dropout', type=float, default=config['train']['dropout'], help='dropout.')

parser.add_argument('--num_nodes', type=int, default=config['data']['num_nodes'], help='Number of graph nodes.')
parser.add_argument('--history_length', type=int, default=config['data']['history_length'], help='Number of historical values.')
parser.add_argument('--predict_length', type=int, default=config['data']['predict_length'], help='Number of predicted values.')
parser.add_argument('--train_ratio', type=float, default=config['data']['train_ratio'], help='Training set partition rate.')
parser.add_argument('--val_ratio', type=float, default=config['data']['val_ratio'], help='Validation set partition rate.')
parser.add_argument('--test_ratio', type=float, default=config['data']['test_ratio'], help='Test set split ratio.')

parser.add_argument('--mae_thresh', default=config['test']['mae_thresh'], type=eval)
parser.add_argument('--mape_thresh', default=config['test']['mape_thresh'], type=float)

parser.add_argument('--input_dim', type=int, default=config['model']['input_dim'], help='Model input value dimension.')
parser.add_argument('--gat_units', type=int, default=config['model']['gat_units'], help='Number of hidden layers of GAT.')
parser.add_argument('--gat_heads', type=int, default=config['model']['gat_heads'], help='Number of long attention spans.')
parser.add_argument('--gatOut_dim', type=int, default=config['model']['gatOut_dim'], help='GAT Output Dimension.')
parser.add_argument('--lstm_units', type=int, default=config['model']['lstm_units'], help='Number of hidden layers of LSTM.')
parser.add_argument('--num_layers', type=int, default=config['model']['num_layers'], help='Number of LSTM layers.')
parser.add_argument('--tcn_units', type=int, default=config['model']['tcn_units'], help='Number of hidden layers of TCN.')
parser.add_argument('--d', type=int, default=config['model']['d'])
parser.add_argument('--kernel_size', type=int, default=config['model']['kernel_size'])

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

nodes_num = args.num_nodes

model = STCGAT(device=device, input_dim=args.input_dim, gat_units=args.gat_units, gatOut_dim=args.gatOut_dim, gat_heads=args.gat_heads,
               dropout=args.dropout, predict_length=args.predict_length, lstm_units=args.lstm_units,
               num_layers=args.num_layers, alpha=args.alpha, tcn_units=args.tcn_units, d=args.d,
               kernel_size=args.kernel_size).to(device)

optimizer = optim.Adam(model.parameters(),
                       lr=args.lr)
criterion = nn.MSELoss()

criterion_SoftDTW = SoftDTW(use_cuda=False, gamma=args.gamma)


def res(model, test_loader, graph, flow_norm):
    model.eval()

    pred = []
    label = []

    with torch.no_grad():
        pbar = tqdm(test_loader)

        for data in pbar:
            flow = data[0].to(device)
            B = flow.shape[0]
            T = flow.shape[1]
            predict_value = model(flow, graph).to(
                torch.device("cpu")).view(B, T, -1)

            prediction = recover_data(flow_norm[0], flow_norm[1],
                                      predict_value.transpose(0, 1).numpy())
            target = recover_data(flow_norm[0], flow_norm[1],
                                  data[1].view(B, T, -1).transpose(0, 1).numpy())

            pbar.set_description("Val")

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

    for i in range(T):
        mae, rmse, mape, _, _ = All_Metrics(label[:, :, i], pred[:, :, i],
                                            args.mae_thresh, args.mape_thresh)
        maes.append(mae)
        rmses.append(rmse)
        mapes.append(mape)
        print('Horizon {}, MAE:{:.2f}, RMSE: {:.2f}, MAPE: {:.2f}%'.format(i + 1, mae, rmse, mape * 100))

    return np.mean(maes), np.mean(rmses), np.mean(mapes) *100


def test(model, test_loader, graph, flow_norm):
    model.eval()

    pred = []
    label = []

    with torch.no_grad():
        pbar = tqdm(test_loader)

        for data in pbar:
            flow = data[0].to(device)
            B = flow.shape[0]
            T = flow.shape[1]
            predict_value = model(flow, graph).to(
                torch.device("cpu")).view(B, T, -1)

            prediction = recover_data(flow_norm[0], flow_norm[1],
                                      predict_value.transpose(0, 1).numpy())
            target = recover_data(flow_norm[0], flow_norm[1],
                                  data[1].view(B, T, -1).transpose(0, 1).numpy())

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

    result_dir_data = "result/{}/data_result".format(DATASET)
    result_dir_pic = "result/{}/picture_result".format(DATASET)
    create_dir_not_exist(result_dir_data)
    create_dir_not_exist(result_dir_pic)
    result_file = "{}/result.h5".format(result_dir_data)
    file_obj = h5py.File(result_file, "w")
    file_obj["predict"] = pred.reshape(nodes_num, -1)[:, :, np.newaxis]  # [N, T, D]
    file_obj["target"] = label.reshape(nodes_num, -1)[:, :, np.newaxis]  # [N, T, D]

    file_obj.close()


    for i in range(T):
        mae, rmse, mape, _, _ = All_Metrics(label[:, :, i], pred[:, :, i],
                                            args.mae_thresh, args.mape_thresh)
        # mae, rmse, mape, r2_new = Evaluation.total_3(np.round(label[:, :, i]), np.round(pred[:, :, i]))
        maes.append(mae)
        rmses.append(rmse)
        mapes.append(mape)

        print('Horizon {}, MAE:{:.2f}, RMSE: {:.2f}, MAPE: {:.2f}%'.format(i + 1, mae, rmse, mape*100))

    return np.mean(maes), np.mean(rmses), np.mean(mapes)*100


def train(model, train_loader, graph, flow_norm):

    RESUME = False  # Whether to continue breakpoint training
    start_epoch = -1
    checkpointDir = "saveModels/{}/checkpoint".format(DATASET)
    create_dir_not_exist(checkpointDir)

    if RESUME:
        path_checkpoint = "{}/ckpt_best_9.pth".format(checkpointDir)  # Mount Breakpoint
        checkpoint = torch.load(path_checkpoint)

        model.load_state_dict(checkpoint['net'])

        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']

    # Train model
    bad_counter = 0
    best = args.epochs + 1
    best_epoch = 0
    for epoch in range(start_epoch + 1, args.epochs):
        model.train()
        total_train_loss = 0.
        start_time = time.time()
        pbar = tqdm(train_loader)
        num = 0
        for data in pbar:
            optimizer.zero_grad()
            flow = data[0].to(device)
            labels_value = data[1]

            predict_value = model(flow, graph).to(
                torch.device("cpu"))

            B = predict_value.shape[0]
            T = predict_value.shape[2]
            _y = predict_value.transpose(1, 2).view(B, T, -1)
            y = data[1].transpose(1, 2).view(B, T, -1)
            loss_SoftDTW = criterion_SoftDTW(y, _y)
            loss_SoftDTW.mean().backward()
            optimizer.step()

            loss_train = criterion(predict_value, labels_value)

            pbar.set_postfix({'loss': '{:02.4f}'.format(loss_train.item())})  # 输入一个字典，显示实验指标
            pbar.set_description("Trainer")

            total_train_loss += loss_train.item()
            num += 1

        end_time = time.time()
        checkpoint = {
            "net": model.state_dict(),
            'optimizer': optimizer.state_dict(),
            "epoch": epoch
        }
        torch.save(checkpoint, '{}/ckpt_best_%s.pth'.format(checkpointDir) % (str(epoch)))

        print("Epoch: {:04d}, loss_train:{:.4f}, Time: {:02.2f} mins".format(epoch + 1,
                                                                             1000 * total_train_loss / len(train_data),
                                                                             (end_time - start_time) / 60))
        if epoch % 10 == 0 and epoch != 0:
            mae, rmse, mape = res(model, val_loader, graph, flow_norm)
            print('Average Horizon, MAE:{:.2f}, RMSE: {:.2f}, MAPE: {:.2f}%'.format(mae, rmse, mape))

        average_loss = total_train_loss/num

        torch.save(model.state_dict(), '{}.pkl'.format(epoch))

        if average_loss < best:
            best = average_loss
            best_epoch = epoch
            bad_counter = 0
        else:
            bad_counter += 1

        files = glob.glob('*.pkl')
        for file in files:
            if file.split('.')[0].isdigit():
                epoch_nb = int(file.split('.')[0])
                if epoch_nb < best_epoch:
                    os.remove(file)

    files = glob.glob('*.pkl')
    for file in files:
        if file.split('.')[0].isdigit():
            epoch_nb = int(file.split('.')[0])
            if epoch_nb > best_epoch:
                os.remove(file)

    # Restore best model
    os.rename('{}.pkl'.format(best_epoch), 'best_{}.pkl'.format(DATASET))
    print('Loading the best epoch.')
    model.load_state_dict(torch.load('best_{}.pkl'.format(DATASET)))

    # Testing
    mae, rmse, mape = test(model, test_loader, graph, flow_norm)
    print('Average Horizon, MAE:{:.2f}, RMSE: {:.2f}, MAPE: {:.2f}%'.format(mae, rmse, mape))

    visualize_result(h5_file="result/{}/data_result/result.h5".format(DATASET),
                     nodes_id=1, time_se=[0, 12 * 24],  # nodes_id：Number of visualization nodes， time_se：Visualization time range
                     visualize_file="result/{}/picture_result/node".format(DATASET))


if __name__ == "__main__":
    train_data, val_data, test_data, flow_norm = get_dataloader(dataset=DATASET,
                                                                split_ratio=[args.train_ratio, args.val_ratio, args.test_ratio], lag=args.history_length,
                                                                horizon=args.predict_length)
    graph = preprocess_grap(LoadData.to_tensor(
        get_adjacent_matrix(distance_file=DATASET, num_nodes=nodes_num))).to(device)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=8)

    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=8)

    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=8)
    if Mode == 'Train':
        train(model=model, train_loader=train_loader, graph=graph, flow_norm=flow_norm)
    elif Mode == 'Test':
        files = glob.glob('*.pkl')
        for file in files:
            if 'best_{}.pkl'.format(DATASET) == file:
                print('Loading the best epoch.')
                model.load_state_dict(torch.load('best_{}.pkl'.format(DATASET)))
                # Testing
                mae, rmse, mape = test(model, test_loader, graph, flow_norm)
                print('Average Horizon, MAE:{:.2f}, RMSE: {:.2f}, MAPE: {:.2f}%'.format(mae, rmse, mape))
                break
            else:
                print("Error:Model not yet trained！")
    else:
        print("Please set the mode to Train or Test！")

