import numpy as np
from util import my_data
from DSTA.D_STA import Dsta
from util.functions import function1, time_record, param_cal, cnn_criterion
from util.Models import NewNet
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as opti
import torch.nn as nn
import time
import torch
import torchvision

# #制造样本
# x = np.arange(0, 100, 0.2)
# xArr = []
# yArr = []
# for i in x:
#     lineX = [1]
#     lineX.append(i)
#     xArr.append(lineX)
#     yArr.append(0.5 * i + 3 + random.uniform(0, 1) * 4 * math.sin(i) )
#
# xMat = np.mat(xArr)
# yMat = np.mat(yArr).T
# xTx = xMat.T * xMat
# if np.linalg.det(xTx) == 0.0:
#     print("Can't inverse")
# ws = xTx.I * xMat.T * yMat
# print(ws)
#
# y = xMat * ws
# # 画图
# plt.title("linear regression")
# plt.xlabel("independent variable")
# plt.ylabel("dependent variable")
# plt.plot(x, yArr, 'go')
# plt.plot(x, y, 'r', linewidth=2)
# plt.show()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_loader, test_loader = my_data.cifar_mini(train_size=32, test_size=1024, num_workers=8)
print('数据读取完毕')
in_size = [32, 32, 3]
feature_num = [64, 128, 256, 256, 64]
conv_ks = [7, 5, 3, 3, 5]
conv_str = [1, 1, 1, 1, 1]
pool_ks = [3, 3, 0, 0, 3]
pool_str = [2, 2, 0, 0, 2]
padding = [3, 2, 1, 1, 2]

param = np.array([feature_num, conv_ks, conv_str, pool_ks, pool_str, padding])
criterion = nn.NLLLoss()
epoch = 80
interval = 20
c = []


def fitness(param):
    net = NewNet(in_size=in_size, out_dim=100, param=param).to(device)
    optimizer = opti.Adam(net.parameters())
    for i in range(epoch):
        net.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            (data, target) = data.to(device), target.to(device)
            net.zero_grad()
            output = net(data)  # 网络计算
            loss = criterion(output, target)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            if batch_idx % interval == 0:  # 监控进度
                print('Train Epoch: {} ({:.2f}%)\tLoss: {:.6f}'
                      .format(i + 1,
                              100. * (batch_idx + i * len(train_loader)) / (epoch * len(train_loader))
                              , loss.item()))
        net.eval()
        with torch.torch.no_grad():
            test_correct = 0
            for t_data, t_target in test_loader:
                (t_data, t_target) = t_data.to(device), t_target.to(device)
                output = net(t_data)
                predict = output.data.max(1, keepdim=True)[1]  # 计算准确数
                test_correct += predict.eq(t_target.data.view_as(predict)).sum()
            accuracy = 100. * test_correct / len(test_loader.dataset)
            c.append(accuracy)
            print(accuracy)


s = time.time()
fitness(param)
time_record(s)

np.save('./save/test1.npy', c)

