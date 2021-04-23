import numpy as np
import torch.optim as opti
import torch.nn as nn
import time
from util.Models import NewNet
from util.functions import time_record, cnn_criterion, param_cal, mls_sigmod, cur2, normal_distribution
import torch
from util import my_data
from DSTA.D_STA import Dsta
train_loader, test_loader = my_data.cifar(train_size=128, test_size=1024, num_workers=8)

print('数据读取完毕')
in_size = [32, 32, 3]
out_dim = 100
epoch = 10
diverse = 0
iteration = 120
interval = 78

feature_num = [64, 128, 256, 256, 128]
conv_ks = [3, 5, 3, 5, 3]
conv_str = [1, 2, 1, 2, 1]
pool_ks = [3, 0, 0, 0, 3]
pool_str = [2, 0, 0, 0, 2]
padding = [1, 2, 1, 2, 1]
param = np.array([feature_num, conv_ks, conv_str, pool_ks, pool_str, padding])
param_range = [[64, 256], [2, 7], [1, 2], [1, 1], [1, 1], [0, 3]]

criterion = nn.NLLLoss()


def test_net(net):
    test_correct = 0
    net.eval()
    with torch.no_grad():
        for data, target in test_loader:
            (data, target) = data.cuda(diverse), target.cuda(diverse)
            output = net(data)
            predict = output.data.max(1, keepdim=True)[1]  # 计算准确数
            test_correct += predict.eq(target.data.view_as(predict)).sum().item()
    accuracy = 100. * test_correct / len(test_loader.dataset)
    return accuracy


num_epochs = []
learn_curve = []


def fitness(new_param, old_solution=0, train=True):
    net = NewNet(in_size=in_size, out_dim=out_dim, param=new_param).cuda(diverse)
    optimizer = opti.Adam(net.parameters())
    accuracy = []
    s = time.time()
    for i in range(epoch):
        for batch_idx, (data, target) in enumerate(train_loader):
            net.train()
            (data, target) = data.cuda(diverse), target.cuda(diverse)
            net.zero_grad()
            output = net(data)  # 网络计算
            loss = criterion(output, target)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

    #         if (batch_idx+1) % interval == 0 and train and i < epoch - 1:
    #             acc = test_net(net)
    #             accuracy.append(acc)
    #
    #     if 1 < i < epoch - 1 and train and (i % 2) == 0:
    #         x = np.arange(1, len(accuracy[7:]) + 1) + 7  # test2 第二轮数据开始
    #         a, b, sigma = mls_sigmod(x, accuracy[7:])
    #         ym = cur2(a, b, 50)
    #         prob = normal_distribution(old_solution, ym, sigma)
    #         print('据前{}个点预测值：{:.2f}, 小于最优值的概率：{:2f}%'.format((i+1)*5, ym, prob*100))
    #         if i > 0 and prob > 0.95:
    #             print('在第{}代提前终止训练，返回值{:.2f}'.format(i+1, ym))
    #             num_epochs.append(i+1)
    #             learn_curve.append(accuracy)
    #             time_record(s)
    #             return ym
    # accuracy.append(test_net(net))
    # learn_curve.append(accuracy)
    time_record(s)
    # num_epochs.append(10)
    accu = test_net(net)
    print('最终真值：{:.2f}'.format(accu))
    return accu


dsta = Dsta(param, fitness, param_range, 5)
print('优化器构建完毕')
solution0, param0, best_solution, best_param = [], [], [], []

param0.append(dsta.param)
best_param.append(dsta.best_param)
solution0.append(dsta.solution)
best_solution.append(dsta.best_solution)

s_time = time.time()
duration = 0
for i in range(iteration):
    print('\n第 {} 轮优化'.format(i+1))
    dsta.optimizer(fitness, criterion=cnn_criterion, cnn=in_size)
    print("当前解：{:.2f}%".format(dsta.solution))
    print("当前最优解：{:.2f}%".format(dsta.best_solution))
    print('已用时：')
    time_record(s_time)  # 已耗总时长
    param0.append(dsta.param)
    best_param.append(dsta.best_param)
    solution0.append(dsta.solution)
    best_solution.append(dsta.best_solution)
    if (i+1) % 20 == 0:
        np.save('./save/cifar-100/test5/param0', param0)
        np.save('./save/cifar-100/test5/solution0', solution0)
        np.save('./save/cifar-100/test5/best_param', best_param)
        np.save('./save/cifar-100/test5/best_solution', best_solution)
        np.save('./save/cifar-100/test5/num_epochs', num_epochs)
        np.save('./save/cifar-100/test5/learn_curve', learn_curve)
