import numpy as np
import time
import math


def function1(x):
    x_ = x.T
    Q = [[4, -2, -3, 0, 1, 4, 5, -2],
         [-2, -4, 0, 0, 2, 2, 0, 0],
         [-3, 0, 8, -2, 0, 3, 4, 0],
         [0, 0, -2, -4, 4, 4, 0, 1],
         [1, 2, 0, 4, 100, 2, 0, -2],
         [4, 2, 3, 4, 2, 100, 1, 0],
         [5, 0, 4, 0, 0, 1, 200, 4],
         [-3, 0, 0, 1, -2, 0, 4, 10]]
    c = [-4, 1, -8, 3, -100, -10, -20, 0]
    x_ = x.T
    y = 0.5 * np.matmul(np.matmul(x, Q), x_) + np.matmul(c, x_)
    return -y


def restrain_param(param, scope):
    for i in range(len(param)):
        index = np.where(param[i] > scope[i][1])
        param[i][index] = scope[i][1]
        index = np.where(param[i] < scope[i][0])
        param[i][index] = scope[i][0]
    return param


def time_record(start):
    end = time.time()
    duration = end - start
    hour = duration // 3600
    minute = (duration - hour * 3600) // 60
    second = duration - hour * 3600 - minute * 60
    print('hour: %d, minute: %d, second: %d' % (hour, minute, second))
    return duration


def cnn_criterion(in_size, param):
    length, height = in_size[0], in_size[1]
    param_sum = 0
    flops = 0
    for i in range(len(param.T)):
        if param.T[i][1] < param.T[i][2] or param.T[i][3] < param.T[i][4] or param.T[i][1] <= param.T[i][5]:
            return False
        length = math.floor((length - param.T[i][1] + 2*param.T[i][5]) / param.T[i][2] + 1)
        height = math.floor((height - param.T[i][1] + 2*param.T[i][5]) / param.T[i][2] + 1)
        if param.T[i][3] != 0 and param.T[i][4] != 0:
            length = math.floor((length - param.T[i][3]) / param.T[i][4] + 1)
            height = math.floor((height - param.T[i][3]) / param.T[i][4] + 1)
        if length < 1 or height < 1:  # 计算出特征子图大小小于1
            return False
        if i == 0:
            conv_param = param.T[i][0] * in_size[2] * param.T[i][1] * param.T[i][1] + param.T[i][0]
            conv_flops = (in_size[2] * param.T[i][1] * param.T[i][1] * param.T[i][0] * length * height) / 10000000
        else:
            conv_param = param.T[i][0] * param.T[i-1][0] * param.T[i][1] * param.T[i][1] + param.T[i][0]
            conv_flops = (param.T[i-1][0] * param.T[i][1] * param.T[i][1] * param.T[i][0] * length * height) / 10000000
        flops += conv_flops
        param_sum += conv_param
    liner_num = param[0][4] * length * height
    param_sum += liner_num * math.floor(liner_num / 2) + math.floor(liner_num / 2)
    param_sum += math.floor(liner_num / 2) * math.floor(liner_num / 2) + math.floor(liner_num / 2)
    param_sum += math.floor(liner_num / 2) * in_size[2]  # 全连接层的参数量
    flops += liner_num * math.floor(liner_num / 2) / 10000000
    flops += math.floor(liner_num / 2) * math.floor(liner_num / 2) / 10000000
    flops += math.floor(liner_num / 2) * in_size[2] / 10000000
    if param_sum > 2500000 or flops > 10:
        return False
    return True


def init_param(init_set, conv_layer):
    param_init = np.empty([init_set.shape[0], conv_layer])
    for i in range(len(param_init)):
        param_init[i] = np.random.randint(init_set[i][0], init_set[i][1]+1, conv_layer)
    return param_init.astype(np.int)


def param_cal(in_size, param):
    length, height = in_size[0], in_size[1]
    param_sum = 0
    flops = 0
    for i in range(len(param.T)):
        length = math.floor((length - param.T[i][1] + 2*param.T[i][5]) / param.T[i][2] + 1)
        height = math.floor((height - param.T[i][1] + 2*param.T[i][5]) / param.T[i][2] + 1)
        if param.T[i][3] != 0 and param.T[i][4] != 0:
            length = math.floor((length - param.T[i][3]) / param.T[i][4] + 1)
            height = math.floor((height - param.T[i][3]) / param.T[i][4] + 1)
        if i == 0:
            conv_param = param.T[i][0] * in_size[2] * param.T[i][1] * param.T[i][1] + param.T[i][0]
            conv_flops = (in_size[2] * param.T[i][1] * param.T[i][1] * param.T[i][0] * length * height) / 10000000
        else:
            conv_param = param.T[i][0] * param.T[i-1][0] * param.T[i][1] * param.T[i][1] + param.T[i][0]
            conv_flops = (param.T[i-1][0] * param.T[i][1] * param.T[i][1] * param.T[i][0] * length * height) / 10000000
        flops += conv_flops
        param_sum += conv_param
    liner_num = param[0][4] * length * height
    param_sum += liner_num * math.floor(liner_num / 2) + math.floor(liner_num / 2)
    param_sum += math.floor(liner_num / 2) * math.floor(liner_num / 2) + math.floor(liner_num / 2)
    param_sum += math.floor(liner_num / 2) * in_size[2]  # 全连接层的参数量
    flops += liner_num * math.floor(liner_num / 2) / 10000000
    flops += math.floor(liner_num / 2) * math.floor(liner_num / 2) / 10000000
    flops += math.floor(liner_num / 2) * in_size[2] / 10000000
    return param_sum, flops


def mls_sigmod(x, y):
    x, y = np.array(x), np.array(y)
    n = len(x)
    v, u = np.log(x), y
    v_, u_ = np.average(v), np.average(u)
    Lvv, Lvu, Luu = 0, 0, 0
    for i in range(n):
        Lvv += np.power((v[i] - v_), 2)
        Lvu += (v[i] - v_) * (u[i] - u_)
        Luu += np.power((u[i] - u_), 2)
    b = Lvu / Lvv
    a = u_ - b * v_
    sigma = (Luu - b * Lvu) / (n - 2)
    return a, b, math.sqrt(sigma)


def sigmod(a, b, x):
    return 1 / (a + b * np.exp(-x))


def cur1(a, b, x):
    return x / (a * x + b)


def cur2(a, b, x):
    return a + b * np.log(x)


def fx_st_normal_distribution(x):
    return math.exp((-x**2)/2)/(math.sqrt(2*math.pi))


def st_normal_distribution(x):
    # 处理x<0(目标点在分布中心左侧)的情况
    if x < 0:
        return 1-st_normal_distribution(-x)
    if x == 0:
        return 0.5
    # 求标准正态分布的概率密度的积分
    s = 1/10000
    xk = []
    for i in range(1, int(10000 * x)):
        xk.append(i*s)
    integral = (fx_st_normal_distribution(0)+fx_st_normal_distribution(x))/2  # f(0)和f(x)各算一半
    for each in xk:
        integral += fx_st_normal_distribution(each)
    return 0.5+integral*s


def normal_distribution(x, u, s):
    if s <= 0:
        return 0
    if u <= 0:
        return 1
    else:
        z = (x-u)/s
        return st_normal_distribution(z)


