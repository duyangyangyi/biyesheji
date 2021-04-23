import torch
import numpy as np
import random


def default_criterion(cnn, param):
    return True


def op_swap(parameter, old_set, SE, criterion=default_criterion, cnn=None):
    """
        交换算子
    :param parameter: 参数矩阵，对每一行进行状态转移变换，对于一维参数向量也必须传入二维array数据
    :param old_set: 参数范围，一个有两个元素的向量，如[min, max], 如果min=max,该层参数不参与优化
    :param SE: 搜索强度产生候选解的个数
    :param criterion: 对于产生的候选解的限制条件
    :param cnn:对于CNN优化特别设置的参数，非CNN优化可不填
    :return: 返回一个三维数组， 即SE个二维参数矩阵
    """
    shape = np.insert(parameter.shape, 0, SE)
    param_array = np.empty(shape)  # 创建一个三维的空数组，用来容纳产生的候选解
    num = 0  # 候选解队列索引
    i = 0
    while num != SE:  # 直到产生足够满足条件的候选解退出循环
        i += 1
        if i > 2000:
            print('swap产生候选解超时')
            return False
        temp_param = parameter.copy()  # .cpoy()只赋值，不赋地址
        for i in range(len(temp_param)):  # 分别对参数矩阵每一行参数进行变换
            o_set = np.arange(old_set[i][0], old_set[i][1]+1)  # 根据设置的范围产生一个整数列
            m = len(o_set)-1
            if m < 1:  # 如果old_set中min=max,o_set长度为1，不参与优化，直接跳过该层参数
                continue
            r = torch.randperm(len(temp_param[i]))
            t = r[:2]  # 随机产生两个不相等的整数
            temp1 = temp_param[i][t[0]]
            temp_param[i][t[0]] = temp_param[i][t[1]]
            temp_param[i][t[1]] = temp1  # 交换两个值

        if criterion(cnn, temp_param) and (temp_param != parameter).any():  # 如果满足条件，加入候选解序列，不满足跳过，继续产生
            param_array[num] = temp_param.copy()
            num = num + 1
    return param_array.astype(np.int)


def op_substitute(parameter, old_set, SE, criterion=default_criterion, cnn=None):
    """
        替换算子
    :param parameter: 参数矩阵，对每一行进行状态转移变换，对于一维参数向量也必须传入二维array数据
    :param old_set: 参数范围，一个有两个元素的向量，如[min, max], 如果min=max,该层参数不参与优化
    :param SE: 搜索强度产生候选解的个数
    :param criterion: 对于产生的候选解的限制条件
    :param cnn:对于CNN优化特别设置的参数，非CNN优化可不填
    :return: 返回一个三维数组， 即SE个二维参数矩阵
    """
    shape = np.insert(parameter.shape, 0, SE)
    param_array = np.empty(shape)  # 创建一个三维的空数组，用来容纳产生的候选解
    num = 0  # 候选解队列索引
    i = 0
    while num != SE:  # 直到产生足够满足条件的候选解退出循环
        i += 1
        if i > 2000:
            print('substitute产生候选解超时')
            return False
        temp_param = parameter.copy()  # .cpoy()只赋值，不赋地址
        for i in range(len(temp_param)):  # 分别对参数矩阵每一行参数进行变换
            o_set = np.arange(old_set[i][0], old_set[i][1]+1)  # 根据设置的范围产生一个整数列
            m = len(o_set)-1
            if m < 1:  # 如果old_set中min=max,o_set长度为1，不参与优化，直接跳过该层参数
                continue
            n = len(temp_param[i])-1
            r1, r2 = random.randint(0, n), random.randint(0, m-1)  # 随机产生一个参数索引和整数列索引
            index = np.where(o_set == temp_param[i][r1])
            new_set = np.delete(o_set, index)  # 删除整数列中与随机指定位置的参数相等的数
            temp_param[i][r1] = new_set[r2]  # 重新赋值
        if criterion(cnn, temp_param) and (temp_param != parameter).any():  # 如果满足条件，加入候选解序列，不满足跳过，继续产生
            param_array[num] = temp_param.copy()
            num = num + 1
    return param_array.astype(np.int)


def op_shift(parameter, old_set, SE, criterion=default_criterion, cnn=None):
    """
        转换算子
    :param parameter: 参数矩阵，对每一行进行状态转移变换，对于一维参数向量也必须传入二维array数据
    :param old_set: 参数范围，一个有两个元素的向量，如[min, max], 如果min=max,该层参数不参与优化
    :param SE: 搜索强度产生候选解的个数
    :param criterion: 对于产生的候选解的限制条件
    :param cnn:对于CNN优化特别设置的参数，非CNN优化可不填
    :return: 返回一个三维数组， 即SE个二维参数矩阵
    """
    shape = np.insert(parameter.shape, 0, SE)
    param_array = np.empty(shape)  # 创建一个三维的空数组，用来容纳产生的候选解
    num = 0  # 候选解队列索引
    i = 0
    while num != SE:  # 直到产生足够满足条件的候选解退出循环
        i += 1
        if i > 2000:
            print('shift产生候选解超时')
            return False
        temp_param = parameter.copy()
        for i in range(len(temp_param)):  # 分别对参数矩阵每一行参数进行变换
            o_set = np.arange(old_set[i][0], old_set[i][1]+1)  # 根据设置的范围产生一个整数列
            m = len(o_set)-1
            if m < 1:  # 如果old_set中min=max,o_set长度为1，不参与优化，直接跳过该层参数
                continue
            r = torch.randperm(len(temp_param[i])).numpy()
            t = r[:2]  # 随机产生两个不相等的整数
            if t[0] < t[1]:
                temp = np.insert(temp_param[i], t[1]+1, temp_param[i][t[0]])  # 将t[0]数据插入t[1]数据的后面
                temp = np.delete(temp, t[0])  # 删除原t[0]数据
                temp_param[i] = temp
            else:  # 同上
                temp = np.insert(temp_param[i], t[0]+1, temp_param[i][t[1]])
                temp = np.delete(temp, t[1])
                temp_param[i] = temp
        if criterion(cnn, temp_param) and (temp_param != parameter).any():  # 如果满足条件，加入候选解序列，不满足跳过，继续产生
            param_array[num] = temp_param.copy()
            num = num + 1
    return param_array.astype(np.int)


def op_symmetry(parameter, old_set, SE, criterion=default_criterion, cnn=None):
    """
            对称算子
    :param parameter: 参数矩阵，对每一行进行状态转移变换，对于一维参数向量也必须传入二维array数据
    :param old_set: 参数范围，一个有两个元素的向量，如[min, max], 如果min=max,该层参数不参与优化
    :param SE: 搜索强度产生候选解的个数
    :param cnn:对于CNN优化特别设置的参数，非CNN优化可不填
    :param criterion: 对于产生的候选解的限制条件
    :return: 返回一个三维数组， 即SE个二维参数矩阵
    :return:
    """
    shape = np.insert(parameter.shape, 0, SE)
    param_array = np.empty(shape)  # 创建一个三维的空数组，用来容纳产生的候选解
    num = 0  # 候选解队列索引
    i = 0  # 避免陷入死循环
    while num != SE:  # 直到产生足够满足条件的候选解退出循环
        i += 1
        if i > 2000:
            print('symmetry产生候选解超时')
            return False
        temp_param = parameter.copy()
        for i in range(len(temp_param)):  # 分别对参数矩阵每一行参数进行变换
            o_set = np.arange(old_set[i][0], old_set[i][1]+1)  # 根据设置的范围产生一个整数列
            m = len(o_set)-1
            if m < 1:  # 如果old_set中min=max,o_set长度为1，不参与优化，直接跳过该层参数
                continue
            r = torch.randperm(len(temp_param[i])).numpy()
            t = r[:2]  # 随机产生两个不相等的整数
            if t[0] < t[1]:
                temp = np.flipud(temp_param[i][t[0]:t[1]+1])  # 将切片部分进行翻转
                temp_param[i][t[0]:t[1]+1] = temp  # 将翻转的切片再赋值回原向量
            else:  # 同上
                temp = np.flipud(temp_param[i][t[1]:t[0]+1])
                temp_param[i][t[1]:t[0]+1] = temp
        if criterion(cnn, temp_param) and (temp_param != parameter).any():  # 如果满足条件，加入候选解序列，不满足跳过，继续产生
            param_array[num] = temp_param.copy()
            num = num + 1
    return param_array.astype(np.int)


def op_direction(old_param, direction, old_set, criterion=default_criterion, cnn=None):
    new_param = old_param + direction
    for i in range(len(new_param)):
        if old_set[i][0] == old_set[i][1]:
            continue
        new_param[i] = np.clip(new_param[i], old_set[i][0], old_set[i][1])
    if criterion(cnn, new_param):
        return new_param.astype(np.int)
    else:
        return np.zeros_like(old_param)


def operation_memo(old_param, old_solution, fitness, memorize, old_set, criterion=default_criterion, cnn=None):
    new_param = op_direction(old_param, memorize, old_set, criterion, cnn)
    if (new_param == np.zeros_like(old_param)).all():
        return old_param, old_solution
    else:
        new_solution = fitness(new_param, old_solution)
        if new_solution > old_solution:
            return new_param, new_solution
        else:
            return old_param, old_solution


def operation(operator, old_param, old_solution, fitness, old_set, SE, criterion=default_criterion, cnn=None):
    param_array = None
    if operator == "shift":
        param_array = op_shift(old_param, old_set, SE, criterion=criterion, cnn=cnn)
    elif operator == "swap":
        param_array = op_swap(old_param, old_set, SE, criterion=criterion, cnn=cnn)
    elif operator == "substitute":
        param_array = op_substitute(old_param, old_set, SE, criterion=criterion, cnn=cnn)
    elif operator == "symmetry":
        param_array = op_symmetry(old_param, old_set, SE, criterion=criterion, cnn=cnn)

    new_solution = np.empty(param_array.shape[0])
    for i in range(param_array.shape[0]):
        print('当前评价 ' + operator + ' 算子第 {}个候选解'.format(i+1))
        new_solution[i] = fitness(param_array[i], old_solution)
    index = np.argmax(new_solution)

    if new_solution[index] > old_solution:  # 如果产生一个较好的解
        direction = shift_dir(param_array[index]-old_param)
        new_param = op_direction(param_array[index], direction, old_set, criterion=criterion, cnn=cnn)  # 使用方向算子
        if (new_param == np.zeros_like(old_param)).all():  # 判断方向算子产生的解是否合理
            return param_array[index], new_solution[index], direction
        else:  # 如果方向算子产生的解合理
            print('在 ' + operator + ' 算子中进行方向算子计算')
            solution_dir = fitness(new_param, old_solution)
            if solution_dir > new_solution[index]:
                return new_param, solution_dir, direction
            else:  # 如果方向算子的解不好
                return param_array[index], new_solution[index], direction
    else:
        if np.random.rand(1) < 0.15:
            return param_array[index], new_solution[index], np.zeros_like(old_param)
        return old_param, old_solution, np.zeros_like(old_param)


def shift_dir(direction):
    for i in range(len(direction)):
        for j in range(len(direction[i])):
            if direction[i][j] > 0:
                direction[i][j] = 1
            if direction[i][j] < 0:
                direction[i][j] = -1
    return direction



# random.rand()：根据给定维度生成[0,1)之间的数据
# random.randn()：根据给定维度（不给维度时为单个数），产生符合标准正态分布的随机数
# random.normal()：产生可定义均值和标准差的正态分布随机数
# random.randint()：返回给定维度的随机整数
# random.random()\random.sample：返回给定维度的[0,1)之间的随机数
# random.choice()：从给定的一维数组中生成随机数
# random.seed()：当设置相同的seed,每次生成的随机数相同，不设置seed,则每次会生成不同的随机数，数字一般可随意设置
