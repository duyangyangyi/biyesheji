import torch.nn as nn
import torch.nn.functional as F
import math
import time


class NewNet(nn.Module):
    def __init__(self, in_size, out_dim, param):
        super(NewNet, self).__init__()
        self.line_num = 0
        self.in_size = in_size
        self.out_dim = out_dim
        self.param = param
        self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, = None, None, None, None, None
        self.dense = None
        self.init_conv()
        self.init_dense()

    def init_conv(self):
        conv = []
        for i in range(self.param.shape[1]):  # 遍历每一列
            if i == 0:  # 是否为第一层卷积
                input_mun = self.in_size[2]
            else:
                input_mun = self.param[0][i-1]
            if self.param[3][i] != 0:  # 判断是否有池化层
                conv_demo = nn.Sequential(
                    nn.Conv2d(input_mun, self.param[0][i], kernel_size=self.param[1][i],
                              stride=self.param[2][i], padding=self.param[5][i]),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=self.param[3][i], stride=self.param[4][i]),
                    )
            else:
                conv_demo = nn.Sequential(
                    nn.Conv2d(input_mun, self.param[0][i], kernel_size=self.param[1][i], stride=self.param[2][i],
                              padding=self.param[5][i]),
                    nn.ReLU(),
                    )
            conv.append(conv_demo)
        self.conv1, self.conv2, self.conv3, self.conv4, self.conv5 = conv[0], conv[1], conv[2], conv[3], conv[4]

    def init_dense(self):
        length, height = self.in_size[0], self.in_size[1]
        for conv in self.param.T:
            length = math.floor((length - conv[1] + 2*conv[5]) / conv[2] + 1)
            height = math.floor((height - conv[1] + 2*conv[5]) / conv[2] + 1)
            if conv[3] != 0:
                length = math.floor((length - conv[3]) / conv[4] + 1)
                height = math.floor((height - conv[3]) / conv[4] + 1)
        self.line_num = length * height * self.param[0][-1]
        dense = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.line_num, math.floor(self.line_num / 2)),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(math.floor(self.line_num / 2), math.floor(self.line_num / 2)),
            nn.ReLU(),
            nn.Linear(math.floor(self.line_num / 2), self.out_dim, bias=False),
        )
        self.dense = dense

    def update_net(self, param):
        self.param = param
        self.init_conv()
        # self.init_dense()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(-1, self.line_num)
        x = self.dense(x)
        return F.log_softmax(x, dim=1)  # 逻辑似然代价函数


