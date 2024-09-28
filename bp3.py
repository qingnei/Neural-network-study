import numpy as np


def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))

def sigmoid_derivative(s):
    return s * (1 - s)

if __name__ == '__main__':
    # 第一层：隐层具有3个神经元，输入是2个，所以该层的权重系数v形状为:[2,3]
    v = np.asarray([
        [0.1, 0.2, 0.3],
        [0.15, 0.25, 0.35]
    ])
    b1 = np.asarray([0.35])
    # 第二层：输出层具有2个神经元，输入是3个，所以该层的权重系数w的形状为:[3,2]
    w = np.asarray([
        [0.4, 0.45],
        [0.5, 0.55],
        [0.6, 0.65]
    ])
    b2 = np.asarray([0.65])

    # 当前输入1个样本，每个样本2个特征属性，就相当于输入层的神经元是2个
    x = np.asarray([
        [5.0, 10.0],
        [5.0, 10.0],
        [5.0, 10.0],
        [5.0, 10.0],
        [5.0, 10.0]
    ])
    # 实际值
    d = np.asarray([
        [0.01, 0.99],  # 人为任意给定的
        [0.01, 0.99],  # 人为任意给定的
        [0.01, 0.99],  # 人为任意给定的
        [0.01, 0.99],  # 人为任意给定的
        [0.01, 0.99]  # 人为任意给定的
    ])

    # 第一个隐藏的操作输出
    net_h = np.dot(x, v) + b1  # [N,3] N表示样本数量，3表示每个样本有3个特征
    out_h = sigmoid(net_h)
    # 输出层的操作输出
    net_o = np.dot(out_h, w) + b2  # [N,2] N表示样本数目，2表示每个样本有2个特征/2个输出
    out_o = sigmoid(net_o)
    loss = 0.5 * np.sum(np.power((out_o - d), 2))
    # print(loss)
    # print(net_h)
    # print(out_h)
    # print(net_o)
    # print(out_o)
    # print(x)
    # print("=" * 50)

    # TODO: 基于矩阵的反向传播 --> 基于Numpy实现全连接神经网络
    
    
lr = 0.5

for epoch in range(9999):  # 迭代次数
        # 前向传播
    net_h = np.dot(x, v) + b1
    out_h = sigmoid(net_h)
    net_o = np.dot(out_h, w) + b2
    out_o = sigmoid(net_o)

        # 计算损失
    loss = 0.5 * np.sum(np.power((out_o - d), 2))

        # 反向传播
    t1 = (out_o - d) * sigmoid_derivative(out_o)  # 输出层的梯度
    gd_w = np.dot(out_h.T, t1)
    t2 = np.dot(t1, w.T) * sigmoid_derivative(out_h)  # 隐藏层的梯度
    gd_v = np.dot(x.T, t2)

        # 更新权重和偏置
    w -= lr * gd_w
    v -= lr * gd_v
    

        # 每999次迭代打印一次损失
    if epoch % 999 == 0:
        print(f'Epoch {epoch}, Loss: {loss}')

    # 打印最终的权重和偏置
print("Updated weights:")
print("Weights v:", v)
print("Weights w:", w)