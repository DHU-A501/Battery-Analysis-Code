# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 10:57:28 2017

增加数据量
去除偏离数据并重新训练

@author: WangDian
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from matplotlib import pyplot as plt
import my_Net

# =============================================================================
# 是否使用GPU
# =============================================================================
GPU = 0
# =============================================================================
# 写入Exel文件中
# =============================================================================
def write(filename):
    A_delete = A[index_delete]
    P1_con = np.concatenate((P1[1], P1[2], P1[3]), axis=1)
    P2_con = np.concatenate((P2[1], P2[2], P2[3]), axis=1)
    d1 = pd.DataFrame(A)
    d2 = pd.DataFrame(A2)
    d3 = pd.DataFrame(A_delete)
    d4 = pd.DataFrame(P1_con)
    d5 = pd.DataFrame(P2_con)
    d1.rename(columns={0:'U/mV',1:'Ri/mΩ', 2:'T/℃', 3:'Q/mAh', 4:'SOC/%', 5:'Cycles'}, inplace = True)  #更改列名
    d2.rename(columns={0:'U/mV',1:'Ri/mΩ', 2:'T/℃', 3:'Q/mAh', 4:'SOC/%', 5:'Cycles'}, inplace = True)  #更改列名
    d3.rename(columns={0:'U/mV',1:'Ri/mΩ', 2:'T/℃', 3:'Q/mAh', 4:'SOC/%', 5:'Cycles'}, inplace = True)  #更改列名
    d4.rename(columns={0:'y_true',1:'y_p', 2:'y_d'}, inplace = True)  #更改列名
    d5.rename(columns={0:'y_true',1:'y_p', 2:'y_d'}, inplace = True)  #更改列名
    if 1:
        with pd.ExcelWriter(Path_output + filename + '.xlsx') as writer:
            d1.to_excel(writer, sheet_name = '源数据')
            d2.to_excel(writer, sheet_name = '去偏后数据')
            d3.to_excel(writer, sheet_name = '偏离的数据')
            d4.to_excel(writer, sheet_name = '原误差')
            d5.to_excel(writer, sheet_name = '去偏后误差')

#==============================================================================
# 归一化函数 输入待归一化矩阵  输出归一化结果与复原函数
#==============================================================================
def Norm(X, Path_output=None, name=''):
    x_std = np.std(X,axis=0)
    x_mean = np.mean(X, axis=0)

    X_on = (X - x_mean)/x_std
    if Path_output:
        a = np.array([x_std, x_mean])
        np.save(Path_output + "%s.npy"%name, a)
    def On_Norm(X):
        y = (X - x_mean)/x_std
        return y
    def Un_Norm(X):
        y = X * x_std + x_mean
        return y
    return X_on, On_Norm, Un_Norm

#==============================================================================
# 定义网络
#==============================================================================
Net = my_Net.Net2 #调用 Net2 网络

torch.manual_seed(0) # 设定随机数种子   
  
def Train(A, N_train, Path_output=''):
    global on_x, un_x, on_y, un_y
    global net
    
    X0 = A[:,(1,4,2)]
    y0 = A[:,3]
    
    X, on_x, un_x = Norm(X0, Path_output=Path_output, name='Norm_x')  #归一化
    y, on_y, un_y = Norm(y0, Path_output=Path_output, name='Norm_y')  #归一化
    
    net = Net()
    
    if GPU:
        net.cuda(0)
        Input = Variable(torch.from_numpy(X).cuda())
        target = Variable(torch.from_numpy(y).cuda())
    else:
        Input = Variable(torch.from_numpy(X))
        target = Variable(torch.from_numpy(y))

    #==============================================================================
    # 损失函数
    #==============================================================================
    criterion = nn.MSELoss()
    
    #==============================================================================
    # 权值更新
    #==============================================================================
    net.zero_grad() # 梯度归零
     
    # create your optimizer
    for i in range(0, N_train):
        optimizer = optim.Adadelta(net.parameters())
#        optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.8)  
#        optimizer = optim.RMSprop(net.parameters(), lr=0.01, alpha=0.9) 
#        optimizer = optim.Adam(net.parameters(), lr=0.01, betas=(0.9, 0.99))
#        optimizer = optim.SGD(net.parameters(), lr = 0.01)

        optimizer.zero_grad() # zero the gradient buffers
        Output = net(Input)
        loss = criterion(Output, target)
        loss.backward()
        optimizer.step() # Does the update
        
        if (i+1)%1000==0:
            Output = net(Input)
            if GPU:
                Out = Output.cpu().data.numpy()
            else:
                Out = Output.data.numpy()
            Out = un_y(Out)
            y_d = Out[:, 0] - y0
            y_d_std = np.std(y_d, ddof = 1)
            print('标准差为', y_d_std)
            print("训练中...", i+1)

    # 保存网络，并输出结果
    if Path_output:
        torch.save(net, Path_output + r'Net.pkl') # 保存网络为文件
        torch.save(net.state_dict(), Path_output + r'Net_Dict.pkl')
    
    Output = net(Input)
    if GPU:
        Out = Output.cpu().data.numpy()
    else:
        Out = Output.data.numpy()
    Out = un_y(Out)
    y_d = Out[:, 0] - y0
    y_d_std = np.std(y_d, ddof = 1)
    p_std = y_d/y_d_std
    index_delete = np.where(np.abs(p_std) > 1.5)
    print('标准差为', y_d_std)
    return index_delete

# =============================================================================
# 将网络转换为函数
# =============================================================================
def net_func(Ri, SOC):
    """
    输入Ri与SOC
    输出 Q
    """
    X = np.array([[Ri, SOC]], dtype='float32')
    X = on_x(X)
    Q = net(Variable(torch.from_numpy(X)))
    Q = un_y(Q.data.numpy())
    return float(Q)

def net_func_M(X):
    X = on_x(X)
    if GPU:
        Q_y = net(Variable(torch.from_numpy(X).cuda()))
        Q_y = un_y(Q_y.cpu().data.numpy())
    else:
        Q_y = net(Variable(torch.from_numpy(X)))
        Q_y = un_y(Q_y.data.numpy())
    return Q_y

#==============================================================================
# 查看训练效果
#==============================================================================
def Ana_Data(A, plot=True):
    X = A[:, (1, 4, 2)]
    y_true = A[:, 3][:, np.newaxis]
    y_p = net_func_M(X)
    y_d = y_p - y_true
    y_d_std = np.std(y_d, ddof = 1)
    p_std = y_d/y_d_std
    #画直方图
    if plot:
        plt.figure()
        plt.hist(y_d, bins=15)
        plt.show()
    return (X, y_true, y_p, y_d, y_d_std, p_std) # P 中元素
    
# =============================================================================
# 主程序
# =============================================================================

# 导入数据
Path_input  = r'.\训练集\训练集全集1.xlsx'
Path_output = r'.\Net_Saved\Net_data_01\\'  #这里可以修改每次保存的文件夹名
# 若目录不存在则新建
if not os.path.exists(Path_output):
    os.makedirs(Path_output)

A = np.array(pd.read_excel(Path_input))
A = A.astype('float32')
if np.median(A[:, 4])<1:
    A[:, 4] *= 100
#A = A[A[:, 4]<=50]
#A = A[A[:, 4]>=0]
# 训练1
index_delete = Train(A, N_train=1000, Path_output='')
P1 = Ana_Data(A, plot=True)

#去偏
A2 = np.delete(A, index_delete[0], axis=0)
print("去除%d个偏差数据"%(len(index_delete[0])), "占比为%4.3f%%"%(100*len(index_delete[0])/len(A)))
#训练2
index_delete2 = Train(A2, N_train=1000, Path_output=Path_output)
P2 = Ana_Data(A2, plot=True)
y_p = P2[2]
y_d = P2[3]
write('保存的数据')

A_con = np.concatenate((A2, y_d), axis=1)
plt.figure()
plt.scatter(A_con[:, 4], A_con[:, 6])
plt.show()

# 分析每段
i1 = 0
k = 1
L = []
for i in range(0, len(A_con)-1):
    if A_con[i, 4] < A_con[i+1, 4] or i==len(A_con)-2:
        i2 = i+1
        temp = A_con[i1, 2]
        cycles = A_con[i1, 5]
        if np.abs(cycles-0)<10:
            soh_t = 1
        elif np.abs(cycles-200)<10:
            soh_t = 0.912
        elif np.abs(cycles-400)<10:
            soh_t = 0.882
        elif np.abs(cycles-600)<10:
            soh_t = 0.851
        
        print('%d : (%d - %d), '%(k-1, i1, i2),'Cycles = %d, '%(cycles), 'T = %d ℃'%(temp),  '↓')
        plt.figure()
        plt.scatter(A_con[i1:i2, 4], y_p[i1:i2, 0]/my_Net.Cn_t_T(A_con[i1, 2]), c='b')
        plt.plot([0, 100], [soh_t, soh_t], c='r')
#        plt.scatter(A_con[i1:i2, 4], A_con[i1:i2, 6])
        plt.show()
        L.append([A_con[i1:i2, 4], y_p[i1:i2, 0]/my_Net.Cn_t_T(A_con[i1, 2])])
        i1 = i2
        k += 1


