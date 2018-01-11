# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 13:58:43 2017

EKF函数

输入量为pandas读取exel并转换为numpy矩阵

R、Q为默认值可改动

电阻是否随测试变化，变化: Res = True, 不变化：Res = False

@author: WangDian
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from   torch.autograd import Variable
import os
import re
import my_Net

# =============================================================================
# 自动获取电池测试数据所在文件夹（必须按照要求放置文件夹，并且不更改名字）
# =============================================================================
def get_Path():
    p1 = os.getcwd() #获取当前目录
    p2 = os.path.dirname(p1) #获取p1上一级目录
    p3 = os.path.dirname(p2) #获取p2上一级目录
    return p3

# =============================================================================
# 提取文件
# =============================================================================
def get_file(Path):
    L0 = []              #存放所有文件的 Path
    for i in Path:        #遍历指定文件夹下的文件, 并存放到 L 列表中
        for root,dirs,files in os.walk(i):
            for filespath in files:
                L0.append(os.path.join(root,filespath))
    files = []
    for i in L0:
        P_Bty = re.search( r'OLD(\d+)_', i)   #电池编号格式
        P_SOH = re.search( r'_(\d+)_', i)     #循环次数格式
        P_Temp = re.search(r'_([-]*\d+)°', i)     #电池温度
        P_File = re.search( r'(B_OLD.+)\.', i)   #电池文件名（不包括扩展名）
        Bty = P_Bty.group(1)
        SOH = int(P_SOH.group(1))
        Temp = int(P_Temp.group(1))
        File_name = P_File.group(1)
        if  ('错误' not in i):     #对文件进行筛选
            files.append([Bty, SOH, i, File_name, Temp])
    return files

# =============================================================================
# #建立OCV查找表
# =============================================================================
Prmt = np.array(pd.read_excel('OCV拟合系数.xlsx'))
def Cf_OCV(Cf):
    k = -1
    for i in range(0, len(Prmt)-1):
        if Prmt[i+1, 0]<Cf<=Prmt[i, 0]:
            k = i
            break
        elif Cf > Prmt[0, 0]:
            k = 0
        elif Cf < Prmt[-1, 0]:
            k = len(Prmt)-2
    p = (Prmt[k+1, :] - Prmt[k, :])/(Prmt[k+1, 0] - Prmt[k, 0]) * (Cf - Prmt[k, 0]) + Prmt[k, :]
    return p[1:]

# =============================================================================
# 归一化 解包
# =============================================================================
def Norm_read(Path):
    p = np.load(Path)
    try:
        x_std = p[0, :]
        x_mean = p[1, :]
    except:
        x_std = p[0]
        x_mean = p[1]
    def On_Norm(X):
        y = (X - x_mean)/x_std
        return y
    def Un_Norm(X):
        y = X * x_std + x_mean
        return y
    return On_Norm, Un_Norm

def Norm_read_all(Path):
    global on_x, un_x, on_y, un_y
    on_x, un_x = Norm_read(Path + r'Norm_x.npy')
    on_y, un_y = Norm_read(Path + r'Norm_y.npy')

# =============================================================================
# 定义网络
# =============================================================================
net = my_Net.Net2() #调用时别忘了括号
Path_Norm = r'.\Net_Saved\Net_data_01\\'           #提取归一化参数
Path_Net = r'.\Net_Saved\Net_data_01\Net_Dict.pkl' #提取神经网络参数
net.load_state_dict(torch.load(Path_Net))
Norm_read_all(Path_Norm)

# =============================================================================
# 设置net函数
# =============================================================================
def net_func(Ri, SOC, T):
    X = np.array([[Ri, SOC, T]], dtype='float32')
    X = on_x(X)
    Q = net(Variable(torch.from_numpy(X)))
    Q = un_y(Q.data.numpy())
    return float(Q)

def net_func_M(X):
    X = on_x(X)
    Q_y = net(Variable(torch.from_numpy(X)))
    Q_y = un_y(Q_y.data.numpy())
    return Q_y

# =============================================================================
# 更新值函数
# =============================================================================
def fun_gen(p, Bias):
    def fun0(x):
        y = np.polyval(p, x)
        y = y - Bias/1000
        return y
    
    def fun0_d(x):
        p_d = p[0:-1].copy()
        #对多项式求导得出新的系数
        for i in range(0, len(p_d)):
            p_d[len(p_d)-i-1] *= (i+1)
        y = np.polyval(p_d, x)
        y *= 1.5
        return y
    
    def fun0_H(x):
        y = np.array([[1, fun0_d(x)]])    # H矩阵
        return y
    return fun0, fun0_d, fun0_H

# =============================================================================
# 更新 Bias 的函数
# =============================================================================
def Bias_get(Cn_mAh):
    if Cn_mAh > 2700:
        Bias = -0.12*(Cn_mAh-2700) + 90
    elif Cn_mAh > 2650:
        Bias = 0.8*(Cn_mAh-2650) + 50
    else:
        Bias = 50
    return Bias

# =============================================================================
# EKF主函数
# =============================================================================
def EKF_new(file, update=True):
    global Um
    Um = np.array(pd.read_excel(file[2]))

# =============================================================================
# 电池参数
# =============================================================================
    Rp = 39e-3        #一级电容  /Ohm      32.93    40.618      39e-3 
    Cp = 1058        #一级电阻  /F       1009.7175    566     1058
    Cn_mAh = 2700
    Cn = Cn_mAh * 3.6    #总电量/A*s    3.2 * 3600
    nk = 100           #nk = 1/3.6; %转换系数  1s*1A = (1/3.6)mAh
    Bias = 30          #SOH的变量
    Ts = 1             #间隔时间  /s
    # =============================================================================
    # 滤波器参数
    # =============================================================================
    R = 1
    Q0 = 1e-1
    I0 = np.identity(2)    #单位矩阵
    Q = Q0 * I0
    
    A = np.array([[1-Ts/(Cp*Rp), 0], [0, 1]])
    B = np.array([[Ts/Cp], [Ts*nk/Cn]])
    
    # =============================================================================
    # 参数传入
    # =============================================================================
    # 变电流数据
    I_b = Um[:, 2]/1000 #电流   设置放电电流为负 /A
    Q_b = np.abs(Um[:, 5]) #电量
    U_b = Um[:, 1] #电压
    R_b = Um[:, 3] #内阻
    T_b = file[4]
    # =============================================================================
    # 初始化
    # =============================================================================
    L = len(Um)
    X = [[]] * L             #存放变量
    X_z = [[]] * L           #存放预测量
    P = [[]] * L             #存放方差
    P_z = [[]] * L           #存放预测方差
    K = [[]] * L
    
    X[0] = np.array([[0], [100]])
    P[0] = np.array([[1, 0], [0, 10]])
    
    p = Cf_OCV(Cn_mAh)#Cn_mAh
    fun0, fun0_d, fun0_H = fun_gen(p, Bias)
    
    # =============================================================================
    # 开始滤波
    # =============================================================================
    Cf_L = []
    for k in range(0, L-1):
        X_z[k] = np.dot(A, X[k]) + B * I_b[k]
        P_z[k] = np.dot(np.dot(A, P[k]), A.transpose()) + Q
        H = fun0_H(X[k][1][0])
        K[k] = np.dot(P_z[k], H.transpose()) / (np.dot(np.dot(H, P_z[k]), H.transpose()) + R)
        X[k+1] = X_z[k] + K[k] * ((U_b[k+1] - R_b[k+1] * I_b[k])/1000 - (fun0(X[k][1][0]) + X_z[k][0][0])) #R_b[k+1]
        
        if X[k+1][1] < 0:
            X[k+1][1] = 0
        P[k+1]   = np.dot((I0 - np.dot(K[k], H)), P_z[k])
        
        #更新OCV相关函数
        if update:
            SOC = X[k][1]
            Ri = R_b[k+1]
            Cn_mAh = net_func(Ri, SOC, T_b)  #获得容量
            Cn = 3.6 * Cn_mAh
            B = np.array([[Ts/Cp], [Ts*nk/Cn]]) 
            p = Cf_OCV(Cn_mAh)
            Bias= Bias_get(Cn_mAh)
            fun0, fun0_d, fun0_H = fun_gen(p, Bias)
            Cf_L.append(Cn_mAh)
        else:
            continue
    
    Cf = np.array(Cf_L)
    X1 = np.zeros([L+1, 5])
    for k in range(0, L):
#        X1[k][0] = X[k][0][0]                        # 估测 Up
        X1[k][1] = X[k][1][0]                        # 估测 SOC
        X1[k][2] = 100 * (1-Q_b[k]/Q_b[-1])          # 电流积分 SOC
        X1[k][3] = U_b[k]                            # 实测电压 Uo
        X1[k][4] = X1[k][1] - X1[k][2]               # SOC误差
    if update:
        X1[0:2, 0] = Cf[0:2]    # 估测的容量 
        X1[2:, 0] = Cf
    else:
        X1[:, 0] = Cn_mAh    # 容量
        
    plt.figure()
    plt.plot(X1[:, 1], 'r')
    plt.plot(X1[:, 2], 'k')
    plt.show()
    return X1

# =============================================================================
# 导入数据
# =============================================================================

# 获取输入文件夹
Path_root = get_Path()  #获取根目录（电池数据上一级目录）
Path = [Path_root + r'\Battery-Analysis-Code\随机电流放电数据分析\随机电流分析结果']   #需要导入的文件夹
files = get_file(Path)

file = files[2]
print(file[3], '未更新↓')
X1= EKF_new(file, update=False)
print('标准差: ', np.std(X1[:, 1] - X1[:, 2]))
print(file[3], '更新↓')
X2= EKF_new(file, update=True)
print('标准差: ', np.std(X2[:, 1] - X2[:, 2]))

# 画图
plt.plot(X2[:, 0])
#plt.plot(X2[:, 0]/my_Net.Cn_t_T(file[4]))
plt.show()
plt.plot(X2[:, 2], X2[:, 1]-X2[:, 2])
plt.show()

#储存数据
d1 = pd.DataFrame(X1)
d2 = pd.DataFrame(X2)
d1.rename(columns={0:'Cf/mAh', 1:'预测SOC', 2:'真实SOC', 3:'测量电压/mV', 4:'SOC误差'}, inplace = True)  #更改列名
d2.rename(columns={0:'Cf/mAh', 1:'预测SOC', 2:'真实SOC', 3:'测量电压/mV', 4:'SOC误差'}, inplace = True)  #更改列名
if 1:
    with pd.ExcelWriter('.\仿真结果\\' + file[3] + '_EKFOCV.xlsx') as writer:
        d1.to_excel(writer, sheet_name = '未更新')      #写入exel文件中
        d2.to_excel(writer, sheet_name = '更新')      #写入exel文件中




