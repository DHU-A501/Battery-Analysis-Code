# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 16:48:11 2017
从电池测试仪导出Exel常规报表

用于分析随机放电数据

输出为 W2017_S01 格式pandas.Dataframe

@author: WangDian
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import re

# =============================================================================
# 自动获取电池测试数据所在文件夹（必须按照要求放置文件夹，并且不更改名字）
# =============================================================================
def get_Path():
    p1 = os.getcwd()
    p2 = os.path.dirname(p1)
    p3 = os.path.dirname(p2)
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

#==============================================================================
# 对测试仪文件做操作
#==============================================================================
def extract_U(R_u):
    R1 = R_u
    R1.drop([0,1,2], axis=0, inplace=True) #去除头部信息
    R1 = R1[[R1.columns[2], R1.columns[4], R1.columns[5], R1.columns[7], R1.columns[11]]]    #读取第1页矩阵
    
    #去除放电前数据
    #去掉工步所在行
    R1 = R1[R1[R1.columns[1]]>2000]    
    R1 = R1[R1[R1.columns[2]]<10000]
    
    R1.reset_index(inplace=True)       #重置索引
    R1.drop('index', axis=1, inplace=True)  #删除原索引
    
    t = R1[R1.columns[4]][0]  #获取开始时间
    t0 = time.mktime(time.strptime(t, '%Y-%m-%d_%H:%M:%S'))
    for i in range(0, len(R1)):
        t_ = R1[R1.columns[4]][i]
        t_2 = time.mktime(time.strptime(t_, '%Y-%m-%d_%H:%M:%S'))
        R1[R1.columns[4]][i] = t_2 - t0   #记录真实时间
    
    A_u = np.array(R1, dtype = 'float')
    # 判断开始阶段是否充电
    if A_u[0, 2] >= 0:
        for i in range(len(A_u)-1):
            if A_u[i, 1] - A_u[i+1, 1]>50:
                k = i+1
                break
    else:
        k = 0

    A_u = A_u[k:, :]
    A_u[:, 0] -= A_u[0, 0]
    A_u[:, 4] -= A_u[0, 4]
    return A_u

#==============================================================================
# 对内阻文件做操作
#==============================================================================
def extract_R(R_r):
    R2 = R_r
    R2.drop([0], axis=0, inplace=True)     #去除头部信息
    A_r = np.array(R2, dtype = 'float')
    
    #去除放电前数据
    for i in range(len(A_r)-1):
        if A_r[i, 2] - A_r[i+1, 2]>75:
            k = i+1
            break
    A_r = A_r[k:, :]
    
    #时间轴归零
    A_r[:, 0] -= A_r[0, 0]
    A_r[:, 1] -= A_r[0, 1]
    
    #找最后一个小于2600mV的位置 作为结束点
    k2 = np.where(A_r[:,2]<2600)[0][-1] 
    
    #进一步判断是否为结束点
    if A_r[k2+1, 2] - A_r[k2, 2] > 50:
        A_r = A_r[:k2+1, :]
    
    #对内阻测试文件做时间调整
    t_r = A_r[-1][1]
    t_u = A_u[-1][4]
    A_r[:, 1] *= t_u/t_r
    A_r[:, 1] = np.round(A_r[:, 1])
    
    return A_r

# =============================================================================
# 获取输入文件夹
# =============================================================================
Path_root = get_Path()  #获取根目录（电池数据上一级目录）
Path = [Path_root + r'\Li-ion-Battery-Data\电池实测2017b\随机电流放电数据']   #需要导入的文件夹

# =============================================================================
# 设置输出文件夹 若不存在则创建文件夹
# =============================================================================
Path_Out = r'.\随机电流分析结果\\' # 文件夹名在这改，不建议修改，此文件夹可能链接到其它程序
if not os.path.exists(Path_Out):
    os.makedirs(Path_Out)

# =============================================================================
# 主程序
# =============================================================================
L = get_file(Path)
for Te in L:
    filename = Path_Out + Te[3] + '_R.xlsx'
    if not os.path.exists(filename):
        R_u = pd.read_excel(Te[2], sheet_name=0)    #读取测试仪文件
        R_r = pd.read_excel(Te[2], sheet_name=1)    #读取内阻检测文件
        
        A_u = extract_U(R_u)
        A_r = extract_R(R_r)
        
        # 判断数据是否分析正确
        plt.figure()
        plt.plot(A_u[:,4], A_u[:,1], c='k')
        plt.plot(A_r[:,1], A_r[:,2], c='r')
        plt.show()
        # =============================================================================
        # 将时间调整为唯一顺序
        # =============================================================================
        t = int(A_u[-1, 4])
        A_list = []
        for i in range(0, t+1):
            i_u = np.where(A_u[:, 4]==i)
            i_r = np.where(A_r[:, 1]==i)
            # 若不存在i_u则插值
            if len(i_u[0])==0:
                i_u = np.where(A_r[:, 1]==i-1)
            
            U = A_u[i_u[0][-1], 1]
            I = A_u[i_u[0][-1], 2]
            Q = A_u[i_u[0][-1], 3]
            
            # 若不存在i_r则插值
            if len(i_r[0])==0:
                i_r = np.where(A_r[:, 1]==i-1)
                R = (A_r[i_r[0][0], 4] + A_r[i_r[0][0]+1, 4])/2
            else:
                R = A_r[i_r[0][0], 4]
            R /= 5
            A_list.append(np.array([[i, U, I, R, Q]]))
        A = np.concatenate(A_list, axis=0)
        
        # =============================================================================
        # 对内阻做中值滤波
        # =============================================================================
        R_r = A[:, 3].copy()
        R_f = R_r.copy()
        k = 5
        for i in range(k, len(R_f)):
            R_f[i] = np.median(R_r[i-k:i+1])
            
        # 分析滤波效果  
        plt.figure()    
        plt.plot(A[:, 3], c='b')
        plt.plot(R_f, c='r')
        plt.show()
        A[:, 3] = R_f
        # =============================================================================
        # 调整电量
        # =============================================================================
        a = A[:, 4]
        b = np.zeros(len(A))
        for i in range(1, len(A)):
            d = a[i] - a[i-1]
            if d<0:
                d = a[i]
            b[i] = d
        b[0] = a[0]
        
        q = np.zeros(len(b))
        q[0] = b[0]
        for i in range(1, len(b)):
            q[i] = q[i-1] + b[i]
        soc = 1 - q/q[-1]
        B = np.concatenate((A[:,0:4], b[:, np.newaxis], q[:, np.newaxis], soc[:, np.newaxis]), axis=1)
        
        d = pd.DataFrame(B)
        d.rename(columns={0:'t/s', 1:'U/mV', 2:'I/A', 3:'R/mΩ', 4:'单次电量/mAh', 5:'累计电量/mAh', 6:'SOC'}, inplace = True)  #更改列名
        if 1:
            with pd.ExcelWriter(filename) as writer:
                d.to_excel(writer, sheet_name = str(0))      #写入exel文件中












