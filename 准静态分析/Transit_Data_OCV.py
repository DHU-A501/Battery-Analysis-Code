# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 16:48:11 2017
从电池测试仪导出Exel常规报表

用于分析准静态文件（必须以 0%电量 准静态电压结尾）

输出为 W2017_S01 格式 pandas.Dataframe

@author: WangDian
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
def get_file(Paths):
    L0 = []              #存放所有文件的 Path
    for i in Paths:        #遍历指定文件夹下的文件, 并存放到 L 列表中
        for root,dirs,files in os.walk(i):
            for filespath in files:
                L0.append(os.path.join(root,filespath))
    L1 = []
    for i in L0:
        P_Bty = re.search( r'OLD(\d+)_', i)   #电池编号格式
        P_SOH = re.search( r'_(\d+)_', i)     #循环次数格式
        P_Temp = re.search(r'_([-]*\d+)°', i)     #电池温度
        P_File = re.search( r'(B_OLD.+)\.', i)   #电池文件名（不包括扩展名）
        Bty = P_Bty.group(1)
        SOH = int(P_SOH.group(1))
        Temp = int(P_Temp.group(1))
        File_name = P_File.group(1)
        if  ('错误' not in i) and ('OCVC' not in i):     #对文件进行筛选
            L1.append([Bty, SOH, i, File_name, Temp])
    return L1

# =============================================================================
# 提取OCV
# =============================================================================
def SOC_OCV_Fun(Path):
    R1 = pd.read_excel(Path, sheet_name=0)    #读取第1页
    R1 = R1.drop([0,1,2,3])        #去除头部信息
    
    R1 = R1[['放电容量(mAh)', '放电比容量(mAh/g)', '效率(%)', '放电能量(mWh)']]

    #读取第2页    
    try:
        R2 = pd.read_excel(Path, sheet_name=1)    
        R2 = R2.drop([0,1])            #去除头部信息
        R2 = R2[['放电容量(mAh)', '放电比容量(mAh/g)', '效率(%)', '放电能量(mWh)']]
        R = pd.concat([R1, R2], axis=0)  #若有两个表则合并
    except:
        R = R1
    
    Re = R['放电比容量(mAh/g)']   #读取电压信息
    
    R = R[R[R.columns[1]]>1000]    #筛选电量信息
    
    Ae = np.array(Re, dtype = 'float')    #Re转换为矩阵
    Ae = Ae[Ae < 2000]                    #筛选电量信息 
    
    #判断是否重复放电至2.5V
    N_OD = -1   #存储过放电次数
    for i in range(0, len(Ae)//2-1):
        if Ae[i*2+1] > Ae[i*2+3]+0.2:
            N_OD += 1
    print("过放电%2d 次"%N_OD)
    if N_OD:
        Ae = Ae[:len(Ae)-N_OD*2]        
    
    A = np.array(R, dtype = 'float') 
    plt.figure(num = 1)
    
    plt.plot(A[:,0], A[:,1])
    plt.show()
    
    L = []        #存储放电时间前一刻节点
    for i in range(1, len(A)):
        if abs(A[i-1][2] - 0)<5 and abs(A[i][2] - (-1600))<5:
            L.append(i-1)
    L.append(len(A)-1) #将最后0%时刻补上
    if N_OD:
        L = L[:-N_OD]
    
    L1 = []        #存储静置最后电压
    for i in L:
        L1.append(A[i][1])
    
    Ae1 = np.zeros([len(Ae), 2])
    Ae1[:, 0] = Ae
    for i in range(0, len(L1)):
        Ae1[2*i, 1] = L1[i]
        
    Ae2 = np.zeros([len(L1), 4])               #最后的矩阵Ae2
    Ae2[:, 0] = np.array(L1)                   #写入电压
    for i in range(1, len(Ae2)):
        Ae2[i, 1] = Ae[2*i-1]                  #写入单次电量
    for i in range(1, len(Ae2)):
        Ae2[i, 2] = Ae2[i, 1] + Ae2[i-1, 2]    #写入累计电量
    Ae2[:, 3] = (Ae2[-1, 2]-Ae2[:, 2])/Ae2[-1, 2]    #计算SOC
    
    Ae3 = pd.DataFrame(Ae2)
    Ae3.rename(columns={0:'电压/mV', 1:'单次电量/mAh', 2:'累计电量/mAh', 3:'SOC'}, inplace = True)   #更改列名
    # 拟合值
    p = np.polyfit(Ae2[:, 3]*100, Ae2[:, 0]/1000, 3)
    return Ae3, p

# =============================================================================
# 获取输入文件夹
# =============================================================================
Path_root = get_Path()  #获取根目录（电池数据上一级目录）
Path = [Path_root + r'\Li-ion-Battery-Data\电池实测2017b\常温OCV数据']   #需要导入的文件夹

# =============================================================================
# 设置输出文件夹 若不存在则创建文件夹
# =============================================================================
Path_Out = r'.\准静态分析结果\\' # 文件夹名在这改，不建议修改，此文件夹可能链接到其它程序
if not os.path.exists(Path_Out):
    os.makedirs(Path_Out)

# =============================================================================
# 主程序
# =============================================================================
files = get_file(Path)
L_p = []
for file in files:
    filename = Path_Out + file[3] + '_R.xlsx'
    # 判断文件是否已存在，若存在则跳过，不存在则输出（若想更新文件请去目录删除该文件并再次运行程序）
    if not os.path.exists(filename):
        R, p= SOC_OCV_Fun(file[2])
        P = pd.DataFrame(p[np.newaxis, :])
        with pd.ExcelWriter(filename) as writer:
            R.to_excel(writer,sheet_name = 'OCV-SOC')            #写入exel文件中
            P.to_excel(writer,sheet_name = '三次拟合系数')            #写入三次拟合系数
        L_p.append([file[1], p])






















