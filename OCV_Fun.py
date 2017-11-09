# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 16:48:11 2017
从电池测试仪导出Exel常规报表

用于分析准静态文件（必须以 0%电量 准静态电压结尾）

输出为 W2017_S01 格式pandas.Dataframe

@author: WangDian
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as pl

def SOC_OCV_Fun(Path):
    R1 = pd.read_excel(Path, sheetname=0)    #读取第1页
    R2 = pd.read_excel(Path, sheetname=1)    #读取第2页
    
    
    R1 = R1.drop([0,1,2,3])        #去除头部信息
    R2 = R2.drop([0,1])            #去除头部信息
    
    Re1 = R1[R1.columns[4]]   #读取电压信息
    Re2 = R2[R1.columns[4]]   #读取电压信息
    
    R1 = R1[[R1.columns[2], R1.columns[4], R1.columns[5], R1.columns[7]]]    #读取第1页矩阵
    R2 = R2[[R2.columns[2], R2.columns[4], R2.columns[5], R2.columns[7]]]    #读取第2页矩阵
    
    R1 = R1[R1[R1.columns[1]]>1000]    #筛选电量信息
    R2 = R2[R2[R2.columns[1]]>1000]    #筛选电量信息 
    
    Ae1 = np.array(Re1, dtype = 'float')    #Re转换为矩阵
    Ae2 = np.array(Re2, dtype = 'float')    #Re转换为矩阵
    
    Ae = np.concatenate((Ae1,Ae2),axis=0)    #合并电量信息 
    Ae = Ae[Ae < 2000]                       #筛选电量信息 
    
    A1 = np.array(R1, dtype = 'float')
    A2 = np.array(R2, dtype = 'float')
    
    A3 = np.concatenate((A1,A2),axis=0)      #总矩阵
    
    pl.figure(num = 1)
    
    pl.plot(A3[:,0], A3[:,1])
    pl.show()
    
    L = []        #存储放电时间前一刻节点
    for i in range(1, len(A3)):
        if abs(A3[i-1][2] - 0)<5 and abs(A3[i][2] - (-1600))<5:
            L.append(i-1)
    L.append(len(A3)-1) #将最后0%时刻补上
    
    L1 = []        #存储静置最后电压
    for i in L:
        L1.append(A3[i][1])
    
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
    
    return Ae3

FileName = "B_OLD073_0002_240mAh_25°C_OCV"
Path = "W:\WangDian\实验2017b\电池实测\常温OCV数据\\" + FileName + ".xls"
R = SOC_OCV_Fun(Path)
with pd.ExcelWriter('W:\WangDian\实验2017b\Python分析\准静态分析程序\\' + FileName + '_R.xls') as writer:
    R.to_excel(writer,sheet_name = str(0))            #写入exel文件中




























