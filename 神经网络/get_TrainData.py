# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 11:15:05 2017

提取30℃下的数据
Q = f(Ri, T)
s.t. SOC=100%/50%/20%

@author: WangDian
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
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
# 获取文件信息
# =============================================================================
def get_data(Paths):
    L0 = []              #存放所有文件的 Path
    for i in Paths:        #遍历指定文件夹下的文件, 并存放到 L 列表中
        for root,dirs,files in os.walk(i):
            for filespath in files:
                L0.append(os.path.join(root,filespath))
    files_Info = []
    for path in L0:
        P_Btr = re.search( r'OLD(\d+)_', path)   #电池编号格式
        P_Cy = re.search( r'_(\d+)_', path)     #循环次数格式
        P_Temp = re.search(r'_([-]*\d+)°', path)     #电池温度
        P_File = re.search( r'(B_OLD.+)\.', path)   #电池文件名（不包括扩展名）
        Btr = P_Btr.group(1)
        Cy = int(P_Cy.group(1))
        try:
            Temp = int(P_Temp.group(1))
        except:
            Temp = 25
        File_name = P_File.group(1)
        
        #对SOH进行筛选
        if  ('错误' not in path)    :   #and abs(Cy-Cycles)<10  and Temp==50
            files_Info.append([Btr, Cy, path, File_name, Temp])
    return files_Info

# =============================================================================
# 获取输入文件夹
# =============================================================================
Path_root = get_Path()  #获取根目录（电池数据上一级目录）
#注意：这里二选一，第一行为待整合数据集，第二行为调用 恒流放电对SOC分析结果 文件夹内文件

if 1:
    Path = [Path_root + r'\Battery-Analysis-Code\恒流放电数据分析\恒流放电对SOC分析结果']   #需要导入的文件夹
else:
    Path = [r'.\待整合数据集']

# =============================================================================
# 设置输出文件夹 若不存在则创建文件夹
# =============================================================================
Path_Out = r'.\训练集\\' # 文件夹名在这改，不建议修改，此文件夹可能链接到其它程序
if not os.path.exists(Path_Out):
    os.makedirs(Path_Out)
    
# =============================================================================
# 主程序
# =============================================================================
files = get_data(Path)
xy = []
for Te in files:
    A = np.array(pd.read_excel(Te[2], sheet_name=0))    
    x = np.concatenate([A[:, 0][:, np.newaxis],        # 0:'U/mV'
                        A[:, 1][:, np.newaxis],        # 1:'Ri/mΩ'
                        np.ones([len(A), 1])*Te[4],    # 2:'T/℃'
                        np.ones([len(A), 1])*A[0, 4],  # 3:'Q/mAh'
                        A[:, 3][:, np.newaxis],        # 4:'SOC/%'
                        np.ones([len(A), 1])*Te[1]], axis=1)  # 5:'Cycles'
    xy.append(x)
    
X = np.concatenate(xy, axis=0)

X_corrcoef = np.corrcoef(X.T) #相关系数
# =============================================================================
# 写入exel文件中
# =============================================================================
d = pd.DataFrame(X)
d.rename(columns={0:'U/mV',1:'Ri/mΩ', 2:'T/℃', 3:'Q/mAh', 4:'SOC/%', 5:'Cycles'}, inplace = True)  #更改列名
if 1:
    with pd.ExcelWriter(Path_Out + r'训练集全集1.xlsx') as writer: #输出的文件名在这修改
        d.to_excel(writer, sheet_name = '0')


















