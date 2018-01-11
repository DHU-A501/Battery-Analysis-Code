# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 17:45:11 2017

输入电池测试仪测试的Exel文件（需恒温条件）
输出标准放电格式

输出为 W2017_100 格式Exel

批量处理

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
    p1 = os.getcwd() #获取当前目录
    p2 = os.path.dirname(p1) #获取p1上一级目录
    p3 = os.path.dirname(p2) #获取p2上一级目录
    return p3

# =============================================================================
# 获取文件夹下的所有文件，使用正则表达式提取文件名中的数据，并按要求筛选
# =============================================================================
def get_file(Path):
    L0 = []              #存放所有文件的 Path
    for i in Path:        #遍历指定文件夹下的文件, 并存放到 L 列表中
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
        if  ('错误' not in i) and ('充放电' not in i) and ('充电' not in i):  #对文件进行筛选
            L1.append([Bty, SOH, i, File_name, Temp])
    return L1

# =============================================================================
# 获取输入文件夹
# =============================================================================
Path_root = get_Path()  #获取根目录（电池数据上一级目录）
Path = [Path_root + r'\Li-ion-Battery-Data\电池实测2017b\变温恒流放电数据', 
        Path_root + r'\Li-ion-Battery-Data\电池实测2017b\常温恒流放电数据']   #需要导入的文件夹

# =============================================================================
# 设置输出文件夹 若不存在则创建文件夹
# =============================================================================
Path_Out = r'.\恒流放电数据分析结果\\' # 文件夹名在这改，不建议修改，此文件夹可能链接到其它程序
if not os.path.exists(Path_Out):
    os.makedirs(Path_Out)
    
# =============================================================================
# 主程序
# =============================================================================
files = get_file(Path) #所有符合筛选条件的文件及其参数
for Te in files:
    filename = Path_Out + Te[3] + '_R.xlsx'
    # 判断文件是否已存在，若存在则跳过，不存在则输出（若想更新文件请去目录删除该文件并再次运行程序）
    if not os.path.exists(filename):
        print(Te[2])
        
        R = pd.read_excel(Te[2])
        R = R.drop([0,1,2,3])        #去除头部信息
        R = R[['放电容量(mAh)', '放电比容量(mAh/g)', '效率(%)', '放电能量(mWh)']]
        Re = R['放电比容量(mAh/g)']   #读取电压信息
        
        #筛选电流信息
        R = R[R[R.columns[1]]>2000]
        R = R[R[R.columns[2]]<6000]    
        
        #转为矩阵
        A = np.array(R, dtype='float32')
        #根据 电量值 与 电流值 判断是否开始放电
        index_start = 98
        for i in range(1, len(A)):
            if (A[i-1, 3] > A[i, 3] and A[i, 2]<0) or (A[i-1, 1] > A[i, 1] and A[i, 2]<0):
                index_start = i
                break
                
        index_end = 8670
        for i in range(1, len(A)):
            if (A[i-1, 3] > A[i, 3] and A[i, 2]>=0) or (A[i-1, 1] < A[i, 1]-10 and abs(A[i-1, 2]-A[index_start, 2])<10):
                index_end = i-1
                break
            elif abs(A[len(A)-1, 2]-A[index_start, 2]) < 10:
                index_end = len(A)-1
                break
        
        extract_A = A[index_start:index_end+1]
        extract_A[:, 0] -= extract_A[0, 0]
        if extract_A[-1, 0]<2000:
            extract_A[:, 0] *= 5
        soc = extract_A[:, 3]/extract_A[-1, 3]
        extract_A = np.concatenate((extract_A, soc[:, np.newaxis]), axis=1)
        
        #画图
        print(Te[3], '↓')
        plt.figure()
        plt.plot(extract_A[:, 0], extract_A[:, 1])
        plt.show()
        
        # 转换为 pandas 的数据格式并写入exel文件
        d = pd.DataFrame(extract_A)
        d.rename(columns={0:'t/s', 1:'U/mV', 2:'I/mA', 3:'Q/mAh', 4:'SOC'}, inplace = True)  #更改列名
        with pd.ExcelWriter(filename) as writer:
            d.to_excel(writer, sheet_name = str(0))      #写入exel文件中
    


