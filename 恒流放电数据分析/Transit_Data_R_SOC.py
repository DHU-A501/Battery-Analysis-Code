# -*- coding: utf-8 -*-
"""
Created on Mon May  8 16:54:30 2017

提取整数SOC下的数据

@author: WangDian
"""

import numpy as np
import pandas as pd
import os
import re

# =============================================================================
# 线性插值函数 (x1, y1) 与 (x2, y2) 之间线性求 (x3, y3)
# =============================================================================
def Linear_In(x1, x2, y1, y2, x3):    
    k = (y2 - y1)/(x2 - x1)
    y3 = k*(x3 - x1) + y1
    return y3

# =============================================================================
# 提取文件
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
        if  ('错误' not in i) and ('充放电' not in i) and ('三电压检测' in i):     #对文件进行筛选
            L1.append([Bty, SOH, i, File_name, Temp])
    return L1

Path = [r'.\恒流放电内阻分析结果']   #需要导入的文件夹  恒流放电

# =============================================================================
# 设置输出文件夹 若不存在则创建文件夹
# =============================================================================
Path_Out = r'.\恒流放电对SOC分析结果\\' # 文件夹名在这改，不建议修改，此文件夹可能链接到其它程序
if not os.path.exists(Path_Out):
    os.makedirs(Path_Out)
    
# =============================================================================
# 主程序
# =============================================================================   
L = get_file(Path)
for Te in L:
    filename = Path_Out + Te[3] + '_SOC.xlsx' 
    if not os.path.exists(filename):
        
        print(Te[2])
        R = pd.read_excel(Te[2], sheet_name=0)
        a = np.array(R, dtype = 'float')
        
        a = np.column_stack((a, -(a[:, 7] - a[-1, 7])/a[-1, 7]))                     #计算SOC
        
        a2 = a[:,[1,4,5,8,7]]
        
        a2[:, 4] -= a2[-1, 4] # 将容量翻正
        
        a3 = np.zeros([101, 5])
        k = 0
        for soc in np.linspace(100, 0, 101):
            if soc >=1:
                for i in range(0, len(a2)-1):
                    if ((a2[i+1, 3] < soc/100) and (a2[i, 3] >= soc/100)):
                        a3[k, :] = Linear_In(a2[i+1, 3], a2[i, 3], a2[i+1, :], a2[i, :], soc/100)
                        a3[k, 3] = soc
                        k += 1
            elif soc == 0:
                a3[-1, :] = a2[-1, :]
        
        a4 = np.zeros([6, 5])
        k = 0
        for soc in np.linspace(100, 0, 6):
            if soc >=1:
                for i in range(0, len(a2)-1):
                    if ((a2[i+1, 3] < soc/100) and (a2[i, 3] >= soc/100)):
                        a4[k, :] = Linear_In(a2[i+1, 3], a2[i, 3], a2[i+1, :], a2[i, :], soc/100)
                        a4[k, 3] = soc
                        k += 1
            elif soc == 0:
                a4[-1, :] = a2[-1, :]
                
        d1 = pd.DataFrame(a3)
        d2 = pd.DataFrame(a4)
        d1.rename(columns={0:'U/mV',1:'内阻/mOhm', 2:'截距', 3:'SOC', 4:'Q/mAh'}, inplace = True)  #更改列名
        d2.rename(columns={0:'U/mV',1:'内阻/mOhm', 2:'截距', 3:'SOC', 4:'Q/mAh'}, inplace = True)  #更改列名
        
        with pd.ExcelWriter(filename) as writer:
            d1.to_excel(writer, sheet_name = '101点SOC数据')      #写入exel文件中
            d2.to_excel(writer, sheet_name = '6点SOC数据')      #写入exel文件中










