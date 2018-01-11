# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 21:47:24 2017

通过实验室测试的变温的Exel文件来计算压降与内阻

输出为 W2017_100 格式Exel

批量处理

@author: WangDian
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
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

# =============================================================================
# 获取输入文件夹
# =============================================================================
Path_root = get_Path()  #获取根目录（电池数据上一级目录）
Path = [Path_root + r'\Li-ion-Battery-Data\电池实测2017b\变温恒流放电数据', 
        Path_root + r'\Li-ion-Battery-Data\电池实测2017b\常温恒流放电数据']   #需要导入的文件夹

# =============================================================================
# 设置输出文件夹 若不存在则创建文件夹
# =============================================================================
Path_Out = r'.\恒流放电内阻分析结果\\' # 文件夹名在这改，不建议修改，此文件夹可能链接到其它程序
if not os.path.exists(Path_Out):
    os.makedirs(Path_Out)
    
# =============================================================================
# 主程序
# =============================================================================   
L = get_file(Path)
for Te in L:    
    filename = Path_Out + Te[3] + '_R.xlsx'
    # 判断文件是否已存在，若存在则跳过，不存在则输出（若想更新文件请去目录删除该文件并再次运行程序）
    if not os.path.exists(filename):
        print(Te[2])
        Err = 0 # 错误指令 0为无错
        
        R = pd.read_excel(Te[2], sheet_name=1)
        
        R = R.drop(0) #去掉第一行
        
        a = np.array(R, dtype = 'float')
        a = np.delete(a, 0, axis=1)    #去掉第一列
        a = a[a[:,3]>1]     #去掉等于0的错误情况
        
        d1 = []                                 
        d2 = []
        try:
            for i in range(1,len(a)):                     #去头尾
                if (a[i-1, 1] - a[i, 1])>65:
                    d1.append(i)   
                elif (a[i, 1] - a[i-1, 1])>90:
                    d2.append(i)
                elif np.abs(a[-1, 1] - 2500)<10:
                    d2.append(len(a))
            if d1==[]:
                d1 = [0]
            k = 0
            for t1 in d1:
                for t2 in d2:
                    if t2-t1>500:
                        a = a[t1:t2, :]
                        k = 1
                        break
                if k==1:
                    break
        except:
            Err = 1
            print("---------------数据出错，请检查数据!------------")
            
        pl.plot(a[:,0],a[:,1])
        pl.show()
        
        a[:, 0] = a[:, 0] - a[0, 0]                                     #首行时间归零
        b = a[:, 0:2]       
        b = np.column_stack((b, a[:, 1] - a[:, 2]))                     #计算两次压降值
        b = np.column_stack((b, b[:, 2] + a[:, 3]))
        
        x = np.array([5, 10])
        c = np.zeros([len(b), 6])
        j = 0
        for i in b:
            p = np.polyfit(x, i[2:4], 1)                                   #拟合直线
            i2 = np.concatenate((i, p), axis=0)
            c[j, :] = i2.transpose()
            j += 1
        
        c = np.column_stack((c, np.zeros([len(c), 3])))
        
        c[:, 6] = -1.6                                                       #加上电流值
        
        c[0, 7] = 0
        for i in range(1, len(c)):
            c[i, 7] = c[i-1, 7] + 5/18 * (c[i, 0]-c[i-1, 0]) * c[i, 6]          #加上电量值
        c[:, 8] = 1 - c[:, 7]/c[-1, 7]
        pl.plot(c[:,0],c[:,4])
        pl.show()
        
        d = pd.DataFrame(c)
        d.rename(columns={0:'t/s', 1:'U/mV', 2:'dU1/mV', 3:'dU2/mV', 4:'内阻/mOhm', 5:'截距', 6:'I/A', 7:'Q/mAh', 8:'SOC'}, inplace = True)  #更改列名
        
        if Err == 0:
                with pd.ExcelWriter(filename) as writer:
                    d.to_excel(writer, sheet_name = str(0))      #写入exel文件中
        else:
            print(Te[3], "-数据出错，请检查数据!-")






