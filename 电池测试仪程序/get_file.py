# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 16:50:18 2017

生成电池随机放电工步文件

@author: WangDian
"""

import struct
import numpy as np

def num2bytes(x):
    a = struct.pack("d", x).hex()
    b = bytes().fromhex(a)
    return b

def replace_IQ(set_I, set_Q, step, t):
    """
    输入待设置的电流、电量、工步及Bytes串
    返回Bytes串
    """
    Index = [(step-1)*80 + 28, (step-1)*80 + 108] #从28开始，每节一小段
    t1 = t_zero
    t_I = num2bytes(set_I)
    t_Q = num2bytes(set_Q)
    
    # 分区修改
    t_flag = bytes().fromhex('020c 0000 0101 0000')
    t1 = t_flag + t1[9:]
    t1 = t1[0:16] + t_I + t1[24:]  #合并结果为新段落  I:16-24
    t1 = t1[0:48] + t_Q + t1[56:]  #合并结果为新段落  Q:48-56
    t = t[0:Index[0]] + t1 + t[Index[1]:] #将新段落替换加入原串
    return t

#==============================================================================
# 读取源文件
#==============================================================================
Bin = open('源.job','rb')
t = Bin.read()
t_head = t[:28]     #文件头
t_zero = t[668:748]  #全0段落

#==============================================================================
# 设置待修改的值
#==============================================================================
np.random.seed(2)  #修改随机种子
Set = np.random.rand(60, 2)

Set[:, 0] *= 5 #电流值设为0-5
Set[:, 1] = Set[:, 1] * 13 + 2 #电量设为 2-15

Q_all = np.sum(Set[:,1])  #估算总放电容量
T = 3.6 * Set[:,1]/Set[:,0]  #估算各工步时间
T_all = np.sum(T)
I_mean = 3.6 * Q_all/T_all
t2 = t
for i in  range(len(Set)):
    t2 = replace_IQ(float(Set[i][0])*1000, float(Set[i][1]), i+1, t2)

print('总放电容量为：%6.4f mAh'%Q_all)
print('总放电时间为：%6.4f 秒'%T_all)
print('平均放电电流：  %6.4f A'%I_mean)
#==============================================================================
# 输出成二进制文件
#==============================================================================
# 在引号里输入待输出的文件名，注意若重名将自动覆盖！
with open("t3.job", "wb") as file:
    file.write(t2)







