## 该文档为 神经网络 文件夹 的说明文档
***
#### get_TrainData.py
该程序负责将待训练的数据整合到一个Excel表中

输入：（二选一）
1. 恒流放电对SOC分析结果 文件夹。
2. 待整合数据集 文件夹（提前放置好自己想要的数据即可）
>在61-64行处做选择，if后面 1对应选择一，0对应选择二。

```
if 1:
    Path = [Path_root + r'\Battery-Analysis-Code\恒流放电数据分析\恒流放电对SOC分析结果']   #需要导入的文件夹
else:
    Path = [r'.\待整合数据集']
```

输出：在 训练集 文件夹下，文件名在代码最后段修改。
```
if 1:
    with pd.ExcelWriter(Path_Out + r'训练集全集1.xlsx') as writer: #输出的文件名在这修改
        d.to_excel(writer, sheet_name = '0')
```
>若已存在同名文件，再次运行时会覆盖！

***

#### my_Net.py
这是一个包含网络定义的头文件
其中的每个类代表一个单独神经网络，可以继续定义，只要在调用时调用相应的网络即可。
> 最好先学习 PyTorch 的相关知识！

***

#### ANN.py
该程序是网络训练程序
- 输入：get_TrainData.py 输出在训练集文件夹下的excel文件。
- 输出：在Net_Saved文件夹下输出，每次输出为一个包含多个文件的文件夹。
- 下列代码修改输出与输入的文件名
```
Path_input  = r'.\训练集\训练集全集1.xlsx' #这里可以修改每次输入的文件名
Path_output = r'.\Net_Saved\Net_data_01\\'  #这里可以修改每次保存的文件夹名
```
在此处调用头文件中的网络
`Net1 = my_Net.Net2  #调用 Net2 网络`
这里设定随机种子值，每个种子值对应一个固定的训练结果！
`torch.manual_seed(0) # 设定随机数种子`

***

#### 修改训练参数
**ANN.py**中有两处需要修改
- 第一处
```
def Train(A, N_train, Path_output=''):
    global on_x, un_x, on_y, un_y
    global net
    
    X0 = A[:,(1,4,2)]   #修改输入参数
    y0 = A[:,3]
```
- 第二处
```
def Ana_Data(A, plot=True):
    X = A[:, (1, 4, 2)]   #修改输入参数
    y_true = A[:, 3][:, np.newaxis]
```
若改变了输入参数数量则需要去 **my_Net.py** 头文件中进行修改
```
class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.wb1   = nn.Linear(2, 50)  #这里 2 即为输入参数数量
        self.wb2   = nn.Linear(50, 25)
        self.wb3   = nn.Linear(25, 10)
        self.wb4   = nn.Linear(10, 1)
    def forward(self, x):
        x = F.relu(self.wb1(x))
        x = F.relu(self.wb2(x))
        x = F.relu(self.wb3(x))
        x = self.wb4(x)
        return x

```
***
