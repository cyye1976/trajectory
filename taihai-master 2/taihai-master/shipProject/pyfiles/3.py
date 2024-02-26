import warnings
import argparse
warnings.filterwarnings('ignore')
import numpy as np
np.random.seed(1337)
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import  MinMaxScaler
import time

parser = argparse.ArgumentParser(description="path")
parser.add_argument('-i', '--inputPath',default='./')
parser.add_argument('-o', '--outputPath',default='./')
args = parser.parse_args()
if not os.path.exists(args.outputPath):
    os.makedirs(args.outputPath)
start_time=time.time()
#归一化建模
scaler1=MinMaxScaler()
scaler2=MinMaxScaler()
scaler3=MinMaxScaler()
scaler4=MinMaxScaler()
scaler5=MinMaxScaler()
scaler6=MinMaxScaler()

temp_data = pd.DataFrame()
for value in os.listdir(args.inputPath+'data/all_data/'):
    data=pd.read_csv(args.inputPath+'data/all_data/'+value)
    pd.set_option('mode.chained_assignment', None)
    temp_data = pd.concat([temp_data,data],axis=0)
#位置信息
scaler1.fit(temp_data[['latitude']].values)
scaler2.fit(temp_data[['longitude']].values)
#速度信息
scaler3.fit(temp_data[['speed']].values)
scaler4.fit(temp_data[['ac']].values)
#航向信息
scaler5.fit(temp_data[['cog']].values)
scaler6.fit(temp_data[['rot']].values)
#预处理
path=args.inputPath+'data/航道 1/'
x_train1=[]
x_longest=0
# index 一艘船舶轨迹文件夹的序号,value 为船舶轨迹文件夹的名字
for index,value in (enumerate(os.listdir(path))):
            temp_list = []
            data1 = pd.read_csv(path + value )
            # print('data1:',data1)
            pd.set_option('mode.chained_assignment', None)
            df = data1[['latitude','longitude','speed','ac','cog','rot']]
            # print('df1:',df)
            df['latitude'] = scaler1.transform(df[['latitude']])
            df['longitude'] = scaler2.transform(df[['longitude']])
            df['speed'] = scaler3.transform(df[['speed']])
            df['ac'] = scaler4.transform(df[['ac']])
            df['cog'] = scaler5.transform(df[['cog']])
            df['rot'] = scaler6.transform(df[['rot']])
            # df['time'] = scaler7.transform(df[['time']])
            # df['dt'] = scaler8.transform(df[['dt']])
            # print('df:',df)
            if x_longest<df.shape[0]:
                x_longest=df.shape[0]
            size = (30,6)  # 定义0填充的长度
            train_zeros = np.zeros(size)#构造0矩阵
            for len_index in range(len(df)):
                train_zeros[len_index,0] = df.ix[len_index,'latitude']
                train_zeros[len_index,1] = df.ix[len_index,'longitude']
                train_zeros[len_index,2] = df.ix[len_index,'speed']
                train_zeros[len_index,3] = df.ix[len_index,'ac']
                train_zeros[len_index,4] = df.ix[len_index,'cog']
                train_zeros[len_index,5] = df.ix[len_index,'rot']
            x_train1.append(train_zeros)
x_train1=np.array(x_train1)
print('航道 1的 shape:',x_train1.shape)
print('x_longest:',x_longest)
#
path = args.inputPath+'data/航道 2/'
x_train2= []
x_longest = 0
# index 一艘船舶轨迹文件夹的序号,value 为船舶轨迹文件夹的名字
for index, value in (enumerate(os.listdir(path))):
    temp_list = []
    data2 = pd.read_csv(path + value)
    pd.set_option('mode.chained_assignment', None)
    # print(value)
    df = data2[['latitude','longitude','speed','ac','cog','rot']]
    df['latitude'] = scaler1.transform(df[['latitude']])
    df['longitude'] = scaler2.transform(df[['longitude']])
    df['speed'] = scaler3.transform(df[['speed']])
    df['ac'] = scaler4.transform(df[['ac']])
    df['cog'] = scaler5.transform(df[['cog']])
    df['rot'] = scaler6.transform(df[['rot']])
    if x_longest < df.shape[0]:
        x_longest = df.shape[0]
    size = (30, 6)  # 定义0填充的长度
    train_zeros = np.zeros(size)  # 构造0矩阵
    for len_index in range(len(df)):
        train_zeros[len_index, 0] = df.ix[len_index, 'latitude']
        train_zeros[len_index, 1] = df.ix[len_index, 'longitude']
        train_zeros[len_index, 2] = df.ix[len_index, 'speed']
        train_zeros[len_index, 3] = df.ix[len_index, 'ac']
        train_zeros[len_index, 4] = df.ix[len_index, 'cog']
        train_zeros[len_index, 5] = df.ix[len_index, 'rot']
        # train_zeros[len_index, 6] = df.ix[len_index, 'time']
        # train_zeros[len_index, 7] = df.ix[len_index, 'dt']
    x_train2.append(train_zeros)
x_train2 = np.array(x_train2)
print('航道 2的 shape:', x_train2.shape)
print('x_longest:', x_longest)
#
#
x_train1=np.reshape(x_train1,(-1,30,6))
print('reshape 航道 1:',x_train1.shape)
np.save(args.outputPath+'航道 1.npy',x_train1)
np.save(args.outputPath+'航道 1label.npy',np.array([1 for index in range(len(x_train1))]))


x_train2=np.reshape(x_train2,(-1,30,6))
print('reshape 航道 2:',x_train2.shape)
np.save(args.outputPath+'航道 2.npy',x_train2)
np.save(args.outputPath+'航道 2label.npy',np.array([2 for index in range(len(x_train2))]))

end_time=time.time()
print('time cost:',end_time-start_time)

