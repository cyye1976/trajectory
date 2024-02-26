import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1337)
import os
import argparse

def diff_gj(data):
    # print('未分割数据:',未分割数据)
    gj_list = []
    temp_list = []
    # print('len(未分割数据)',len(未分割数据))
    for index in range(len(data) - 1):
        # print('index:',index)
        # print('未分割数据:',未分割数据)
        if np.abs(data[index] - data[index + 1]) == 1:
            temp_list.append(data[index])
        else:
            temp_list.append(data[index])
            temp_list.append(data[index] + 1)
            gj_list.append(temp_list)
            temp_list = []
    return gj_list

parser = argparse.ArgumentParser(description="path")
parser.add_argument('-i', '--inputPath',default='./')
parser.add_argument('-o', '--outputPath',default='./')
args = parser.parse_args()
for value in (os.listdir(args.inputPath)):
     print(value)
     zero_b = []
     zero_l = []
     data=pd.read_excel(args.inputPath+value)
     lon = list(data['经度'])
     lat = list(data['纬度'])
 #diff 的作用就是前一个经度减去下一个经度的值,如果有n个点，那diff出来的就是n-1个值,正数表示的就是递减，负数表示的就是递增
     lat_diff = list(np.diff(lat))
     # lat_diff = list(np.diff(lon))
     zero_b = [lat_diff.index(value) for value in lat_diff if value >0]#判断经度差列表的值是否大于 0,如果大于 0 则返回这个值的 index 值，存在zero_b列表中
     #print('zero_b:',zero_b )
     zero_l = [lat_diff.index(value) for value in lat_diff if value < 0]#判断经度差列表的值是否小于 0,如果小于 0 则返回这个值的 index 值，存在zero_l列表中
     #print('zero_l:',zero_l)
     zero_b = diff_gj(zero_b)
     zero_l = diff_gj(zero_l)
     # zero_b.extend(zero_l)
     if not os.path.exists(args.outputPath+'特征工程/航道 1/'):
         os.makedirs(args.outputPath+'特征工程/航道 1/')
     for value in zero_b:
         temp_data = data.loc[value[0]:value[-1],:]
         # print(temp_data)
         
         temp_data.to_csv(args.outputPath+'特征工程/航道 1/{}_{}.csv'.format(value[0],value[-1]),index=False)
     print('zero_b:',zero_b)
     if not os.path.exists(args.outputPath+'特征工程/航道 2/'):
         os.makedirs(args.outputPath+'特征工程/航道 2/')
     for value in zero_l:
         temp_data = data.loc[value[0]:value[-1],:]
         temp_data.to_csv(args.outputPath+'特征工程/航道 2/{}_{}.csv'.format(value[0],value[-1]),index=False)
     #print('zero_l:',zero_l)
 #轨迹可视化
path1 = args.outputPath+'特征工程/航道 1/'
for index,value in enumerate(os.listdir(path1)):
     # print(index)
     # print(value)
     data1=pd.read_csv(path1+value)
     long1 = data1['经度']
     lat1 = data1['纬度']
     plt.plot(long1, lat1)
     plt.title('The ship trajectory1')
     plt.xlabel('longitude')
     plt.ylabel('latitude')
     # plt.show()
path1 = args.outputPath+'特征工程/航道 2/'
for index,value in enumerate(os.listdir(path1)):
     # print(index)
     # print(value)
     data1=pd.read_csv(path1+value)
     long1 = data1['经度']
     lat1 = data1['纬度']
     plt.plot(long1, lat1)
     plt.title('The ship trajectory')
     plt.xlabel('longitude')
     plt.ylabel('latitude')
     # plt.show()
if not os.path.exists(args.outputPath+"fig/"):
    os.makedirs(args.outputPath+"fig/")
plt.savefig(args.outputPath+"fig/"+'subtracks.png')
print("\n<outputFigPath>:"+args.outputPath+"fig/"+'subtracks.png')
