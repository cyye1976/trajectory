import os
import pickle
import math
import argparse
from math import radians, cos, sin, asin, sqrt,fabs
#计算两点之间距离公式
EARTH_RADIUS=6371           # 地球平均半径，6371km
def hav(theta):
    s = sin(theta / 2)
    return s * s
def get_distance_hav(lat0, lng0, lat1, lng1):
    "用haversine公式计算球面两点间的距离。"
    # 经纬度转换成弧度
    lat0 = radians(lat0)
    lat1 = radians(lat1)
    lng0 = radians(lng0)
    lng1 = radians(lng1)

    dlng = fabs(lng0 - lng1)
    dlat = fabs(lat0 - lat1)
    h = hav(dlat) + cos(lat0) * cos(lat1) * hav(dlng)
    distance = 2 * EARTH_RADIUS * asin(sqrt(h))*1000
    return distance

parser = argparse.ArgumentParser(description="path")
parser.add_argument('-i', '--inputPath',default='./')
parser.add_argument('-o', '--outputPath',default='./')
args = parser.parse_args()
inputPath, inputPath2 = args.inputPath.split(',')
with open(inputPath+'data.pkl', 'rb') as f:
    simData = pickle.load(f)
print('子轨迹数量:',len(simData))
if not os.path.exists(args.outputPath):
    os.makedirs(args.outputPath)
simTrjComps=[]
#遍历数据集中的每条子轨迹
for index,simtrss in enumerate(simData) :
    trjsCom = []
    # print ('子轨迹长度:', len(simtrss))
    # print('子轨迹：', index, simtrss)
    # 遍历子轨迹中的每个点
    for i in range(0,len(simtrss)):
            rec = []
            # 子轨迹第一个点
            if i==0:
               #时间，纬度，经度，speed，航向，rot，ac
               # print('simtrss:',simtrss)
               rec=[simtrss[i][0],simtrss[i][1],simtrss[i][2],simtrss[i][3],simtrss[i][4],0,0]
               # print(simtrss[i][3])
               # print(simtrss[i][3]*0.5144444)
               print('第 0 个点特征:',rec)
            #第二个点
            elif i==1:
                # 欧式距离
                locC=get_distance_hav(simtrss[i-1][1],simtrss[i-1][2],simtrss[i][1],simtrss[i][2])
                # print('loc:',locC)
                #添加时间
                rec.append(simtrss[i][0])
                #添加纬度
                rec.append(simtrss[i][1])
                #添加经度
                rec.append(simtrss[i][2])
                #计算时间差
                time=simtrss[i][0] - simtrss[i - 1][0]
                #添加速度
                speed=locC/time
                rec.append(speed)
                #添加航向
                course = simtrss[i][4]
                rec.append(course)
                #添加 rot
                rec.append(math.atan((simtrss[i][2]-simtrss[i-1][2])/(simtrss[i][1]-simtrss[i-1][1])))
                #添加加速度
                rec.append((speed)/(time))
#
            else:
                #i=2,从第3个点开始
                #欧式距离
                locC = get_distance_hav(simtrss[i - 1][1], simtrss[i - 1][2], simtrss[i][1], simtrss[i][2])
                # print('loc:',locC )
                #求上一个点跟上上个点的欧式距离
                loc_previous=get_distance_hav(simtrss[i - 2][1], simtrss[i - 2][2], simtrss[i-1][1], simtrss[i-1][2])
                #添加time
                rec.append(simtrss[i][0])
                #添加纬度
                rec.append(simtrss[i][1])
                #添加经度
                rec.append(simtrss[i][2])
                # rec.append(locC)
                #计算时间间隔
                time=simtrss[i][0] - simtrss[i - 1][0]
                #添加speed
                speed=locC/time
                rec.append(speed)
                speed_previous=loc_previous/(simtrss[i-1][0] - simtrss[i - 2][0])
                #添加cog
                course= simtrss[i][4]
                rec.append(course)
                #添加rot
                rec.append(math.atan((simtrss[i][2]-simtrss[i-1][2])/(simtrss[i][1]-simtrss[i-1][1])))
                #添加加速度
                rec.append((speed-speed_previous)/(time))
                # print ('course:',course)
            # print('rec:', rec)
            trjsCom.append(rec)
    # print ('trjsCom:',trjsCom)
    print('子轨迹长度:',len(trjsCom))
    simTrjComps.append(trjsCom)
# print(len(simTrjComps))
pickle.dump(simTrjComps,open(args.outputPath+'simTrjComps.pkl','wb+'))
import  pickle
import pandas as pd
with open(args.outputPath+'/simTrjComps.pkl', 'rb') as f:
    w = pickle.load(f)
print('子轨迹数量:',len(w))

for index, trs in enumerate(w):
    # print(trs)
    c_list=['time','latitude','longitude','speed','cog','rot','ac']
    trs=pd.DataFrame(trs,columns=c_list)
    # trs.columns=['time','latitude','longitude','speed','cog','rot','ac']
    # print(index,trs)
    if not os.path.exists(args.outputPath+'data/all_data/'):
        os.makedirs(args.outputPath+'data/all_data/')
    trs.to_csv(args.outputPath+'data/all_data/'+str(index)+'.csv')

    if index<len(os.listdir(inputPath2+'特征工程/航道 1')):
        if not os.path.exists(args.outputPath+'data/航道 1/'):
            os.makedirs(args.outputPath+'data/航道 1/')
        trs.to_csv(args.outputPath+'data/航道 1/'+str(index)+'.csv')
    else:
        if not os.path.exists(args.outputPath+'data/航道 2/'):
            os.makedirs(args.outputPath+'data/航道 2/')
        trs.to_csv(args.outputPath + 'data/航道 2/'+str(index)+'.csv')
