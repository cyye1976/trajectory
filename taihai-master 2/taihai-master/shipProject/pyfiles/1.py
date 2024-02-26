import os
import pandas as pd
import pickle
import argparse
def forfor(a):
    return [item for sublist in a for item in sublist]
parser = argparse.ArgumentParser(description="path")
parser.add_argument('-i', '--inputPath',default='./')
parser.add_argument('-o', '--outputPath',default='./')
args = parser.parse_args()
train = []
dir_list = [args.inputPath+'特征工程/航道 1', args.inputPath+'特征工程/航道 2']
for value1 in dir_list:
    print('---------', value1, '----------')
    # value2 是文件名字
    for value2 in os.listdir(value1):
        train_list = []
        # parse_dates = ['col_name']   # 指定某行读取为日期格式
        temp_data = pd.read_csv('{}/{}'.format(value1, value2), parse_dates=['时间'])
        temp_days = temp_data['时间'].dt.day
        temp_hour = temp_data['时间'].dt.hour
        temp_min = temp_data['时间'].dt.minute
        temp_sec = temp_data['时间'].dt.second
        temp_data = temp_data[['时间', '经度', '纬度', '速度', '对地航向']]
        day = temp_days[0]
        # print(value2)
        # print(temp_data )
        # 遍历每条轨迹每个点的day属性
        for index, temp_day in enumerate(temp_days):
            # 如果是同一天数据
            if temp_day == day:
                # print('index=',index,'temp_day=',temp_day)
                temp_time = temp_hour * 60 * 60 + temp_min * 60 + temp_sec
                temp = temp_data.ix[[index], ['时间', '纬度', '经度', '速度', '对地航向']]
                temp['时间'] = temp_time
                temp_list = temp.values.tolist()
                # print('temp_list',temp_list)
                temp_list = forfor(temp_list)
                train_list.append(temp_list)
            else:
                # print('index=',index,'temp_day=',temp_day)
                temp_time = 86400 + temp_hour * 60 * 60 + temp_min * 60 + temp_sec
                temp = temp_data.ix[[index], ['时间', '纬度', '经度', '速度', '对地航向']]
                temp['时间'] = temp_time
                temp_list = temp.values.tolist()
                temp_list = forfor(temp_list)
                train_list.append(temp_list)
        train.append(train_list)
#print('子轨迹数量:', len(train))
if not os.path.exists(args.outputPath):
    os.makedirs(args.outputPath)
pickle.dump(train,open(args.outputPath+'data.pkl','wb+'))
