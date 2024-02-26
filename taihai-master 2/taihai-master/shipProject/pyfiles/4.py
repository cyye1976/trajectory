# -*- coding: utf-8 -*-
import numpy as np
from sklearn.model_selection import train_test_split
import time
import argparse
import os
parser = argparse.ArgumentParser(description="path")
parser.add_argument('-i', '--inputPath',default='./')
parser.add_argument('-o', '--outputPath',default='./')
args = parser.parse_args()
if not os.path.exists(args.outputPath):
    os.makedirs(args.outputPath)
start_time=time.time()
#label 处理
label1=np.load(args.inputPath+'航道 1label.npy')
print('航道 1label.shape',label1.shape)
label2=np.load(args.inputPath+'航道 2label.npy')
print('航道 2label.shape',label2.shape)
labels= np.concatenate([label1,label2],axis=0)
np.save(args.outputPath+'jiangsu_yaodi_labels.npy',labels)
print('all_data_labels的 shape:',labels.shape)
#data 处理
a=np.load(args.inputPath+'航道 1.npy')
# a=np.array(a).reshape(-1,180,1 )
print('航道 1.shape:',a.shape)
b=np.load(args.inputPath+'航道 2.npy')
# b=np.array(b).reshape(-1,180,1)
data = np.concatenate([a,b],axis=0)
print('data.shape:',data.shape)
print('------------------------------------')

x_train,x_val,label_train,label_val= train_test_split(data,labels,test_size=0.3,shuffle=True)
print('x_train_shape:',x_train.shape)
print('x_val_shape:',x_val.shape)
y_labels = np.concatenate((label_train,label_val))
np.save(args.outputPath+'encoded_data_label.npy', y_labels)
# #训练集特征切分
x_train_location=x_train[:,:,0:2]
print('x_train_location:',x_train_location.shape)
x_train_speed=x_train[:,:,2:4]
print('x_train_speed:',x_train_speed.shape)
x_train_cog=x_train[:,:,4:6]
print('x_train_cog:',x_train_cog.shape)
if not os.path.exists(args.outputPath+'分类特征和标签/'):
    os.makedirs(args.outputPath+'分类特征和标签/')
np.save(args.outputPath+'分类特征和标签/x_train_location.npy',x_train_location)
np.save(args.outputPath+'分类特征和标签/x_train_speed.npy',x_train_speed)
np.save(args.outputPath+'分类特征和标签/x_train_cog.npy',x_train_cog)
np.save(args.outputPath+'分类特征和标签/label_train.npy',label_train)
#验证集特征切分
x_val_location=x_val[:,:,0:2]
print('x_val_location:',x_val_location.shape)
x_val_speed=x_val[:,:,2:4]
print('x_val_speed:',x_val_speed.shape)
x_val_cog=x_val[:,:,4:6]
print('x_val_cog:',x_val_cog.shape)
np.save(args.outputPath+'分类特征和标签/x_val_location.npy',x_val_location)
np.save(args.outputPath+'分类特征和标签/x_val_speed.npy',x_val_speed)
np.save(args.outputPath+'分类特征和标签/x_val_cog.npy',x_val_cog)
np.save(args.outputPath+'分类特征和标签/label_val.npy',label_val)

end_time=time.time()
print('time cost:',end_time-start_time)
