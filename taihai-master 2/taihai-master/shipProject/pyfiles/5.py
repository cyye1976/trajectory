import numpy as np
np.random.seed(1337)
from keras.models import Model
from keras.layers import  Input,Conv1D,MaxPooling1D,UpSampling1D
import matplotlib.pyplot as plt
from keras.optimizers import Adam
import logging
from sklearn.metrics import recall_score, f1_score,precision_score
from KuhnMunkres import err_rate,best_map
from sklearn import mixture
from sklearn.metrics import classification_report
import time
import os
import argparse
#logging.getLogger([name=None])指定name，返回一个名称为name的Logger实例
def show_train_history1(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Loaction_Auto_encoder Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(args.outputPath+'figure/loss1.png')

def show_train_history2(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Speed_Auto_encoder Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(args.outputPath+'figure/loss2.png')

def show_train_history3(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Cog_Auto_encoder Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(args.outputPath+'figure/loss3.png')

def generate_default_params():
    '''
    Generate default hyper parameters
    '''
    return {
        'learning_rate': 0.01,
        'epochs': 180,
        'batch_size':256
    }

parser = argparse.ArgumentParser(description="path")
parser.add_argument('-i', '--inputPath',default='./')
parser.add_argument('-o', '--outputPath',default='./')
args = parser.parse_args()
if not os.path.exists(args.outputPath):
    os.makedirs(args.outputPath)
if not os.path.exists(args.outputPath+'figure/'):
    os.makedirs(args.outputPath+'figure/')
x_train1=np.load(args.inputPath+'分类特征和标签/x_train_location.npy')
x_train1=np.array(x_train1).reshape(-1,60,1 )
print('x_train_location.shape:',x_train1.shape)
x_val1=np.load(args.inputPath+'分类特征和标签/x_val_location.npy')
x_val1=np.array(x_val1).reshape(-1,60,1 )
print('x_val_location.shape:',x_val1.shape)
#加载速度训练集和验证集
x_train2=np.load(args.inputPath+'分类特征和标签/x_train_speed.npy')
x_train2=np.array(x_train2).reshape(-1,60,1 )
print('x_train_speed.shape:',x_train2.shape)
x_val2=np.load(args.inputPath+'分类特征和标签/x_val_speed.npy')
x_val2=np.array(x_val2).reshape(-1,60,1 )
print('x_val_speed.shape:',x_val2.shape)
#加载航向训练集和验证集
x_train3=np.load(args.inputPath+'分类特征和标签/x_train_cog.npy')
x_train3=np.array(x_train3).reshape(-1,60,1 )
print('x_train_cog.shape:',x_train3.shape)
x_val3=np.load(args.inputPath+'分类特征和标签/x_val_cog.npy')
x_val3=np.array(x_val3).reshape(-1,60,1 )
print('x_val_cog.shape:',x_val3.shape)

#加载航向训练集和验证集
def training1(PARAMS):
    x = Input(shape=(60,1))
    # 一层卷积自编码器
    # 编码器
    conv1_1 = Conv1D(filters=4, kernel_size=4, strides=2, activation='relu', padding='same')(x)
    encoded = pool1 = MaxPooling1D(30, padding='same', name='shuchu')(conv1_1)
    # #解码器
    conv2_2 = Conv1D(filters=4, kernel_size=4, strides=1, activation='relu', padding='same')(encoded)
    up1 = UpSampling1D(60)(conv2_2)
    decoded = Conv1D(filters=1, kernel_size=4, activation='tanh', padding='same')(up1)

    autoencoder=Model(inputs=x,outputs=decoded)
    encoder=Model(inputs=x,outputs=autoencoder.get_layer('shuchu').output )
    adam=Adam(lr=PARAMS['learning_rate'])
    autoencoder.compile(optimizer=adam,loss='mse')
    #创建一个实例 history
    autoencoder.summary()
    #迭代训练
    train_history=autoencoder.fit(x_train1,x_train1,
                    validation_data=(x_val1,x_val1),
                    epochs=PARAMS['epochs'],
                    batch_size=PARAMS['batch_size'],
                    shuffle=False)
    loss = autoencoder.evaluate(x_val1, x_val1, verbose=0)
    print('x_train.shape:',x_train1.shape)
    print('x_val.shape:',x_val1.shape)
    x_test=np.concatenate((x_train1,x_val1))
    print('x_test.shape:',x_test.shape)
    encoded_data = encoder.predict(x_test)
    print('encoded_data.shape:',encoded_data.shape)
    return encoded_data,train_history
def training2(PARAMS):
    x = Input(shape=(60, 1))
    # 一层卷积自编码器
    # 编码器
    conv1_1 = Conv1D(filters=4, kernel_size=4, strides=2, activation='relu', padding='same')(x)
    encoded = pool1 = MaxPooling1D(30, padding='same', name='shuchu')(conv1_1)
    # #解码器
    conv2_2 = Conv1D(filters=4, kernel_size=4, strides=1, activation='relu', padding='same')(encoded)
    up1 = UpSampling1D(60)(conv2_2)
    decoded = Conv1D(filters=1, kernel_size=4, activation='tanh', padding='same')(up1)

    autoencoder = Model(inputs=x, outputs=decoded)
    encoder = Model(inputs=x, outputs=autoencoder.get_layer('shuchu').output)

    adam = Adam(lr=PARAMS['learning_rate'])
    autoencoder.compile(optimizer=adam, loss='mse')
    # 创建一个实例 history
    autoencoder.summary()
    # 迭代训练
    train_history=autoencoder.fit(x_train2, x_train2,
                    validation_data=(x_val2, x_val2),
                    epochs=PARAMS['epochs'],
                    batch_size=PARAMS['batch_size'],
                    shuffle=False)
                    # callbacks=[SendMetrics()])
    loss = autoencoder.evaluate(x_val2, x_val2, verbose=0)
    print('x_train.shape:', x_train2.shape)
    print('x_val.shape:', x_val2.shape)
    x_test = np.concatenate((x_train2, x_val2))
    print('x_test.shape:', x_test.shape)
    encoded_data = encoder.predict(x_test)
    print('encoded_data.shape:', encoded_data.shape)
    return encoded_data,train_history
def training3(PARAMS):
    x = Input(shape=(60, 1))
    # 一层卷积自编码器
    # 编码器
    conv1_1 = Conv1D(filters=4, kernel_size=4, strides=2, activation='relu', padding='same')(x)
    encoded = pool1 = MaxPooling1D(30, padding='same', name='shuchu')(conv1_1)
    # #解码器
    conv2_2 = Conv1D(filters=4, kernel_size=4, strides=1, activation='relu', padding='same')(encoded)
    up1 = UpSampling1D(60)(conv2_2)
    decoded = Conv1D(filters=1, kernel_size=4, activation='tanh', padding='same')(up1)

    autoencoder = Model(inputs=x, outputs=decoded)
    encoder = Model(inputs=x, outputs=autoencoder.get_layer('shuchu').output)

    adam = Adam(lr=PARAMS['learning_rate'])
    autoencoder.compile(optimizer=adam, loss='mse')
    # 迭代训练
    train_history=autoencoder.fit(x_train3, x_train3,
                    validation_data=(x_val3, x_val3),
                    epochs=PARAMS['epochs'],
                    batch_size=PARAMS['batch_size'],
                    shuffle=False)
                    # callbacks=[SendMetrics()])
    loss = autoencoder.evaluate(x_val3, x_val3, verbose=0)
    print('x_train.shape:', x_train3.shape)
    print('x_val.shape:', x_val3.shape)
    x_test = np.concatenate((x_train3, x_val3))
    print('x_test.shape:', x_test.shape)
    encoded_data = encoder.predict(x_test)
    print('encoded_data.shape:', encoded_data.shape)
    return encoded_data,train_history
# def GMMcluster(data,y_true):
#     print('---------------jiangsu_GMM----------------')
#     clf = mixture.GaussianMixture(n_components=2,covariance_type='full',random_state=2019)
#     clf.fit(data)
#     y_pred = clf.predict(data)
#     rate, y_pred = err_rate(y_true, y_pred)
#     print('聚类后的精度precision为:', precision_score(y_true, y_pred, average='macro') * 100, '%')
#     print('聚类后的准确率acc是：{} %'.format((1 - rate) * 100))
#     print('recall score weighted:', recall_score(y_true, y_pred, average='macro') * 100, '%')
#     print('f1 score weighted:', f1_score(y_true, y_pred, average='macro') * 100, '%')
#     target_names = ['class 0', 'class 1']
#     print(classification_report(y_true, y_pred, target_names=target_names))
if __name__ == '__main__':
    PARAMS = generate_default_params()
    X_location,train_history1=training1(PARAMS)
    X_speed,train_history2=training2(PARAMS)
    X_cog,train_history3=training3(PARAMS)
    # print(X_location,train_history1)
    show_train_history1(train_history1, 'loss', 'val_loss')
    show_train_history2(train_history2, 'loss', 'val_loss')
    show_train_history3(train_history3, 'loss', 'val_loss')

    # y_true = np.load('./encoded_data_label.npy')
    # # print('y_true.shape:', y_true.shape)
    # # data
    X_location = X_location.reshape(X_location.shape[0], -1)
    print('X_location.reshape.shape:', X_location.shape)
    X_speed = X_speed.reshape(X_speed.shape[0], -1)
    print('X_speed.reshape.shape:', X_speed.shape)
    X_cog = X_cog.reshape(X_cog.shape[0], -1)
    print('X_cog.reshape.shape:', X_cog.shape)
    X = np.concatenate((X_location, X_speed, X_cog), axis=1)
    np.save(args.outputPath+'encoded_data.npy',X)
    print('多特征融合自编码器提取轨迹的特征维度为:', X.shape)
    print("\n<outputFigPath>:"+args.outputPath+'figure/loss1.png')
    print("\n<outputFigPath>:"+args.outputPath+'figure/loss2.png')
    print("\n<outputFigPath>:"+args.outputPath+'figure/loss3.png')





