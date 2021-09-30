import keras
import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import LeaveOneOut
from model import __create_Conv3D_net
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, CSVLogger
import matplotlib.pyplot as plt
import pandas as pd
import configparser
import os

# ========= 读入配置 =============
config = configparser.RawConfigParser()
config.read('./config/configuration.txt')

# ========= 模型名称 =============
model_name = config.get('experiment name', 'name')

# ========== 数据地址 ===============
Data_path = config.get('data paths', 'Data_path')
DataImg_Path = os.path.join(Data_path, "ImageData.npy")
Label_Path = os.path.join(Data_path, "Label.npy")

# ========= 训练参数配置 =============
nbEpoch = int(config.get('training settings', 'nbEpoch'))
batch_size = int(config.get('training settings', 'batch_size'))
lr = float(config.get('training settings', 'lr'))

# ========= 模型保存路径 =============
result_path = config.get('data paths', 'result_path')
model_save_path = config.get('data paths', 'model_save_path')
model_save_weights_path = os.path.join(model_save_path, model_name + '.h5')


def run():
    configTF = tf.compat.v1.ConfigProto()
    configTF.gpu_options.allow_growth = True
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=configTF))

    K.set_image_data_format('channels_last')

    # 数据获取
    # imgData [156,320,256,24] train_label [156,]
    imgData = np.load(DataImg_Path)
    label = np.load(Label_Path)
    imgData = np.transpose(imgData, [0, 2, 3, 1])
    train_label = to_categorical(label)

    # 数据存储地址
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    all_result_file = open(os.path.join(result_path, 'all_result.txt'), "w")
    all_result_file.close()

    # 模型创建
    model_Go = model = __create_Conv3D_net(simple=156, frames=24, width=256, height=320, nb_classes=6)
    adam = keras.optimizers.Adam(
        lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    model.compile(optimizer=adam, loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    # loo = LeaveOneOut()
    # i = 0
    # for train, test in loo.split(imgData, train_label):

    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)
    model_save_weights_path = os.path.join(model_save_path, model_name + '.h5')
    fig_save_path = os.path.join(model_save_path,model_name + '.jpg')

    early_stopping = EarlyStopping(
        monitor='val_loss', patience=50, verbose=1)
    save_model = ModelCheckpoint(
        filepath=model_save_weights_path,
        monitor='val_accuracy',
        save_best_only=True)
    csv_logger = CSVLogger(model_save_path + model_name + '.csv')

    model.fit(imgData, train_label,
              epochs=nbEpoch,
              validation_split=0.2,
              batch_size=batch_size,
              callbacks=[early_stopping, save_model, csv_logger],
              verbose=1,
              shuffle=True)
    records = pd.read_csv(model_save_path + model_name + '.csv')

    plt.figure()
    plt.subplot(211)
    plt.plot(records['val_loss'], label="validation")
    plt.plot(records['loss'], label="training")
    plt.yticks([0.00, 0.50, 1.00, 1.50])
    plt.title('Loss value', fontsize=12)

    ax = plt.gca()
    ax.set_xticklabels([])

    plt.subplot(212)
    plt.plot(records['val_accuracy'], label="validation")
    plt.plot(records['accuracy'], label="training")
    plt.yticks([0.5, 0.6, 0.7, 0.8])
    plt.title('Accuracy', fontsize=12)
    ax.legend()
    plt.savefig(fig_save_path)




run()
