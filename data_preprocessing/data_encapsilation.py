# ============== 数据封装 ======================
import os
import cv2
import numpy as np
import pandas as pd

# ================获取路径 ============
#datasets_path = "CASME2_preprocessed_Li/data/CASME2/Xiaobai/"
# data_path = os.path.join(datasets_path, "Cropped")
# label_path = os.path.join(datasets_path, "Tag")
datasets_path = "D:/NUS_datasets/CASME_2"
data_path = os.path.join(datasets_path, "CASME2_Cropped")
label_path = os.path.join(datasets_path, "Tag")
# =========== 读取label ==============
os.chdir(label_path)
orig_dir = os.getcwd()

Tag = pd.read_csv('Tag.csv', usecols=[3], dtype='int')
Tag = np.array(Tag).reshape(255, )
print(Tag.shape)

# ==============读取data=============
os.chdir(data_path)
orig_dir = os.getcwd()

i = 0  # 去除标签为6的类[other]类
j = 0  # 表明样本数量 k为帧数
imgData = np.zeros((156, 126, 320, 256),dtype=int)
subject_list = os.listdir(orig_dir)
for subject in subject_list:  # [sub01 ... sub26]
    subject_path_list = os.path.join(orig_dir, subject)
    file_list = os.listdir(subject_path_list)

    for file in file_list:  # [EP...]
        file_path_list = os.path.join(subject_path_list, file)

        # 去除others类
        if Tag[i] != 6:
            img_list = os.listdir(file_path_list)
            k = 0
            for img in img_list:

                img_path = os.path.join(file_path_list, img)
                img = cv2.imread(img_path, 0)
                img = cv2.resize(img, (256, 320), interpolation=cv2.INTER_CUBIC)
                img_array = np.array(img)
                # if k < 24:
                #     imgData[j, k, :, :] = img_array
                #     k = k + 1
                #     print(j)
                # else:
                #     break
                imgData[j, k, :, :] = img_array
                k = k + 1
                print(j)
            j = j + 1

        i = i + 1
print(np.shape(imgData))

delete_index = np.where(Tag == 6)
Tag = np.delete(Tag, delete_index)
print(Tag.shape)

#data_root_path = "/data/CASME2/CASME2_preprocessed_LiXiaobai/data_encapsilation/"
data_root_path = "D:/NUS_datasets/CASME_2/data_encapsilation/"
if not os.path.exists(data_root_path):
    os.mkdir(data_root_path)

# ================ 封装训练数据 shape = [156,24,320,256] =============
np.save(os.path.join(data_root_path, f"ImageData.npy"), imgData)
# ================ 封装标签    shape = [156,]           ==============
np.save(os.path.join(data_root_path, f"Label.npy"), Tag)
