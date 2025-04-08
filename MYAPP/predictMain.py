# coding: utf-8

# In[ ]:
import os

import tensorflow as tf

import keras
from keras.engine.saving import load_model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Activation, Dropout, Flatten

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

import numpy as np

#------------------------------
# sess = tf.Session()
# keras.backend.set_session(sess)
#------------------------------
#variables
num_classes =5
batch_size = 50
epochs = 30
#------------------------------

import os, cv2, keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.engine.saving import load_model
# manipulate with numpy,load with panda
import numpy as np
# import pandas as pd

# data visualization
import cv2
import matplotlib
import matplotlib.pyplot as plt
# import seaborn as sns

# get_ipython().run_line_magic('matplotlib', 'inline')



def read_dataset1(path):
    data_list = []
    label_list = []

    file_path = os.path.join(path)
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    res = cv2.resize(img, (48, 48), interpolation=cv2.INTER_CUBIC)
    data_list.append(res)
    # label = dirPath.split('/')[-1]

            # label_list.remove("./training")
    return (np.asarray(data_list, dtype=np.float32))




def predict(fn):
    dataset=read_dataset1(fn)
    (mnist_row, mnist_col, mnist_color) = 48, 48, 1

    dataset = dataset.reshape(dataset.shape[0], mnist_row, mnist_col, mnist_color)
    dataset=dataset/255
    mo = load_model(r"C:\Users\safat\PycharmProjects\AI_DIESESE_PREDICTION\MYAPP\model1MAIN1.h5")

    # predict probabilities for test set

    my_list = os.listdir(r'C:\DATASET\DATASETmain')
    yhat_classes = mo.predict_classes(dataset, verbose=0)[0]


    print("Main class",my_list[yhat_classes])

    if my_list[yhat_classes]=="zzzz":
        return(["invalid","invalid"])
    elif my_list[yhat_classes]=="Breast Cancer":
        # dataset = read_dataset1(fn)
        # (mnist_row, mnist_col, mnist_color) = 48, 48, 1

        # dataset = dataset.reshape(dataset.shape[0], mnist_row, mnist_col, mnist_color)
        # dataset = dataset / 255
        mod1 = load_model(r"C:\Users\safat\PycharmProjects\AI_DIESESE_PREDICTION\MYAPP\model1.h5")

        # predict probabilities for test set

        my_listsub = os.listdir(r'C:\Users\safat\PycharmProjects\AI_DIESESE_PREDICTION\dataset\Breast Cancer')
        breastCls = mod1.predict_classes(dataset, verbose=0)[0]
        return my_list[yhat_classes],my_listsub[breastCls]
    elif my_list[yhat_classes]=="Kidney Cancer":
        mod2 = load_model(r"C:\Users\safat\PycharmProjects\AI_DIESESE_PREDICTION\MYAPP\model1Kidney.h5")
        my_listsub = os.listdir(r'C:\Users\safat\PycharmProjects\AI_DIESESE_PREDICTION\dataset\Kidney Cancer')
        kidneyCls = mod2.predict_classes(dataset, verbose=0)[0]
        return my_list[yhat_classes],my_listsub[kidneyCls]
    elif my_list[yhat_classes]=="Colon Cancer":
        mod3 = load_model(r"C:\Users\safat\PycharmProjects\AI_DIESESE_PREDICTION\MYAPP\model1colon.h5")
        my_listsub = os.listdir(r'C:\Users\safat\PycharmProjects\AI_DIESESE_PREDICTION\dataset\Colon Cancer')
        colonCls = mod3.predict_classes(dataset, verbose=0)[0]
        return my_list[yhat_classes],my_listsub[colonCls]
    elif my_list[yhat_classes]=="Lung Cancer":
        mod4 = load_model(r"C:\Users\safat\PycharmProjects\AI_DIESESE_PREDICTION\MYAPP\model1lung.h5")
        my_listsub = os.listdir(r'C:\Users\safat\PycharmProjects\AI_DIESESE_PREDICTION\dataset\lung cancer')
        lungCls = mod4.predict_classes(dataset, verbose=0)[0]
        return my_list[yhat_classes],my_listsub[lungCls]




#
#     print(yhat_classes)

# print(predict(r"C:\Users\safat\PycharmProjects\AI_DIESESE_PREDICTION\dataset\Breast Cancer\breast_benign\breast_benign_0001.jpg"))
# print(predict(r"C:\Users\safat\PycharmProjects\AI_DIESESE_PREDICTION\dataset\Breast Cancer\breast_malignant\breast_malignant_0010.jpg"))
# print(predict(r"C:\Users\safat\PycharmProjects\AI_DIESESE_PREDICTION\dataset\Colon Cancer\colon_bnt\colon_bnt_0007.jpg"))
# print(predict(r"C:\Users\safat\PycharmProjects\AI_DIESESE_PREDICTION\dataset\Colon Cancer\colon_aca\colon_aca_0060.jpg"))
# print(predict(r"C:\Users\safat\PycharmProjects\AI_DIESESE_PREDICTION\dataset\Kidney Cancer\kidney_tumor\kidney_tumor_0015.jpg"))
# print(predict(r"C:\Users\safat\PycharmProjects\AI_DIESESE_PREDICTION\dataset\Kidney Cancer\kidney_normal\kidney_normal_0002.jpg"))
# print(predict(r"C:\Users\safat\PycharmProjects\AI_DIESESE_PREDICTION\dataset\lung cancer\lung_aca\lung_aca_0010.jpg"))
# print(predict(r"C:\Users\safat\PycharmProjects\AI_DIESESE_PREDICTION\dataset\lung cancer\lung_bnt\lung_bnt_0010.jpg"))
# print(predict(r"C:\Users\safat\PycharmProjects\AI_DIESESE_PREDICTION\dataset\lung cancer\lung_scc\lung_scc_0009.jpg"))