# # coding: utf-8
#
# # In[ ]:
# import os
#
# import tensorflow as tf
#
# import keras
# from keras.engine.saving import load_model
# from keras.models import Sequential
# from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
# from keras.layers import Dense, Activation, Dropout, Flatten
#
# from keras.preprocessing import image
# from keras.preprocessing.image import ImageDataGenerator
#
# import numpy as np
#
# #------------------------------
# # sess = tf.Session()
# # keras.backend.set_session(sess)
# #------------------------------
# #variables
# num_classes =4
# batch_size = 80
# epochs = 80
# #------------------------------
#
# import os, cv2, keras
# import numpy as np
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Flatten
# from keras.layers import Conv2D, MaxPooling2D
# from keras.engine.saving import load_model
# # manipulate with numpy,load with panda
# import numpy as np
# # import pandas as pd
#
# # data visualization
# import cv2
# import matplotlib
# import matplotlib.pyplot as plt
# # import seaborn as sns
#
# # get_ipython().run_line_magic('matplotlib', 'inline')
#
#
# # Data Import
# def read_dataset():
#     data_list = []
#     label_list = []
#     my_list = os.listdir(r'C:\DATASET\DATASETmain')
#     for pos,pa in enumerate(my_list):
#
#         print(pa,"==================")
#         for root, dirs, files in os.walk(r'C:\DATASET\DATASETmain\\' + pa):
#
#          for f in files:
#             file_path = os.path.join(r'C:\DATASET\DATASETmain\\'+pa, f)
#             img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
#             res = cv2.resize(img, (48, 48), interpolation=cv2.INTER_CUBIC)
#             data_list.append(res)
#             # label = dirPath.split('/')[-1]
#             label = pa
#             label_list.append(pos)
#             # label_list.remove("./training")
#     return (np.asarray(data_list, dtype=np.float32), np.asarray(label_list))
#
#     print("=====================",label_list)
#
#
# from sklearn.model_selection import train_test_split
# # load dataset
# x_dataset, y_dataset = read_dataset()
# X_train, X_test, y_train, y_test = train_test_split(x_dataset, y_dataset, test_size=0.2, random_state=0)
#
# y_train1=[]
# for i in y_train:
#     emotion = keras.utils.to_categorical(i, num_classes)
#     print(i,emotion)
#     y_train1.append(emotion)
#
# y_train=y_train1
# x_train = np.array(X_train, 'float32')
# y_train = np.array(y_train, 'float32')
# x_test = np.array(X_test, 'float32')
# y_test = np.array(y_test, 'float32')
#
# x_train /= 255  # normalize inputs between [0, 1]
# x_test /= 255
# print("x_train.shape",x_train.shape)
# x_train = x_train.reshape(x_train.shape[0], 48, 48, 1)
# x_train = x_train.astype('float32')
# x_test = x_test.reshape(x_test.shape[0], 48, 48, 1)
# x_test = x_test.astype('float32')
#
# print(x_train.shape[0], 'train samples')
# print(x_test.shape[0], 'test samples')
# # ------------------------------
# # construct CNN structure
#
# model = Sequential()
#
# # 1st convolution layer
# model.add(Conv2D(64, (5, 5), activation='relu', input_shape=(48, 48, 1)))
# model.add(MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))
#
# # 2nd convolution layer
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))
#
# # 3rd convolution layer
# model.add(Conv2D(128, (3, 3), activation='relu'))
# model.add(Conv2D(128, (3, 3), activation='relu'))
# model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))
#
# model.add(Flatten())
#
# # fully connected neural networks
# model.add(Dense(1024, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(1024, activation='relu'))
# model.add(Dropout(0.2))
#
# model.add(Dense(num_classes, activation='softmax'))
# # ------------------------------
# # batch process
#
# print(x_train.shape)
#
# gen = ImageDataGenerator()
# train_generator = gen.flow(x_train, y_train, batch_size=batch_size)
#
# # ------------------------------
#
# model.compile(loss='categorical_crossentropy'
#               , optimizer=keras.optimizers.Adam()
#               , metrics=['accuracy']
#               )
#
# # ------------------------------
#
# if not os.path.exists("model1MAIN.h5"):
#
#     model.fit_generator(train_generator, steps_per_epoch=batch_size, epochs=epochs)
#     model.save("model1MAIN.h5")  # train for randomly selected one
# else:
#     model = load_model("model1MAIN.h5")  # load weights
# from sklearn.metrics import confusion_matrix
# yp=model.predict_classes(x_test,verbose=0)
# cf=confusion_matrix(y_test,yp)
# print(cf)

import os
import cv2
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.engine.saving import load_model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Variables
num_classes = 5  # Number of classes
batch_size = 110
epochs = 110


# Data Import
def read_dataset():
    data_list = []
    label_list = []
    i=-1
    my_list = os.listdir(r'C:\DATASET\DATASETmain')  # Change path to your dataset folder
    for pos, pa in enumerate(my_list):
        i=i+1
        print(pa, "==================",i)
        for root, dirs, files in os.walk(r'C:\DATASET\DATASETmain\\' + pa):
            for f in files:
                file_path = os.path.join(r'C:\DATASET\DATASETmain\\' + pa, f)
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                res = cv2.resize(img, (48, 48), interpolation=cv2.INTER_CUBIC)
                data_list.append(res)
                label = pa  # Assuming label is the directory name
                label_list.append(i)
    return np.asarray(data_list, dtype=np.float32), np.asarray(label_list)


# Load dataset
x_dataset, y_dataset = read_dataset()
X_train, X_test, y_train, y_test = train_test_split(x_dataset, y_dataset, test_size=0.2, random_state=0)

# One-hot encode the labels
y_train1 = []
for i in y_train:
    emotion = keras.utils.to_categorical(i, num_classes)
    y_train1.append(emotion)
y_train = y_train1

x_train = np.array(X_train, 'float32')
y_train = np.array(y_train, 'float32')
x_test = np.array(X_test, 'float32')
y_test = np.array(y_test, 'float32')

# Normalize the images
x_train /= 255  # Normalize inputs between [0, 1]
x_test /= 255

# Reshape images for the CNN model
x_train = x_train.reshape(x_train.shape[0], 48, 48, 1)
x_train = x_train.astype('float32')
x_test = x_test.reshape(x_test.shape[0], 48, 48, 1)
x_test = x_test.astype('float32')

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Construct CNN model
model = Sequential()

# 1st convolution layer
model.add(Conv2D(64, (5, 5), activation='relu', input_shape=(48, 48, 1)))
model.add(MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))

# 2nd convolution layer
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

# 3rd convolution layer
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

model.add(Flatten())

# Fully connected neural networks
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(num_classes, activation='softmax'))
print(model.compile)

# Batch process
print(x_train.shape)

gen = ImageDataGenerator()
train_generator = gen.flow(x_train, y_train, batch_size=batch_size)

# Compile model
model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

# If the model does not exist, train it
if not os.path.exists("model1MAIN1.h5"):
    model.fit_generator(train_generator, steps_per_epoch=batch_size, epochs=epochs)
    model.save("model1MAIN1.h5")
else:
    model = load_model("model1MAIN1.h5")  # Load weights


# Implementing confidence threshold
def predict_with_threshold(model, x_data, threshold=0.7):
    probabilities = model.predict(x_data, verbose=0)
    predictions = []

    for prob in probabilities:
        if max(prob) < threshold:
            predictions.append('uncertain')  # Classify as uncertain
        else:
            predictions.append(np.argmax(prob))  # Get the class with highest probability

    return predictions

# # Predict using the model with the threshold
# yp = predict_with_threshold(model, x_test)
#
# # Confusion Matrix
# print(yp)
# cf = confusion_matrix(y_test, yp)
# print(cf)
#
# # Optional: Visualize the confusion matrix (if desired)
# plt.figure(figsize=(10, 7))
# plt.imshow(cf, interpolation='nearest', cmap=plt.cm.Blues)
# plt.title('Confusion Matrix')
# plt.colorbar()
# tick_marks = np.arange(num_classes)
# plt.xticks(tick_marks, range(num_classes))
# plt.yticks(tick_marks, range(num_classes))
# plt.xlabel('Predicted label')
# plt.ylabel('True label')
# plt.show()


