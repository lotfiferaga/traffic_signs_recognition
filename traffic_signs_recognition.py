#imports 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import tensorflow as tf 
from PIL import Image 
import os 
from sklearn.model_selection import train_test_split 
from tensorflow.keras.utils import to_categorical 
from tensorflow.keras.models import Sequential
from keras.layers import Conv2D , MaxPool2D ,Dense , Flatten , Dropout
from tensorflow.python.keras.backend import dropout 

data = []
labels = []
classes = 43 
current_path = os.getcwd()

for x in range(classes) :
    path = os.path.join(current_path,'train',str(x))
    images = os.listdir(path)

    for i in images:
        try :
            image= Image.open(path + '\\' +i )
            image = image.resize((30,30))
            image = np.array(image)

            data.append(image)    
            labels.append(i)
        except:
            print("Error loading image ")

data = np.array(data)
labels = np.array(labels)

print(data.shape , labels.shape)

x_train,x_test , y_train , y_test = train_test_split(data , labels , test_size = 0.2 , random_state=42)

print(x_train.shape,x_test.shape , y_train.shape , y_test.shape )

y_train = to_categorical(y_train , 43)
y_test = to_categorical(y_test , 43)

#builing the CNN model 

model = Sequential()
model.add(Conv2D(filters=32,kernel_size=(5,5),activation='relu',input_shape= x_train.shape[1:]))
model.add(Conv2D(filters=32 ,kernel_size=(5,5), activation='relu'))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters = 64, kernel_size=(3,3),activation='relu'))
model.add(Conv2D(filters = 64, kernel_size=(3,3),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256 , activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(43, activation='softmax'))

# compiling the model 
model.compile(loss='categorical_crossentropy', optimizer='adam' ,metrics=['accuracy'])

#train and validate the model 

history = model.fit(x_train, y_train , batch_size = 64, epochs = 15 , validation_data=(x_test, y_test))

