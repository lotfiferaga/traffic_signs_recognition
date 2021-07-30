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
