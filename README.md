# Traffic Signs Recognition
There are several different types of traffic signs like speed limits, no entry, traffic signals, turn left or right, children crossing, no passing of heavy vehicles, etc. Traffic signs classification is the process of identifying which class a traffic sign belongs to.

## Traffic Signs

### Why know your traffic signs?
Traffic signs play a vital role in directing, informing and controlling
road users' behaviour in an effort to make the roads as safe as
possible for everyone. This makes a knowledge of traffic signs
essential. Not just for new drivers or riders needing to pass their
theory test, but for all road users, including experienced
professional drivers

## What are the types of traffic signs?
There are three basic types of traffic sign: signs that give orders, signs that warn and signs that give information

# Dataset of this project 
the dataset of thsi project is public and available at Kaggle :
https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign

The dataset contains more than 50,000 images of different traffic signs. It is further classified into 43 different classes. The dataset is quite varying, some of the classes have many images while some classes have few images. The size of the dataset is around 300 MB. The dataset has a train folder which contains images inside each class and a test folder which we will use for testing our model.

# Steps To build this project 
1.Explore the dataset        
2.Build a CNN model          
3.Train and validate the model        
4.Test the model with test dataset    

# Architecture of the built model 
2 Conv2D layer (filter=32, kernel_size=(5,5), activation=”relu”)   
MaxPool2D layer ( pool_size=(2,2))    
Dropout layer (rate=0.25)        
2 Conv2D layer (filter=64, kernel_size=(3,3), activation=”relu”)       
MaxPool2D layer ( pool_size=(2,2))       
Dropout layer (rate=0.25)     
Flatten layer to squeeze the layers into 1 dimension     
Dense Fully connected layer (256 nodes, activation=”relu”)     
Dropout layer (rate=0.5)     
Dense layer (43 nodes, activation=”softmax”)  

