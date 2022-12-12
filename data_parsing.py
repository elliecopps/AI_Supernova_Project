
import tensorflow as tf
import pandas as pd
import numpy as np
from numpy import save, load
from keras.layers import Dense, Input, Conv2D, MaxPool2D, LeakyReLU, Flatten, Dropout
import cv2
from keras.models import Model, load_model
from keras import backend as K
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

def getDatasets(fileName, xName, yName):
    #open dataset to read
    #mistake!!! - resizing the images without resizing the boxes I think led to me messing a lot of things up?
    imagePath = "../dataverse_files/VOC2007/images/"
    file = open(fileName, 'r')
    data = []
    box_data = []
    lines = file.readlines()
    for i, line in enumerate(lines):
        if i % 100 == 0:
            print("i: ", i)
        try:
            line = line.strip().split(", ")
            #reading image in color so I can use a pretrained keras model
            img = cv2.imread(imagePath + line[0]+'.jpg', cv2.IMREAD_COLOR)
            im = cv2.resize(img, (216, 216)) #resize image to be compatible with the amount of space on my computer
            data.append(im)
            box_data.append([int(line[1]), int(line[2]), int(line[3]), int(line[4])])
        except:
            print("image not found")
    #normalize image color data to go from 0 to 1
    data = np.array(data)/255
    #normalize bounding box location
    box_data = np.array(box_data) / 216
    save(xName, data)
    save(yName, box_data)
    return 

def buildModel():
    model_input = Input(shape=(216,216,3))
    x = block1(16, model_input)
    x = block1(32, x)
    x = block1(64, x)
    x = Flatten()(x)
    x = block2(512, x)
    x = block2(256, x)
    x = block2(128, x)
    x = block2(64, x)
    model_outputs = Dense(4)(x) #fully-connected layer connects everything to the output prediction

    model = Model(inputs=[model_input], outputs=[model_outputs])
    
    model.compile( tf.keras.optimizers.Adam(0.0001), #learning rate is 0.001 to start, can decrease
                 loss=custom_loss, #wrote custom loss function to include the IOU
                 metrics=[IOU])
    
    return model


def block1(filters,X):
    x = Conv2D(filters, kernel_size=(3,3), strides=1 )(X) #convolutional layer
    x = LeakyReLU(0.2)(x) #correction layer replaces negative numbers with 0.2 (not zero! no dead neurons)
    x = Conv2D(filters, (3,3), strides=1)(x)
    x = LeakyReLU(0.2)(x)
    x = MaxPool2D((2,2))(x) #pooling layer reduces image size and preserves important characteristics
    return x

def block2(units,X):  
    x = Dense(units)(X)
    x = Dropout(0.1)(x)
    x = LeakyReLU(0.2)(x)    
    return x


def train_model(model):
    #load datasets that I manipulated earlier
    x_train = load('xTrain.npy') #image data
    y_train = load('yTrain.npy') #tells where the supernova boxes are
    x_val = load('xTest.npy') #validation image data
    y_val = load('yTest.npy') #validation supernova boxes

    #Using a checkpoint allows me to stop training and come back to the same point
    #Save only checkpoints that improve upon my previous models
    filepath = 'best-model-v2.h5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    
    #If I get stuck, reducing the learning rate can help to keep improving the model
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.00001)

    history = model.fit(x_train, y_train, batch_size=4, epochs=40, validation_data=(x_val, y_val), verbose=2, callbacks=[checkpoint, reduce_lr])
    predictions = model.predict(x_val)
    print("predictions: ", predictions)
    #save predictions so I can plot them later
    save("predictionsv2.npy", predictions)
    #save history so I can see how training progressed
    history_df = pd.DataFrame(history.history)
    hist_file = 'historyv2.csv'
    with open(hist_file, 'w') as f:
        history_df.to_csv(f)
    return

def IOU(true, pred):
    """Calculates the intersection over union for my custom loss metric
        IOU is a measure of how much the bounding boxes that I predict overlap
        with the actual bounding box of each supernova"""
    x1 = K.maximum(true[:,0], pred[:,0]) #farther right of boxes left sides
    y1 = K.maximum(true[:,1], pred[:,1]) #highest coordinate of top coordinates
    x2 = K.minimum(true[:,2], pred[:,2]) #farther left of boxes right sides
    y2 = K.minimum(true[:,3], pred[:,3]) #lower of box tops
    intersection = K.maximum(0.0, x2-x1)*K.maximum(0.0,y2-y1) #where the two boxes intersect
    box1area = (true[:,2]-true[:,0])*(true[:,3]-true[:,1])
    box2area = (pred[:,2]-pred[:,0]) * (pred[:,3]-pred[:,1])
    union = box1area + box2area - intersection
    iou = intersection / union
    return iou

def custom_loss(y_true, y_pred):
    '''mean_square is the mean squared error between the predicted and true box, 
        and iou is the intersection over union -- mean squared typically used for regression'''
    mean_square = tf.losses.mean_squared_error(y_true, y_pred)
    iou = IOU(y_true , y_pred)
    return mean_square + (1-iou)

if __name__ == "__main__":
    #getDatasets("training_annotations.txt", 'xTrain.npy', 'yTrain.npy')
    #getDatasets("testing_annotations.txt", 'xTest.npy', 'yTest.npy')

    model = buildModel()
    train_model(model)