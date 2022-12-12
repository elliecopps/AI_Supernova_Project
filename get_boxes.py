from email.mime import image
import os
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET

"""
This file takes my full dataset and splits it into a test and train dataset
Each dataset will be kept in a text file
Each line will contain [filename, xmin, ymin, width, height]
Coordinates correspond to supernova location in each image
"""


def getTrainSet():
    """Takes trainval.txt from my data to get the full list of images to train my model"""
    trainFile = open("../dataverse_files/VOC2007/ImageSets/Main/train.txt", "r")
    data = trainFile.read()
    return [x for x in data.split('\n') if x != '']


def buildDatasets(trainList):
    """
    Takes in the list of images to train on and produces two text files with the train 
    set and test set
    Since the dataset is augmented, a training image of 0001.png will also add 0001_01, 
    0001_02 etc. to the dataset. This prevents me from training on images that I will test on
    """
    annotationsPath = "../dataverse_files/VOC2007/Annotations/"
    trainFile = open("training_annotations.txt", "w")
    testFile = open("testing_annotations.txt", "w")
    width = 0
    height = 0
    maxW = 0
    maxH = 0
    tot = 0
    for imagename in os.listdir(annotationsPath):
        #imagename has the VERSION of the image also, use this in the text file so I know which image exactly the box corresponds to
        with open(annotationsPath+imagename, "r+") as i:
            xmlData = i.read()
        data = BeautifulSoup(xmlData, "xml")

        #generalName does NOT have the version
        generalName = remove_tag(str(data.find('filename')))
        if str(remove_tag(str(data.find('name')))) != 'nova':
            print('cont')
            continue
        xmin = remove_tag(data.find('xmin'))
        ymin = remove_tag(data.find('ymin'))
        xmax = remove_tag(data.find('xmax'))
        ymax = remove_tag(data.find('ymax'))
        width += int(remove_tag(data.find('width')))
        height += int(remove_tag(data.find('height')))
        maxW = max(maxW, int(remove_tag(data.find('width'))))
        maxH = max(maxH, int(remove_tag(data.find('height'))))
        tot += 1

        if generalName.split('.')[0] in trainList:
            trainFile.write(imagename.split('.')[0])
            trainFile.write(", ")
            trainFile.write(xmin)
            trainFile.write(", ")
            trainFile.write(ymin)
            trainFile.write(", ")
            trainFile.write(xmax)
            trainFile.write(", ")
            trainFile.write(ymax)
            trainFile.write('\n')
        else:
            testFile.write(imagename.split('.')[0])
            testFile.write(", ")
            testFile.write(xmin)
            testFile.write(", ")
            testFile.write(ymin)
            testFile.write(", ")
            testFile.write(xmax)
            testFile.write(", ")
            testFile.write(ymax)
            testFile.write('\n')
    trainFile.close()
    testFile.close()
    print("avg width: ", width/tot)
    print("avg height: ", height/tot)
    print("max width: ", maxW)
    print("max height: ", maxH)

def remove_tag(el):
    el = str(el)
    el = el.split(">")[1]
    el = el.split("<")[0]
    return el

if __name__ == "__main__":
    trainList = getTrainSet()
    buildDatasets(trainList)
    #avg width: 370
    #avg height: 390
    #max width: 925
    #max height: 580