'''
* convert2YOLO
* Version: 1.0
* Author: gbsim (asdf2kr@naver.com)
* Since 2020-06-30
'''

import sys
import os
import shutil
from util import printProgress

class Converter:
    def __init__(self, config):
        """
        self.imageDir = imageDir
        '''
            self.imageDir (str, imageDirectory)
            /home/coco/data/images
            -images
                -train2014
                    *.jpg
                -valid2014
                    *.jpg
        '''
        self.imageData = imageData
        self.imageInfo = imageInfo
        self.type = dataType # Train or valid
        self.currentDir = os.getcwd()

        self.copy = False
        """
        # self.textFile = "text"
        self.config = config
        self.manipastFile = config.manipastFile
        self.labelDict = config.labelDict

        self.progressCnt = 0
        self.trainList = ""
        self.validList = ""

    def analysis(self, data):
        self.length = len(data)
        printProgress(self.progressCnt, self.length, ' Analysis Progress: ', 'Complete')
        for key in data:
            temp = [False for i in range(5)]
            for idx in range(1, data[key]["objects"]["num_obj"] + 1):
                cls = data[key]["objects"][str(idx)]["name"]
                if cls in self.labelDict:
                    self.config.classNum[self.labelDict[cls]] += 1
                    temp[self.labelDict[cls]] = True
            for i, t in enumerate(temp):
                if t:
                    self.config.imgNum[i] += 1
            print('[Info] {}/{} image'.format(self.progressCnt, self.length))
            print(self.config.imgNum)
            print(self.config.classNum)
            self.progressCnt += 1
            printProgress(self.progressCnt, self.length, ' Analysis Progress: ', 'Complete')
        print("[Info] total Image {}".format(self.length))
        print(self.config.imgNum)
        print(self.config.classNum)

    def coordinateConvert(self, type, info):
        if type == "xyMinMax":
            width, height = info[0], info[1]
            xmin, ymin, xmax, ymax = info[2], info[3], info[4], info[5]

            dw = 1. / width
            dh = 1. / height

            cx = ((xmin + xmax) / 2.0) * dw
            cy = ((ymin + ymax) / 2.0) * dh

            w = (xmax - xmin) * dw
            h = (ymax - ymin) * dh

            return [round(cx, 6), round(cy, 6), round(w, 6), round(h, 6)]

    def yolo(self, data, save, copy):
        self.length = len(data)
        copyDir = os.path.join(self.config.copyDir, self.config.datasets)
        if not os.path.exists(copyDir):
            os.makedirs(copyDir)

        printProgress(self.progressCnt, self.length, ' YOLO Parsing Progress: ', 'Complete')
        for key in data:
            outputStr = ''
            fileName, fextension = os.path.splitext(key)

            path = data[key]["info"]["path"]
            type= data[key]["info"]["type"]
            imgWidth = data[key]["size"]["width"]
            imgHeight = data[key]["size"]["height"]

            for idx in range(1, data[key]["objects"]["num_obj"] + 1):
                cls = data[key]["objects"][str(idx)]["name"]
                xmin = data[key]["objects"][str(idx)]["bbox"]["xmin"]
                ymin = data[key]["objects"][str(idx)]["bbox"]["ymin"]
                xmax = data[key]["objects"][str(idx)]["bbox"]["xmax"]
                ymax = data[key]["objects"][str(idx)]["bbox"]["ymax"]

                bboxCoordinate = self.coordinateConvert("xyMinMax", [imgWidth, imgHeight, xmin, ymin, xmax, ymax])
                if cls in self.labelDict:
                    outputStr += "{} {} {} {} {}\n".format(self.labelDict[cls], bboxCoordinate[0], bboxCoordinate[1], bboxCoordinate[2], bboxCoordinate[3]) # using join
            imgFile = os.path.join(path, key) #imgFile = ''.join(path.split('.')[:-1]) + fextension
            if outputStr != '' or self.config.negative:
                if save:
                    if copy:
                        textFile = os.path.join(copyDir, fileName + '.txt') #textFile = ''.join(path.split('.')[:-1]) + '.txt'
                        with open(textFile, 'a+') as labelFile:
                            labelFile.write(outputStr)
                        shutil.copy(imgFile, os.path.join(copyDir, key))
                    else:
                        textFile = os.path.join(path, fileName + '.txt') #textFile = ''.join(path.split('.')[:-1]) + '.txt'
                        with open(textFile, 'a+') as labelFile:
                            labelFile.write(outputStr)
                else:
                    textFile = os.path.join(path, fileName + '.txt')
                    print("[Info]  textFile {} \noutputStr {}".format(textFile, outputStr))
                if type == 'train':
                    self.trainList += "{}\n".format(os.path.join(copyDir, key))
                else:
                    self.validList += "{}\n".format(os.path.join(copyDir, key))
            self.progressCnt += 1
            printProgress(self.progressCnt, self.length, ' YOLO Parsing Progress: ', 'Complete')

        if save:
            with open(os.path.join(os.getcwd(), self.config.datasets + "-train-" + self.manipastFile), "a+") as manipastFile:
                manipastFile.write(self.trainList)
            with open(os.path.join(os.getcwd(), self.config.datasets + "-valid-" + self.manipastFile), "a+") as manipastFile:
                manipastFile.write(self.validList)
            print("[Info] Save the manipastFile. ({})".format(os.path.join(os.getcwd(), self.config.datasets + self.manipastFile)))
