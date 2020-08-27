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
import cv2

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
            self.progressCnt += 1
            printProgress(self.progressCnt, self.length, ' Analysis Progress: ', 'Complete')

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

            if cx > 1.0: cx = 1.0
            if cx < 0.0: cx = 0.0
            if cy > 1.0: cy = 1.0
            if cy < 0.0: cy = 0.0
            if w > 1.0: w = 1.0
            if w < 0.0: w = 0.0
            if h > 1.0: h = 1.0
            if h < 0.0: h = 0.0
            return [round(cx, 6), round(cy, 6), round(w, 6), round(h, 6)]

    def isPerson(self, label):
        if ((label == "Cyclist") or (label == "people") or (label == "person") or (label == "rider") or (label == "pedestrian")):
            return False
        else:
            return False
    def yolo(self, data, save, copy):
        self.length = len(data)
        if copy:
            copyDir = os.path.join(self.config.copyDir, self.config.datasets)
            if not os.path.exists(copyDir):
                os.makedirs(copyDir)

        printProgress(self.progressCnt, self.length, ' YOLO Parsing Progress: ', 'Complete')
        fname = None
        for key in data:
            bExcept = False
            outputStr = ''
            fileName, fextension = os.path.splitext(key)
            fname = fileName.split('-')[0]
            path = data[key]["info"]["path"]
            type = data[key]["info"]["type"]
            ignore = None
            # ignore = data[key]["info"]["ignore"]
            imgWidth = data[key]["size"]["width"]
            imgHeight = data[key]["size"]["height"]

            if not copy:
                copyDir = path
            for idx in range(1, data[key]["objects"]["num_obj"] + 1):
                cls = data[key]["objects"][str(idx)]["name"]
                xmin = data[key]["objects"][str(idx)]["bbox"]["xmin"]
                ymin = data[key]["objects"][str(idx)]["bbox"]["ymin"]
                xmax = data[key]["objects"][str(idx)]["bbox"]["xmax"]
                ymax = data[key]["objects"][str(idx)]["bbox"]["ymax"]

                bboxCoordinate = self.coordinateConvert("xyMinMax", [imgWidth, imgHeight, xmin, ymin, xmax, ymax])
                if cls in self.labelDict:
                    outputStr += "{} {} {} {} {}\n".format(self.labelDict[cls], bboxCoordinate[0], bboxCoordinate[1], bboxCoordinate[2], bboxCoordinate[3]) # using join
                """    if (self.isPerson(cls) == True and (bboxCoordinate[2] >= 0.3 or bboxCoordinate[3] >= 0.5 or bboxCoordinate[2] > bboxCoordinate[3] * 1.5)):
                        bExcept = True
                        print("[Info]  File {} is excepted.".format(fileName))
                """

            imgFile = os.path.join(path, key) #imgFile = ''.join(path.split('.')[:-1]) + fextension
            # except
            if bExcept == True: continue
            if outputStr != '' or self.config.negative:
                if save:
                    copyDir2 = os.path.join(self.config.copyDir, 'label', self.config.datasets + '-label_txt')
                    if not os.path.exists(copyDir2):
                        os.makedirs(copyDir2)
                    textFile = os.path.join(copyDir2, fileName + '.txt') #textFile = ''.join(path.split('.')[:-1]) + '.txt'
                    # textFile = os.path.join(copyDir, fileName + '.txt') #textFile = ''.join(path.split('.')[:-1]) + '.txt'
                    with open(textFile, 'w+') as labelFile:
                        labelFile.write(outputStr)
                    if copy:
                        # Generative masked images. (detarc, Visdron)
                        if ignore is not None:
                            img = cv2.imread(imgFile)
                            for box in ignore:
                                cv2.rectangle(img, (int(box["xmin"]), int(box["ymin"])), (int(box["xmax"]), int(box["ymax"])), (102, 115, 123), -1)
                            cv2.imwrite(os.path.join(copyDir, key), img)
                        else:
                            shutil.copy(imgFile, os.path.join(copyDir, key))
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
            with open(os.path.join(os.getcwd(), self.config.datasets + "-train-" + self.manipastFile), "w+") as manipastFile:
            # with open(os.path.join(copyDir, "train-" + self.manipastFile), "a+") as manipastFile:
                manipastFile.write(self.trainList)
            with open(os.path.join(os.getcwd(), self.config.datasets + "-valid-" + self.manipastFile), "w+") as manipastFile:
            # with open(os.path.join(copyDir, "valid-" + self.manipastFile), "a+") as manipastFile:
                manipastFile.write(self.validList)
            print("[Info] Save the manipastFile. ({})".format(os.path.join(os.getcwd(), self.config.datasets + ' - ' + self.manipastFile)))
