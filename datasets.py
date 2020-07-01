'''
* convert2YOLO
* Version: 1.0
* Author: gbsim (asdf2kr@naver.com)
* Since 2020-06-30
'''
import os
import sys
import json
import random
from util import printProgress
import xml.etree.ElementTree as ET
# Common Data format
"""
data
{
    "imageName" :
                {
                    "info" :
                                {
                                    "path": <string> # image directory
                                    "type": <string> # Train or Valid
                                }
                    "size" :
                                {
                                    "width" : <string>
                                    "height" : <string>
                                }

                    "objects" :
                                {
                                    "num_obj" : <int>
                                    "<index>" :
                                                {
                                                    "name" : <string>
                                                    "bbox" :
                                                                {
                                                                    "xmin" : <float>
                                                                    "ymin" : <float>
                                                                    "xmax" : <float>
                                                                    "ymax" : <float>
                                                                }
                                                }
                                    ...


                                }
                }
"""

class coco:
    '''
    # COCO Datasets format.
        "annotations":[
        {
            "id": 123465,
            "category_id": 2,
            "iscrowd": 0,
            "segmentation": [[164.81, 417.51]]
            "image_id": 242287,
            "area": 42061.,
            "bbox": [19.2, 383.3, 314., 244.,]

        }
        ]
    '''
    # https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch
    def parse(jsonPath):
        data = {}
        try:
            json_data = json.load(open(jsonPath))
            images_info = json_data["images"]
            cls_info = json_data["categories"]
            progressCnt = 0
            printProgress(progressCnt, len(json_data["annotations"]), ' YOLO Parsing Progress: ', 'Complete')

            for anno in json_data["annotations"]:
                imageId = anno["image_id"]
                clsId = anno["category_id"]

                for info in images_info:
                    if info["id"] == imageId:
                        fileName, imgWidth, imgHeight = info["file_name"], info["width"], info["height"]

                for category in cls_info:
                    if category["id"] == clsId:
                        cls = category["name"]

                size = {
                    "width": imgWidth,
                    "height": imgHeight
                }

                bbox = {
                    "xmin": anno["bbox"][0],
                    "ymin": anno["bbox"][1],
                    "xmax": anno["bbox"][2] + anno["bbox"][0],
                    "ymax": anno["bbox"][3] + anno["bbox"][1]
                }

                info = {
                    "name": cls,
                    "bbox": bbox
                }

                if filenmae in data:
                    obj_idx = data[filename]["objects"]["num_obj"] + 1
                    data[filename]["objects"][str(obj_idx)] = info
                    data[filename]["objects"]["num_obj"] += 1
                else:
                    obj = {
                        "num_obj": 1,
                        "1": info
                    }
                    data[filename] = {
                        "size": size,
                        "objects": obj
                    }

                progressCnt += 1
                printProgress(progressCnt, len(json_data["annotations"]), ' YOLO Parsing Progress: ', 'Complete')

            return data

        except Exception as e:
            msg = "ERROR : {}, \n".format(e)
            return None

class bdd100k:
    def __init__(self, config):

        '''
            Num of images : 100,000
            Num of vehicles : 1,095,289
            Resolution: 1280 x 720
        '''
        self.currentDir = config.datasetsDir
        self.trainImgDir = os.path.join(config.datasetsDir, 'bdd100k', 'images', '100k', 'train')
        self.valImgDir = os.path.join(config.datasetsDir, 'bdd100k', 'images', '100k', 'val')
        self.labelDir = os.path.join(config.datasetsDir, 'bdd100k', 'labels')
        self.progressCnt = 0

        print("[Info] Load bdd100k datasets.")

    def parse(self):
        data = {}
        labelFiles = os.listdir(self.labelDir)
        for labelFile in labelFiles:
            if labelFile.find('train') != -1:
                imgDir = self.trainImgDir
                type = 'train'
            else:
                imgDir = self.valImgDir
                type = 'valid'

            with open(os.path.join(self.labelDir, labelFile)) as jsonFile:
                jsonData = json.load(jsonFile)
                self.progressCnt = 0
                self.length = len(jsonData)
                printProgress(self.progressCnt, self.length, (' Bdd100k ' + type + ' Parsing Progress: '), 'Complete')
                for image in jsonData:
                    filename = image["name"]
                    size = {
                        "width": float(1280),
                        "height": float(720)
                    }
                    '''
                    from PIL import Image
                    img = Image.open(os.path.join(imgDir, filename))
                    size = {
                        "width": float(img.size[0]),
                        "height": float(img.size[1])
                    }
                    '''
                    info = {
                        "path": imgDir,
                        "type": type
                    }
                    for label in image['labels']:
                        if not 'box2d' in label:
                            continue
                        bbox = {
                        "xmin": float(label['box2d']['x1']),
                        "ymin": float(label['box2d']['y1']),
                        "xmax": float(label['box2d']['x2']),
                        "ymax": float(label['box2d']['y2'])
                        }
                        bboxInfo = {
                            "name": label['category'],
                            "bbox": bbox
                        }

                        if filename in data:
                            obj_idx = data[filename]["objects"]["num_obj"] + 1
                            data[filename]["objects"][str(obj_idx)] = bboxInfo
                            data[filename]["objects"]["num_obj"] += 1
                        else:
                            obj = {
                                "num_obj": 1,
                                "1": bboxInfo
                            }
                            data[filename] = {
                                "info": info,
                                "size": size,
                                "objects": obj
                            }
                        bbox = {}
                        bboxInfo = {}

                    self.progressCnt += 1
                    printProgress(self.progressCnt, self.length, (' Bdd100k ' + type + ' Parsing Progress: '), 'Complete')
        return data
class mot:
    def __init__(self, labelDir):
        self.true = True
class detrac:
    def __init__(self, labelDir):
        self.true = True
class crowdhuman:
    def __init__(self, config):
        self.config = config
        self.fextension = '.jpg'
        self.labelDir = os.getcwd()
        print("[Info] Load crowdhuman dataset.")
        self.trainImgDir = os.path.join(os.getcwd(), 'Images__train')
        self.valImgDir = os.path.join(os.getcwd(), 'Images__val')
        if not os.path.exists(self.imgDir):
            os.makedirs(self.imgDir)

    def parse(self):
        files = os.listdir(self.config.datasetsDir)

        # Fine the annotation files.
        if file in files:
            if file.split('.')[-1] == 'odgt':
                labelFiles.append(file)

        # Move the images from Images__train#num to Imgaes__train.
        for trainFile in ["Images__train01", "Images__train02", "Images__train03"]:
            currentDir = os.path.join(os.getcwd(), trainFile))
            files = os.listdir(currentDir)
            for file in files:
                shutil.move(os.path.join(currentDir, file), self.trainImgDir)


        for labelFile in labelFiles:
            if labelFile.find('train') != -1:
                imgDir = self.trainImgDir
                type = 'train'
            else:
                imgDir = self.valImgDir
                type = 'valid'

            with open(os.path.join(self.labelDir, labelFile)) as jsonFile:
                jsonData = json.load(jsonFile)
                self.progressCnt = 0
                self.length = len(jsonData)
                printProgress(self.progressCnt, self.length, (' CrowdHuman ' + type + ' Parsing Progress: '), 'Complete')

                for image in jsonData:
                    filename = image["ID"] + '.jpg'
                    img = cv2.imread(os.path.join(imgDir, filename))
                    size = {
                        "width": float(img.shape[1]),
                        "height": float(img.shape[0])
                    }
                    info = {
                        "path": imgDir,
                        "type":type
                    }
                    for label in image['gtboxes']:
                        if 'extra' in label and 'ignore' in label['extra'] and label['extra']['ignore'] == 1:
                            continue
                        if 'extra' in label and 'unsure' in label['extra'] and label['extra']['unsure'] == 1:
                            continue
                        bbox = {
                        "xmin": float(label['fbox'][0]),
                        "ymin": float(label['fbox'][1]),
                        "xmax": float(label['fbox'][0]) + float(label['fbox'][2]),
                        "ymax": float(label['fbox'][1]) + float(label['fbox'][3])
                        }
                        bboxInfo = {
                            "name": 'person',
                            "bbox": bbox
                        }

                        if filename in data:
                            obj_idx = data[filename]["objects"]["num_obj"] + 1
                            data[filename]["objects"][str(obj_idx)] = bboxInfo
                            data[filename]["objects"]["num_obj"] += 1
                        else:
                            obj = {
                                "num_obj": 1,
                                "1": bboxInfo
                            }
                            data[filename] = {
                                "info": info,
                                "size": size,
                                "objects": obj
                            }
                        bbox = {}
                        bboxInfo = {}

                    self.progressCnt += 1
                    printProgress(self.progressCnt, self.length, (' CrowdHuman ' + type + ' Parsing Progress: '), 'Complete')
        return data

class coco_yolo:
    def __init__(self, config):
        self.true = True

class aihub:
    def __init__(self, config):

        self.currentDir = config.datasetsDir
        self.xmlFiles = []
        self.imgDir = {}
        self.length = 0
        self.progressCnt = 0
        print("[Info] Load AIHub datasets.")
    def search(self, directory):
        #try:
        files = os.listdir(directory)
        for file in files:
            filePath = os.path.join(directory, file)
            if os.path.isdir(filePath):
                self.search(filePath)
            else:
                _, fextension = os.path.splitext(file)
                if fextension == '.xml':
                    self.xmlFiles.append(filePath)
        #except PermissionError:
            #pass
    def genImgDirDict(self, directory):
        #try:
        files = os.listdir(directory)
        for file in files:
            filePath = os.path.join(directory, file)
            if os.path.isdir(filePath):
                self.genImgDirDict(filePath)
            else:
                _, fextension = os.path.splitext(file)
                if fextension == '.jpg' or fextension == '.png':
                    self.imgDir[file] = directory# filePath
        #except PermissionError:
            #pass

    def parse(self):
        data = {}
        self.search(self.currentDir)
        self.genImgDirDict(self.currentDir)
        self.length = len(self.xmlFiles)
        printProgress(self.progressCnt, self.length, ' AIHub Parsing Progress: ', 'Complete')

        for xml in self.xmlFiles:
            doc = ET.parse(xml)
            root = doc.getroot()

            for image in root.findall('image'):
                filename = image.attrib['name']
                type = 'valid' if random.randrange(0, 10) == 0 else 'train'
                size = {
                    "width": float(image.attrib['width']),
                    "height": float(image.attrib['height'])
                }
                info = {
                    "path": self.imgDir[filename],
                    "type": type
                }
                for box in image.iter('box'):
                    bbox = {
                        "xmin": float(box.attrib['xtl']),
                        "ymin": float(box.attrib['ytl']),
                        "xmax": float(box.attrib['xbr']),
                        "ymax": float(box.attrib['ybr'])
                    }

                    bboxInfo = {
                        "name": box.attrib['label'],
                        "bbox": bbox
                    }

                    if filename in data:
                        obj_idx = data[filename]["objects"]["num_obj"] + 1
                        data[filename]["objects"][str(obj_idx)] = bboxInfo
                        data[filename]["objects"]["num_obj"] += 1
                    else:
                        obj = {
                            "num_obj": 1,
                            "1": bboxInfo
                        }
                        data[filename] = {
                            "info": info,
                            "size": size,
                            "objects": obj
                        }
                    bbox = {}
                    bboxInfo = {}

            self.progressCnt += 1
            printProgress(self.progressCnt, self.length, ' AIHub Parsing Progress: ', 'Complete')
        return data
