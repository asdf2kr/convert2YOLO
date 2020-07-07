'''
* convert2YOLO
* Version: 1.0
* Author: gbsim (asdf2kr@naver.com)
* Since 2020-06-30
'''
import os
import sys
import json
import shutil
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
                                    "ignore":
                                            "bbox" : <list>
                                                        {
                                                            "xmin" : <float>
                                                            "ymin" : <float>
                                                            "xmax" : <float>
                                                            "ymax" : <float>
                                                        }
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
    def __init__(self, config):
        self.currentDir = config.datasetsDir
        self.trainImgDir = os.path.join(config.datasetsDir, 'train2017')
        self.valImgDir = os.path.join(config.datasetsDir, 'val2017')
        self.labelDir = os.path.join(config.datasetsDir, 'annotations')
        self.labelFiles = [os.path.join(self.labelDir, 'instances_train2017.json'), os.path.join(self.labelDir, 'instances_val2017.json')]
        self.progressCnt = 0

        print("[Info] Load coco2017 datasets.")

    # https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch
    def parse(self):
        data = {}
        for labelFile in self.labelFiles:
            if labelFile.find('train') != -1:
                imgDir = self.trainImgDir
                type = 'train'
            else:
                imgDir = self.valImgDir
                type = 'valid'

            json_data = json.load(open(labelFile))
            images_info = json_data["images"]
            cls_info = json_data["categories"]
            self.progressCnt = 0
            length = len(json_data["annotations"])
            printProgress(self.progressCnt, length, ' COCO Parsing Progress {}: '.format(type), 'Complete')

            info = {
                "path": imgDir,
                "type": type
            }

            for anno in json_data["annotations"]:
                imageId = anno["image_id"]
                clsId = anno["category_id"]

                for img_info in images_info:
                    if img_info["id"] == imageId:
                        filename, imgWidth, imgHeight = img_info["file_name"], img_info["width"], img_info["height"]

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

                bboxInfo = {
                    "name": cls,
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

                self.progressCnt += 1
                printProgress(self.progressCnt, length, ' COCO Parsing Progress: {}'.format(type), 'Complete')
        return data
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
    def __init__(self, config):
        self.currentDir = config.datasetsDir
        self.imgDir = os.path.join(self.currentDir, 'Insight-MVT_Annotation_Train')
        self.labelDir = os.path.join(self.currentDir, 'label', 'for_detection', 'DETRAC-Train-Annotations-XML')

        self.imgDir = []
        self.xmlFiles = []
        self.progressCnt = 0
    def search(self, directory):
        #try:
        files = os.listdir(directory)
        for file in files:
            filePath = os.path.join(directory, file)
            if os.path.isdir(filePath):
                self.search(filePath)
            else:
                fileName, fextension = os.path.splitext(file)
                if fextension == '.xml':
                    self.xmlFiles.append(filePath)
                    self.imgDir.append(os.path.join(self.currentDir, fileName))

    def parse(self):
        data = {}
        ig_bboxes = []
        self.search(self.labelDir)
        self.length = len(self.xmlFiles)
        #printProgress(self.progressCnt, self.length, ' UA-Detrac Parsing Progress: ', 'Complete')
        size = {
            "width": float(960),
            "height": float(540.),
        }
        for index, xml in enumerate(self.xmlFiles):

            doc = ET.parse(xml)
            root = doc.getroot()
            for ig in root.findall('ignored_region'):
                for box in ig.iter('box'):
                    ig_bbox = {
                        "xmin": float(box.attrib['left']),
                        "ymin": float(box.attrib['top']),
                        "xmax": float(box.attrib['left']) + float(box.attrib['width']),
                        "ymax": float(box.attrib['top']) + float(box.attrib['height'])
                    }
                    ig_bboxes.append(ig_bbox)

            for img in root.iter('frame'):

                imgId = img.attrib['num']
                filename = os.path.join(self.imgDir[index],('img' + ('%05d' % int(img.attrib['num'])) + '.jpg')) # '%03d' % 1          # '001'

                type = 'valid' if random.randrange(0, 10) == 0 else 'train'
                info = {
                    "path": self.imgDir[index],
                    "type": type
                }
                for target in img.findall('target_list/target'):
                    bbox = {
                        "xmin": float(target[0].attrib['left']),
                        "ymin": float(target[0].attrib['top']),
                        "xmax": float(target[0].attrib['left']) + float(target[0].attrib['width']),
                        "ymax": float(target[0].attrib['top']) + float(target[0].attrib['height'])
                    }
                        #print(box.tag, box.attrib)
                    bboxInfo = {
                        "name": target[1].attrib['vehicle_type'],
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
            printProgress(self.progressCnt, self.length, ' UA-Detrac Parsing Progress: ', 'Complete')
        return data


class crowdhuman:
    def __init__(self, config):
        self.config = config
        self.fextension = '.jpg'
        print("[Info] Load crowdhuman dataset.")
        self.trainImgDir = os.path.join(self.config.datasetsDir, 'Images__train')
        self.valImgDir = os.path.join(self.config.datasetsDir, 'Images__val')
        if not os.path.exists(self.trainImgDir):
            os.makedirs(self.trainImgDir)

    def parse(self):
        files = os.listdir(self.config.datasetsDir)

        # Fine the annotation files.
        labelFiles = []
        for file in files:
            if file.split('.')[-1] == 'odgt':
                labelFiles.append(file)

        # Move the images from Images__train#num to Imgaes__train.
        for trainFile in ["Images__train01", "Images__train02", "Images__train03"]:
            currentDir = os.path.join(self.config.datasetsDir, trainFile)
            files = os.listdir(currentDir)

            self.progressCnt = 0
            self.length = len(files)
            printProgress(self.progressCnt, self.length, ' CrowdHuman {} moving Progress: '.format(trainFile), 'Complete')
            for file in files:
                shutil.move(os.path.join(currentDir, file), self.trainImgDir)
                self.progressCnt += 1
                printProgress(self.progressCnt, self.length, ' CrowdHuman {} moving Progress: '.format(trainFile), 'Complete')


        for labelFile in labelFiles:
            if labelFile.find('train') != -1:
                imgDir = self.trainImgDir
                type = 'train'
            else:
                imgDir = self.valImgDir
                type = 'valid'

            with open(os.path.join(self.config.datasetsDir, labelFile)) as jsonFile:
                print("[info] file ", os.path.join(self.config.datasetsDir, labelFile))
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
                #ran = random.randrange(0,3)
                #if ran != 1:
                #    continue
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
