'''
* convert2YOLO
* Version: 1.0
* Author: gbsim (asdf2kr@naver.com)
* Since 2020-06-30
'''

import xml.etree.ElementTree as ET
import os
import shutil
import pickle
import argparse
from datasets import coco, aihub, bdd100k, mot, detrac, crowdhuman, visdrone, kitti, testworks
from converter import Converter
from util import save_commonData, load_commonData
def main():
    ''' Main function'''
    parser = argparse.ArgumentParser(description='Convert from datasets (COCO, UA-DETRAC, MOT2020, Bdd100k, AIHub, CrowdHuman, VisDrone, kitti) to YOLO format.')
    parser.add_argument('--mode', type=str, help='type of program. e.g. parse/convert/both/analysis')
    parser.add_argument('--datasets', type=str, help='type of datasets (COCO, DETRAC, MOT2020, Bdd100k, AIHub, CrowdHuman, VisDrone, kitti)')
    parser.add_argument('--datasetsDir', type=str, help='directory of datasets')

    parser.add_argument('--save', action = 'store_true', help='Save the file.')
    parser.add_argument('--manipastFile', type=str, help='manipast file', default="manipast.txt")

    parser.add_argument('--copy', action = 'store_true', help='Copy of images')
    parser.add_argument('--copyDir', type=str, help='directory of save target images', default='./copy')

    parser.add_argument('--negative', action = 'store_true', help='')

    config = parser.parse_args()

    config.imgNum = [0 for i in range(5)]
    config.classNum = [0 for i in range(5)]
       # Car, Van, Truck, Tram, Pedestrian, Cyclist
    config.labelDict = {"Van": 0, "Car": 0, "van": 0, "car": 0, "bus": 1, "Truck": 2, "truck": 2, "others": 2, "motorcycle": 3, "motor": 3, "motorbike":3, "Cyclist": 4, "people": 4, "person": 4, "rider": 4, "pedestrian": 4}
    # config.labelDict = {"Cyclist": 0, "people": 0, "person": 0, "rider": 0, "pedestrian": 0}
    # config.labelDict = {"Van": 0, "Car": 0, "van": 0, "car": 0, "bus": 1, "Truck": 2, "truck": 2, "others": 2, "motorcycle": 3, "motor": 3, "motorbike":3}

    if config.datasets == "detrac":
        config.copy == True

    commonData = None
    if config.mode == 'parse' or config.mode == 'both':
        if config.datasets == "coco":
            model = coco(config)
        elif config.datasets == "aihub":
            model = aihub(config)
        elif config.datasets == "bdd100k":
            model = bdd100k(config)
        elif config.datasets == "detrac":
            model = detrac(config)
        elif config.datasets == "crowdhuman":
            model = crowdhuman(config)
        elif config.datasets == "detrac":
            model = detrac(config)
        elif config.datasets == "visdrone":
            model = visdrone(config)
        elif config.datasets == "kitti":
            model = kitti(config)
        elif config.datasets == "testworks":
            config.labelDict = {"pedestrian":0, "vehicle": 1, "suv": 2, "van": 3, "small_truck": 4, "large_truck": 5, "bus": 6, "bike": 7, "motor_cycle": 8, "ambulance": 9, "fire_truck": 10, "police_car": 11, "special_car": 12, "wheel_chair": 13, "wreck_car": 14, "smart_mobility": 15}
            model = testworks(config)
        commonData = model.parse()
        save_commonData(commonData, config.datasets)

    if config.mode == 'convert' or config.mode == 'both':
        converter = Converter(config)
        if commonData == None:
            commonData = load_commonData(config.datasets)
        converter.yolo(commonData, save=config.save, copy=config.copy)

    if config.mode == 'analysis':
        converter = Converter(config)
        if commonData == None:
            commonData = load_commonData(config.datasets)
        converter.analysis(commonData)

if __name__ == '__main__':
    main()
