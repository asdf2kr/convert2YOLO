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
from datasets import coco, aihub, bdd100k, mot, detrac, crowdhuman
from converter import Converter
from util import save_commonData, load_commonData
def main():
    ''' Main function'''
    parser = argparse.ArgumentParser(description='Convert from datasets (COCO, UA-DETRAC, MOT2020, Bdd100k, AIHub, CrowdHuman) to YOLO format.')
    parser.add_argument('--mode', type=str, help='type of program. e.g. parse/convert/both/analysis')
    parser.add_argument('--datasets', type=str, help='type of datasets (COCO, UA-DETRAC, MOT2020, Bdd100k, AIHub, CrowdHuman)')
    parser.add_argument('--datasetsDir', type=str, help='directory of datasets')

    parser.add_argument('--save', action = 'store_true', help='Save the file.')
    parser.add_argument('--manipastFile', type=str, help='manipast file', default="manipast.txt")

    parser.add_argument('--copy', action = 'store_true', help='Copy of images')
    parser.add_argument('--copyDir', type=str, help='directory of save target images', default='./copy')

    parser.add_argument('--negative', action = 'store_true', help='')

    config = parser.parse_args()

    config.imgNum = [0 for i in range(5)]
    config.classNum = [0 for i in range(5)]

    config.labelDict = {"van": 0, "car": 0, "bus": 1, "truck": 2, "others": 2, "motorcycle": 3, "motor": 3, "motorbike":3, "person": 4, "rider": 4}
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
