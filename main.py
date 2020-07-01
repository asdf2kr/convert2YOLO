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
from datasets import coco, coco_yolo, aihub, bdd100k, mot, detrac, crowdhuman
from converter import Converter
from util import save_commonData, load_commonData
def main():
    ''' Main function'''
    parser = argparse.ArgumentParser(description='Convert from datasets (COCO, COCO_YOLO, UA-DETRAC, MOT2020, Bdd100k, AIHub, CrowdHuman) to YOLO format.')
    parser.add_argument('--mode', type=str, help='type of program. e.g. parse/convert/both')
    parser.add_argument('--datasets', type=str, help='type of datasets')
    parser.add_argument('--copyDir', type=str, help='directory of save target images', default='./copy')
    parser.add_argument('--datasetsDir', type=str, help='directory of datasets')
    parser.add_argument('--manipastFile', type=str, help='manipast file', default="manipast.txt")
    config = parser.parse_args()
    
    config.labelDict = {"car": 0, "bus": 1, "truck": 2, "motorcycle": 3, "moter": 3, "person": 4, "rider": 4}
    commonData = None
    if config.mode == 'parse' or config.mode == 'both':
        if config.datasets == "coco":
            model = coco(config)
        elif config.datasets == "aihub":
            model = aihub(config)
        elif config.datasets == "bdd100k":
            model = bdd100k(config)
        elif config.datasets == "crowdhuman":
            model = crowdhuman(config)
        commonData = model.parse()
        save_commonData(commonData, config.datasets)

    if config.mode == 'convert' or config.mode == 'both':
        converter = Converter(config)
        if not commonData:
            commonData = load_commonData(config.datasets)
        converter.yolo(commonData, save=True, copy=True)

if __name__ == '__main__':
    main()
