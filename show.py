'''
* convert2YOLO
* Version: 1.0
* Author: gbsim (asdf2kr@naver.com)
* Since 2020-06-30
'''
import cv2
import os
import random
import argparse

labelList = ["car", "bus", "truck", "motor", "person"]
def main():
    ''' Main function'''
    parser = argparse.ArgumentParser(description='Convert from datasets (COCO, COCO_YOLO, UA-DETRAC, MOT2020, Bdd100k, AIHub, CrowdHuman) to YOLO format.')
    parser.add_argument('--imgDir', type=str, help='directory of images.')
    parser.add_argument('--saveDir', type=str, help='directory of images.')
    # parser.add_argument('--img', type=str, help='select the target image.')
    parser.add_argument('--random', type=int, help='', default=0)
    config = parser.parse_args()
    #config.imgDir="E:\\DB\\copy\\aihub"
    #config.saveDir="E:\\DB\\copy\\save"
    #config.random = 5

    if not os.path.exists(config.saveDir):
        os.makedirs(config.saveDir)

    if config.random > 0:
        file = os.listdir(config.imgDir)
        while(config.random):
            rand = random.randrange(0, len(file))
            if not (file[rand].split('.')[-1] == 'jpg' or file[rand].split('.')[-1] == 'PNG'):
                continue

            imgFile =file[rand]
            print("[Info] select the {}".format(os.path.join(config.imgDir, imgFile)))

            img = cv2.imread(os.path.join(config.imgDir, imgFile), cv2.IMREAD_COLOR)
            height = float(img.shape[0])
            width = float(img.shape[1])
            lines = ''

            with open(os.path.join(config.imgDir, ''.join(file[rand].split('.')[:-1]) + '.txt')) as f:
                lines = f.readlines()

            for j in range(len(lines)):

                label, cx, cy, w, h = lines[j].split(' ')[0:5]
                w, h, cx, cy = float(w), float(h), float(cx), float(cy)

                w = width * w
                h = height * h
                cx = width * cx
                cy = height * cy

                tlx = int(cx - (w / 2))
                tly = int(cy - (h / 2))
                brx = int(cx + (w / 2))
                bry = int(cy + (h / 2))

                img = cv2.rectangle(img, (tlx, tly), (brx, bry), (0, 255, 0), 3)
                cv2.putText(img, labelList[int(label)], (tlx, tly), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255), 2, cv2.LINE_AA)

            cv2.imwrite(os.path.join(config.saveDir, imgFile), img)
            config.random -= 1

if __name__ == '__main__':
    main()
