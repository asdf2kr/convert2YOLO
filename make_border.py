'''
* convert2YOLO
* Version: 1.0
* Author: gbsim (asdf2kr@naver.com)
* Since 2020-06-30
'''
import os
import cv2
import argparse
from util import printProgress
def main():
    ''' Main function'''
    parser = argparse.ArgumentParser(description='Convert from datasets (COCO, COCO_YOLO, UA-DETRAC, MOT2020, Bdd100k, AIHub, CrowdHuman) to YOLO format.')
    parser.add_argument('--imgDir', type=str, help='directory of images.', default='./copy')
    parser.add_argument('--saveDir', type=str, help='directory of images.', default='./augmentation')
    parser.add_argument('--manipastFile', type=str, help='manipast file', default="manipast.txt")
    config = parser.parse_args()

    manipast = ""
    files = os.listdir(config.imgDir)
    if not os.path.exists(config.saveDir):
        os.makedirs(config.saveDir)

    progressCnt = 0
    printProgress(progressCnt, len(files), ' Data-augmentation Progress: ', 'Complete')

    BLACK = [0, 0, 0]
    for file in files:
        fileName, fextension = os.path.splitext(file)
        if fextension == '.jpg' or fextension =='.png':
            print("[info] file ", file)
            img = cv2.imread(os.path.join(config.imgDir, file), cv2.IMREAD_COLOR)
            height = float(img.shape[0])
            width = float(img.shape[1])

            ratio = 1
            dx = int(width * ratio / 2.0)
            dy = int(height * ratio / 2.0)
            dst = cv2.copyMakeBorder(img, top=dy, bottom=dy, left=dx, right=dx, borderType=cv2.BORDER_CONSTANT, value=BLACK)

            lines = ''
            writeStr = ''
            writeStr2 = ''

            txtFile = fileName + '.txt'
            with open(os.path.join(config.imgDir, txtFile), 'r+') as f:
                lines = f.readlines()


            for j in range(len(lines)):
                label, cx, cy, w, h = lines[j].split(' ')[0:5]
                cx, cy, w, h = float(cx), float(cy), float(w), float(h)

                if min(w, h) > 0.015 or max(w, h) > 0.03:
                    writeStr += "{} {} {} {} {}\n".format(label, cx, cy, w, h)

                labelWidth = width * w
                labelHeight = height * h
                labelCx = cx * w
                labelCy = cy * h

                newCx = (dx + labelCx) / (width + (width * ratio))
                newCy = (dy + labelCy) / (height + (height * ratio))
                newW =  (w / (1 + ratio))
                newH =  (h / (1 + ratio))

                if min(newW, newH) > 0.015 or max(newW, newH) > 0.03:
                    writeStr2 += "{} {} {} {} {}\n".format(label, newCx, newCy, newW, newH)

            if writeStr:
                with open(os.path.join(config.saveDir, txtFile), 'w+') as f:
                    f.write(writeStr)
            if writeStr2:
                with open(os.path.join(config.saveDir, fileName + '-border.txt'), 'w+') as f:
                    f.write(writeStr2)
                cv2.imwrite(os.path.join(config.saveDir, fileName + '-border' + fextension), dst)
                manipast += os.path.join(config.saveDir, fileName + '-border' + fextension) + '\n'

        progressCnt += 1
        printProgress(progressCnt, len(files), ' Data-augmentation Progress: ', 'Complete')

    with open(os.path.join(config.saveDir, config.manipastFile), "a+") as manipastFile:
        manipastFile.write(manipast)

if __name__ == '__main__':
    main()
