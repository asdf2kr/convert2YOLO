'''
* convert2YOLO
* Version: 1.0
* Author: gbsim (asdf2kr@naver.com)
* Since 2020-06-30
'''
import cv2
import random
import argparse
def main():
    ''' Main function'''
    parser.add_argument('--imgDir', type=str, help='directory of images.')
    # parser.add_argument('--img', type=str, help='select the target image.')
    parser.add_argument('--random', type=int, help='', default=0)
    config = parser.parse_args()
    config.imgDir="E:\\DB\\coco_yolo\\images_ped_car_type\\train"
    config.random = 3
    if config.random > 0:
        file = os.listdir(config.imageDir)
        for i in range(config.random):
            rand = random.randrange(0, len(file))

            if file[rand].split('.')[-1] == 'txt':
                i -= 1
                continue

            imgFile =file[rand]
            print("[Info] select the {}".format(imgFile))

            img = cv.imread(imgFile, cv2.IMREAD_COLOR)
            width = img.shape[0]
            height = img.shape[1]

            lines = ''

            with open(file[rand].split('.')[:-1] + '.txt') as f:
                lines = f.readlines()

            for j in range(len(lines)):
                label, cx, cy, w, h = lines[j].split(' ')
                w = width * w
                h = height * h
                cx = cx * w
                cy = cy * h

                tlx = cx - (w / 2)
                tly = cy - (h / 2)
                brx = cx + (w / 2)
                bry = cy + (h / 2)
                img = cv2.rectangle(img, (tlx, tly), (brx, bry), (0, 255, 0), 3)

            cv2.imshow('image-' + i, img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
