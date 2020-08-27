
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
                "type": type,
                "ignore": None
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
