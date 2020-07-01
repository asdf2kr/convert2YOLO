# convert2YOLO
The purpose of this code is to convert from object detection datasets to YOLO format.

python main.py --datasets [bdd100k/aihub] --mode [parse/convert/both] --copyDir [copy] --datasetDir [C:\\data\\bdd100k] --manipastFile [manipast.txt]

Common data structure
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
                                    "width" : <float>
                                    "height" : <float>
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
                                }
                }
}
  
To-do:
  1. multi-processing
  2. Add (mot2020, ua-detrac)
  3. Revise (config.labelDict)
  4. Debugging (dataset: coco, crowd-human, code: show image, data-augmentation)
  
  
