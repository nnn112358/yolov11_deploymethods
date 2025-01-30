import os
import urllib
import traceback
import time
import sys
import numpy as np
import cv2
from rknn.api import RKNN
from math import exp

ONNX_MODEL = './yolov11n.onnx'
RKNN_MODEL = './yolov11n.rknn'
DATASET = './images_list.txt'

QUANTIZE_ON = True

CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush']

class_num = len(CLASSES)
head_num = 3
nms_thresh = 0.45
object_thresh = 0.5

strides = [8, 16, 32]
map_size = [[80, 80], [40, 40], [20, 20]]

input_height = 640
input_width = 640

meshgrid = []


anchors = int(input_height / strides[0] * input_width / strides[0] + input_height / strides[1] * input_width / strides[1] + input_height / strides[2] * input_width / strides[2])


class DetectBox:
    def __init__(self, classId, score, xmin, ymin, xmax, ymax):
        self.classId = classId
        self.score = score
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        

def GenerateMeshgrid():
    for index in range(head_num):
        for i in range(map_size[index][0]):
            for j in range(map_size[index][1]):
                meshgrid.append(j + 0.5)
                meshgrid.append(i + 0.5)
        
        
def IOU(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2):
    xmin = max(xmin1, xmin2)
    ymin = max(ymin1, ymin2)
    xmax = min(xmax1, xmax2)
    ymax = min(ymax1, ymax2)

    innerWidth = xmax - xmin
    innerHeight = ymax - ymin

    innerWidth = innerWidth if innerWidth > 0 else 0
    innerHeight = innerHeight if innerHeight > 0 else 0

    innerArea = innerWidth * innerHeight

    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)

    total = area1 + area2 - innerArea

    return innerArea / total


def sigmoid(x):
    return 1 / (1 + exp(-x))


def NMS(detectResult):
    predBoxs = []

    sort_detectboxs = sorted(detectResult, key=lambda x: x.score, reverse=True)

    for i in range(len(sort_detectboxs)):
        xmin1 = sort_detectboxs[i].xmin
        ymin1 = sort_detectboxs[i].ymin
        xmax1 = sort_detectboxs[i].xmax
        ymax1 = sort_detectboxs[i].ymax
        classId = sort_detectboxs[i].classId

        if sort_detectboxs[i].classId != -1:
            predBoxs.append(sort_detectboxs[i])
            for j in range(i + 1, len(sort_detectboxs), 1):
                if classId == sort_detectboxs[j].classId:
                    xmin2 = sort_detectboxs[j].xmin
                    ymin2 = sort_detectboxs[j].ymin
                    xmax2 = sort_detectboxs[j].xmax
                    ymax2 = sort_detectboxs[j].ymax
                    iou = IOU(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2)
                    if iou > nms_thresh:
                        sort_detectboxs[j].classId = -1
    return predBoxs


def postprocess(outputs, image_h, image_w):
    print('postprocess ... ')

    scale_h = image_h / input_height
    scale_w = image_w / input_width
    
    detectResult = []
    output_cls = outputs[0]
    output_reg = outputs[1]

    grid_index = -2
    
    for i in range(0, anchors):
        cls_index = 0
        cls_vlaue = -1

        grid_index += 2
        
        for cl in range(class_num):
            val = output_cls[i + cl * anchors]
            if val > cls_vlaue:
                cls_vlaue = val
                cls_index = cl
                
        if cls_vlaue > object_thresh:
            cx = output_reg[i + 0 * anchors]
            cy = output_reg[i + 1 * anchors]
            cw = output_reg[i + 2 * anchors]
            ch = output_reg[i + 3 * anchors]

            if i < map_size[0][0] * map_size[0][1]:
                index = 0
            elif i < map_size[0][0] * map_size[0][1] + map_size[1][0] * map_size[1][1]:
                index = 1
            else:
                index = 2

            xmin = (meshgrid[grid_index + 0] - cx) * strides[index]
            ymin = (meshgrid[grid_index + 1] - cy) * strides[index]
            xmax = (meshgrid[grid_index + 0] + cw) * strides[index]
            ymax = (meshgrid[grid_index + 1] + ch) * strides[index]
            
            xmin *= scale_w
            ymin *= scale_h
            xmax *= scale_w
            ymax *= scale_h

            box = DetectBox(cls_index, cls_vlaue, xmin, ymin, xmax, ymax)
            detectResult.append(box)
    # NMS
    print('detectResult:', len(detectResult))
    predBox = NMS(detectResult)

    return predBox


def export_rknn_inference(img):
    # Create RKNN object
    rknn = RKNN(verbose=False)

    # pre-process config
    print('--> Config model')
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], quantized_algorithm='normal', quantized_method='channel', target_platform='rk3588')  # mmse
    print('done')

    # Load ONNX model
    print('--> Loading model')
    ret = rknn.load_onnx(model=ONNX_MODEL)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=QUANTIZE_ON, dataset=DATASET, rknn_batch_size=1)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export RKNN model
    print('--> Export rknn model')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    # Init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    # ret = rknn.init_runtime(target='rk3566')
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')

    # Inference
    print('--> Running model')
    outputs = rknn.inference(inputs=[img])
    rknn.release()
    print('done')

    return outputs


if __name__ == '__main__':
    print('This is main ...')
    GenerateMeshgrid()
    img_path = './test.jpg'
    orig_img = cv2.imread(img_path)
    img_h, img_w = orig_img.shape[:2]
        
    origimg = cv2.resize(orig_img, (input_width, input_height), interpolation=cv2.INTER_LINEAR)
    origimg = cv2.cvtColor(origimg, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(origimg, 0)

    outputs = export_rknn_inference(img)

    out = []
    for i in range(len(outputs)):
        print(outputs[i].shape)
        out.append(outputs[i].reshape(-1))

    predbox = postprocess(out, img_h, img_w)

    print(len(predbox))

    for i in range(len(predbox)):
        xmin = int(predbox[i].xmin)
        ymin = int(predbox[i].ymin)
        xmax = int(predbox[i].xmax)
        ymax = int(predbox[i].ymax)
        classId = predbox[i].classId
        score = predbox[i].score

        cv2.rectangle(orig_img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        ptext = (xmin, ymin)
        title = CLASSES[classId] + "%.2f" % score
        cv2.putText(orig_img, title, ptext, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        
    cv2.imwrite('./test_rknn_result.jpg', orig_img)