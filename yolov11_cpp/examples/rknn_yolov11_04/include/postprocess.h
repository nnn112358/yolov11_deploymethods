#ifndef _POSTPROCESS_H_
#define _POSTPROCESS_H_

#include <iostream>
#include <stdio.h>
#include <unistd.h>
#include <vector>

typedef signed char int8_t;
typedef unsigned int uint32_t;

typedef struct
{
    float xmin;
    float ymin;
    float xmax;
    float ymax;
    float score;
    int classId;
} DetectRect;

// yolov11
class GetResultRectyolov11
{
public:
    GetResultRectyolov11();

    ~GetResultRectyolov11();

    int GenerateMeshgrid();

    int GetConvDetectionResult(int8_t **outputs, std::vector<int> &qnt_zp, std::vector<float> &qnt_scale, std::vector<float> &DetectiontRects);

    float sigmoid(float x);

private:
    std::vector<float> meshgrid;

    int class_num = 80;
    int headNum = 3;

    int input_w = 640;
    int input_h = 640;
    int strides[3] = {8, 16, 32};
    int mapSize[3][2] = {{80, 80}, {40, 40}, {20, 20}};

    std::vector<float> regdfl;
    float regdeq[16] = {0};

    float nms_thresh = 0.45;
    float object_thresh = 0.5;
    
    int anchors = int(input_h / strides[0] * input_w / strides[0] + input_h / strides[1] * input_w / strides[1] + input_h / strides[2] * input_w / strides[2]);

};

#endif
