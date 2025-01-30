#include "postprocess.h"
#include <algorithm>
#include <math.h>

#define ZQ_MAX(a, b) ((a) > (b) ? (a) : (b))
#define ZQ_MIN(a, b) ((a) < (b) ? (a) : (b))

static inline float fast_exp(float x)
{
    // return exp(x);
    union
    {
        uint32_t i;
        float f;
    } v;
    v.i = (12102203.1616540672 * x + 1064807160.56887296);
    return v.f;
}

static inline float IOU(float XMin1, float YMin1, float XMax1, float YMax1, float XMin2, float YMin2, float XMax2, float YMax2)
{
    float Inter = 0;
    float Total = 0;
    float XMin = 0;
    float YMin = 0;
    float XMax = 0;
    float YMax = 0;
    float Area1 = 0;
    float Area2 = 0;
    float InterWidth = 0;
    float InterHeight = 0;

    XMin = ZQ_MAX(XMin1, XMin2);
    YMin = ZQ_MAX(YMin1, YMin2);
    XMax = ZQ_MIN(XMax1, XMax2);
    YMax = ZQ_MIN(YMax1, YMax2);

    InterWidth = XMax - XMin;
    InterHeight = YMax - YMin;

    InterWidth = (InterWidth >= 0) ? InterWidth : 0;
    InterHeight = (InterHeight >= 0) ? InterHeight : 0;

    Inter = InterWidth * InterHeight;

    Area1 = (XMax1 - XMin1) * (YMax1 - YMin1);
    Area2 = (XMax2 - XMin2) * (YMax2 - YMin2);

    Total = Area1 + Area2 - Inter;

    return float(Inter) / float(Total);
}

static float DeQnt2F32(int8_t qnt, int zp, float scale)
{
    return ((float)qnt - (float)zp) * scale;
}

/****** yolov11 ****/
GetResultRectyolov11::GetResultRectyolov11()
{
}

GetResultRectyolov11::~GetResultRectyolov11()
{
}

float GetResultRectyolov11::sigmoid(float x)
{
    return 1 / (1 + fast_exp(-x));
}

int GetResultRectyolov11::GenerateMeshgrid()
{
    int ret = 0;
    if (headNum == 0)
    {
        printf("=== yolov11 Meshgrid  Generate failed! \n");
    }

    for (int index = 0; index < headNum; index++)
    {
        for (int i = 0; i < mapSize[index][0]; i++)
        {
            for (int j = 0; j < mapSize[index][1]; j++)
            {
                meshgrid.push_back(float(j + 0.5));
                meshgrid.push_back(float(i + 0.5));
            }
        }
    }

    printf("=== yolov11 Meshgrid  Generate success! \n");

    return ret;
}

int GetResultRectyolov11::GetConvDetectionResult(int8_t **outputs, std::vector<int> &qnt_zp, std::vector<float> &qnt_scale, std::vector<float> &DetectiontRects)
{
    int ret = 0;
    if (meshgrid.empty())
    {
        ret = GenerateMeshgrid();
    }

    float xmin = 0, ymin = 0, xmax = 0, ymax = 0;
    float cx = 0, cy = 0, cw = 0, ch = 0;

    float cls_temp = 0;
    float cls_vlaue = 0;
    int cls_index = 0;

    int quant_zp_cls = 0, quant_zp_reg = 0;
    float quant_scale_cls = 0, quant_scale_reg = 0;

    std::vector<DetectRect> detectRects;

    int8_t *output_cls = (int8_t *)outputs[0];
    int8_t *output_reg = (int8_t *)outputs[1];

    quant_zp_cls = qnt_zp[0];
    quant_scale_cls = qnt_scale[0];

    quant_zp_reg = qnt_zp[1];
    quant_scale_reg= qnt_scale[1];

    int index = 0, gridIndex = -2;;
    float sfsum = 0;
    float locval = 0;
    float locvaltemp = 0;


    for(int i = 0; i < anchors; i ++)
    {
        cls_index = 0;
        cls_vlaue = -100000;
        gridIndex += 2;
        for(int cl = 0; cl < class_num; cl ++)
        {
            cls_temp = output_cls[i + cl * anchors];
 
            if (cls_temp > cls_vlaue)
            {
                cls_vlaue = cls_temp;
                cls_index = cl;
            }
        }

        cls_vlaue = DeQnt2F32(cls_vlaue, quant_zp_cls, quant_scale_cls);

        if (cls_vlaue > object_thresh)
        {
            cx = DeQnt2F32(output_reg[i + 0 * anchors], quant_zp_reg, quant_scale_reg);
            cy = DeQnt2F32(output_reg[i + 1 * anchors], quant_zp_reg, quant_scale_reg);
            cw = DeQnt2F32(output_reg[i + 2 * anchors], quant_zp_reg, quant_scale_reg);
            ch = DeQnt2F32(output_reg[i + 3 * anchors], quant_zp_reg, quant_scale_reg);

            if (i < mapSize[0][0] * mapSize[0][1])
            {
                index = 0;
            }
            else if (i < mapSize[0][0] * mapSize[0][1] + mapSize[1][0] * mapSize[1][1])
            {
                index = 1;
            }
            else
            {
                index = 2;
            }

            regdfl.clear();
            for (int lc = 0; lc < 4; lc++)
            {
                sfsum = 0;
                locval = 0;
                for (int df = 0; df < 16; df++)
                {
                    locvaltemp = exp(DeQnt2F32(output_reg[(lc * 16  + df) * anchors + i], quant_zp_reg, quant_scale_reg));
                    regdeq[df] = locvaltemp;
                    sfsum += locvaltemp;
                }
                for (int df = 0; df < 16; df++)
                {
                    locvaltemp = regdeq[df] / sfsum;
                    locval += locvaltemp * df;
                }

                regdfl.push_back(locval);
            }

            xmin = (meshgrid[gridIndex + 0] - regdfl[0]) * strides[index];
            ymin = (meshgrid[gridIndex + 1] - regdfl[1]) * strides[index];
            xmax = (meshgrid[gridIndex + 0] + regdfl[2]) * strides[index];
            ymax = (meshgrid[gridIndex + 1] + regdfl[3]) * strides[index];


            DetectRect temp;
            temp.xmin = xmin / input_w;
            temp.ymin = ymin / input_h;
            temp.xmax = xmax / input_w;
            temp.ymax = ymax / input_h;
            temp.classId = cls_index;
            temp.score = cls_vlaue;
            detectRects.push_back(temp);
        }

    }
  

    std::sort(detectRects.begin(), detectRects.end(), [](DetectRect &Rect1, DetectRect &Rect2) -> bool
              { return (Rect1.score > Rect2.score); });

    // std::cout << "NMS Before num :" << detectRects.size() << std::endl;
    for (int i = 0; i < detectRects.size(); ++i)
    {
        float xmin1 = detectRects[i].xmin;
        float ymin1 = detectRects[i].ymin;
        float xmax1 = detectRects[i].xmax;
        float ymax1 = detectRects[i].ymax;
        int classId = detectRects[i].classId;
        float score = detectRects[i].score;

        if (classId != -1)
        {
            // 将检测结果按照classId、score、xmin1、ymin1、xmax1、ymax1 的格式存放在vector<float>中
            DetectiontRects.push_back(float(classId));
            DetectiontRects.push_back(float(score));
            DetectiontRects.push_back(float(xmin1));
            DetectiontRects.push_back(float(ymin1));
            DetectiontRects.push_back(float(xmax1));
            DetectiontRects.push_back(float(ymax1));

            for (int j = i + 1; j < detectRects.size(); ++j)
            {
                float xmin2 = detectRects[j].xmin;
                float ymin2 = detectRects[j].ymin;
                float xmax2 = detectRects[j].xmax;
                float ymax2 = detectRects[j].ymax;
                float iou = IOU(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2);
                if (iou > nms_thresh)
                {
                    detectRects[j].classId = -1;
                }
            }
        }
    }

    return ret;
}
