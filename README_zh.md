# yolov11_deploymethods

&emsp;&emsp;***由于涉及量化、部署两个领域，难免有不对之处，欢迎指正。***

&emsp;&emsp;本仓库对 yolov11（yolov8）尝试了7种不同的部署方法，在最基础的模型上一步一步的去掉解码相关的操作（移到后处理种进行），直到不能再删减，保留到模型最本质的部分。

&emsp;&emsp;随着解码相关的操作越来越多的移入后处理，模型的推理的时耗在减少，后处理的时耗在增加；但也随着解码操作从模型种移除，量化的效果也在逐步变好。

&emsp;&emsp;对每种方法的优势进行了简单总结，不同的平台、不同的时耗或CPU占用需求，总有一种方法是适用的。当然对想了解部署的也是一个很好的参考学习资料。

&emsp;&emsp;春节期间一天一种部署方法，这个春节收获满满。

&emsp;&emsp;[yolov11的7种部署方法代码链接](https://github.com/cqu20160901/yolov11_deploymethods)

&emsp;&emsp;本仓库种使用的板端芯片rk3588，模型yolov11n，模型输入分辨率640x640，检测类别80类。

# 0 七种方法汇总
|  编号 | 推理时耗ms | 后处理时耗ms |  总时耗ms | CPU占用相比上一种方法 | int8量化友好性 | 
| --|--|--|--|--|--|
|第1种|--|--|最少|最简单|不友好|
|第2种|33.75|4.4972| 38.2472 |同1|不友好|
|第3种|32.44|4.4971| 36.4971 |增多|不友好|
|第4种|30.78|4.55 | 35.33 |增多|较友好|
|第5种|30.75|4.84| 35.58 |增多，增加到最多|较友好|
|第6种|30.24|7.08| 37.32 |同5|较友好|
|第7种|30.17|7.34| 37.51 |同5|友好|

&emsp;&emsp;若NPU负载不是瓶颈，当然可以考虑把多的操作放在NPU上，反之将操作往CPU上挪一部分；若量化掉点较多，则可以考虑量化稍微友好的方式。

# 1 代码目录结构
```python
yolov11_onnx  # onnx 推理脚本、模型、测试图片、测试效果图
yolov11_rknn  # 转并推理 rknn 脚本、模型、测试图片、测试效果图
yolov11_cpp   # 部署 rk388 完整 C++ 代码、模型、测试图片、测试效果图
```

# 2 yolov11（v8）的7种部署方法
## 2.1 第1种部署方法
### 模型结构
&emsp;&emsp;按照yolov11官方导出的onnx模型，模型输出直接是类别和解码后的框，模型结构如下图。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a12c54b6a3a1411fa83d738c64d2f628.png)
### onnx效果
&emsp;&emsp;很遗憾的是转换成rknn的int8模型，检测不到任何结果。转换rknn的int8时把模型输出结果都打印出来发现，模型输出的84这个维度，前4个坐标框值正常，后80个得分输出全为0。导致这样的原因：坐标框值取值范围是1-640，而得分输出的值取值范围0-1，使得对量化很不友好，导致模型得分输出的值基本都为0。尝试转rknn的时不进行量化结果输出正常。因此该方法对量化不友好。这种部署方式模型时耗最长，后处理操作最少。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e03cc9bfcb8e4975a8abbbc5e720d21f.png)
### 板端效果
&emsp;&emsp;由于该种部署方法转rknn的int8时量化效果非常差，因此不做板端部署。


## 2.2 第2种部署方法
### 模型结构
&emsp;&emsp;在第1种部署方法的模型基础上，去掉了最后的把坐标框和得分concat在一起的操作。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/0173b07aa82d4e78b0afbb2292ac3778.png)

### onnx效果
&emsp;&emsp;onnx的测试效果和第一种一样，就不再贴图了。
### 板端效果
&emsp;&emsp;第1种部署方法由于坐标框值取和得分的取值范围差异较大，concat在一起使得量化成int8模型基本不可用。这种方法是去掉了最后的concat，量化能正常输出结果，但在板端测试效果不是很好。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7ebdaae21d294fecb1a0e771b03bf802.png)
### 板端时耗
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/0a3e2ebb351d4b069fc9a89a25b19881.png)
## 2.3 第3种部署方法
### 模型结构
&emsp;&emsp;在第2种部署方法的模型基础上，去掉坐标框解码到模型输入尺寸的计算。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7f85139d730147fcaa7f242a925be567.png)
### onnx效果
&emsp;&emsp;onnx的测试效果和第一种一样，就不再贴图了。

### 板端效果
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/2f7bb99652b943cd9089ff8889ff915a.png)
### 板端时耗
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/8c581058510241bda5aca44fdd7e3492.png)

## 2.4 第4种部署方法
### 模型结构
&emsp;&emsp;在第3种部署方法的模型基础上，继续去掉坐标框的DFL，输出2个头。第2、3两种部署方法，可能是对于量化不友好，导致检测效果明显有问题。该种方法检测效果没有明显问题。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/745267836b4e4cfdbfe2e350c17109e6.png)

### onnx效果
&emsp;&emsp;onnx的测试效果和第一种一样，就不再贴图了。
### 板端效果
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e51f271e966e4d0097e706989e4f86b0.png)

### 板端时耗
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c79c5d2d3697402fb014e2845f09f8ff.png)
## 2.5 第5种部署方法
### 模型结构
&emsp;&emsp;在第4种署方法的模型基础上，继续去掉把坐标框和得分进行分开的split，以及得分的sigmoid函数，输出1个头。到达这一种部署方法后，后处理占用cpu不会在增加。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a0a6b37b6eae4c3791cf9480907197e6.png)

### onnx效果
&emsp;&emsp;onnx的测试效果和第一种一样，就不再贴图了。
### 板端效果
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/3adc7921e8ae47329f9138db5223d9ca.png)
### 板端时耗![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ceee40671af44a338dc1a60f34a21c65.png)
## 2.6 第6种部署方法

### 模型结构
&emsp;&emsp;在第5种署方法的模型基础上，继续把三个检测头concat在一起的操起去掉，输出3个头。

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ff20dd54f4e145c2a5e9200734448acf.png)

### onnx效果
&emsp;&emsp;onnx的测试效果和第一种一样，就不再贴图了。
### 板端效果
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/f5a68457490a4eb3a0f336ec67963c9c.png)

### 板端时耗
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/1c2f7275c0164f868883845612601b9d.png)
## 2.7 第7种部署方法

### 模型结构
&emsp;&emsp;在第6种署方法的模型基础上，继续把三个检测头的坐标框和得分concat在一起的操起去掉，输出6个头。到这一步模型内封装的操作能去的都去了，模型的速度达到了最快，量化友好性达到了最好。

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/f5a9bd64fb6f440683f1d47cae8b9b61.png)

### onnx效果
&emsp;&emsp;onnx的测试效果和第一种一样，就不再贴图了。

### 板端效果
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/1f69b76c34334e6bb85981c252132714.png)

### 板端时耗
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c310bcf013474dd89544280da97dfe5c.png)

