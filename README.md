# Shift-Semantic Transformer and Multistage Feature Fusion for Road Defect Segmentation 

Authors: *Yvon Apedo*, *Huanjie Tao*

---
## Abstract

Road defect segmentation involves detecting and accurately outlining defect regions in images, often applied to tasks such as infrastructure assessment and surface defect detection. Given the superior performance of transformers compared to convolutional neural networks, transformers have been increasingly adopted for road defect segmentation. However, most transformer-based approaches tend to overlook the semantic context of images during the encoding phase. To address this issue, we propose a road defect segmentation model dubbed SSTM-Net. We introduce a Shift Semantic module that effectively captures the contextual semantics of images. Additionally, we enhance the decoder by integrating a Split Attention Block to better discern subtle yet significant differences between the crack and the background. Moreover, we incorporate Spatial and Channel Reconstruction Convolution in the Multistage feature fusion module to reduce spatial and channel redundancies. Comprehensive experiments confirm the effectiveness and robustness of our proposed method. For the CrackLS315 dataset, the model obtained an Optimal Dataset Scale (ODS) score of 91.07% and an Optimal Image Scale (OIS) score of 92.04%. On the CrackForest dataset, the model recorded an ODS of 77.88% and an OIS of 79.89%.

---

## Usage
### Datasets
Download the CrackForest, crackls315,[crack537](https://www.sciencedirect.com/science/article/pii/S0925231219300566) dataset and the file follows the following structure.

```
|-- datasets
    |-- crack315
        |-- train
        |   |-- train.txt
        |   |--img
        |   |   |--<crack1.jpg>
        |   |--gt
        |   |   |--<crack1.bmp>
        |-- valid
        |   |-- Valid_image
        |   |-- Lable_image
        |   |-- Valid_result
        ......
```

train.txt format
```
./dataset/crack315/img/crack1.jpg ./dataset/crack315/gt/crack1.bmp
./dataset/crack315/img/crack2.jpg ./dataset/crack315/gt/crack2.bmp
.....
```
### Train

```
python train.py
```
### Valid

Change the 'pretrain_dir','datasetName' and 'netName' in test.py

```
python test.py
```

---
## Baseline Model Implementation

Our code heavily references the code in [CRACKFORMER](https://github.com/LouisNUST/CrackFormer-II).

---

