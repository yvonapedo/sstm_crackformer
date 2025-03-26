#  Shift-Semantic Transformer and Feature Reconstruction for Pavement Crack Segmentation

Authors: *Yvon Apedo*, *Huanjie Tao*

---


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

