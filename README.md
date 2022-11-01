# TCCNet

[TCCNet: Temporally Consistent Context-Free Network for Semi-supervised Video Polyp Segmentation](https://www.ijcai.org/proceedings/2022/155)

## 1. datasets
- Train/Test datasets should be prepared in folder `dataset` 
- Folder should be ordered as follows,
```
|-- datasets
|   |-- TrainSet
|   |   |-- CVC-ClinicDB-612
|   |   |   |-- 1,5,8,9,10,12,13,14,18,19,20,21,22,23,25,26,27,28
|   |   |   |   |-- Frame
|   |   |   |   |-- GT
|   |   |   |   |-- border
|   |   |-- CVC-colonDB-300
|   |   |   |-- 0,1,2,4,9,10,11
|   |   |   |   |-- Frame
|   |   |   |   |-- GT
|   |   |   |   |-- border

|   |--TestSet
|   |   |-- CVC-ClinicDB-612-Valid
|   |   |   |-- Frame
|   |   |   |   |-- 2,3,4,11,15,17
|   |   |   |-- GT
|   |   |   |   |-- 2,3,4,11,15,17
|   |   |-- CVC-ClinicDB-612-Test
|   |   |   |-- Frame
|   |   |   |   |-- 0,6,7,16,24
|   |   |   |-- GT
|   |   |   |   |-- 0,6,7,16,24
|   |   |-- CVC-colonDB-300
|   |   |   |-- Frame
|   |   |   |   |-- 3,5,6,7,8,12
|   |   |   |-- GT
|   |   |   |   |-- 3,5,6,7,8,12
|   |   |-- ETIS
|   |   |   |-- Frame
|   |   |   |   |-- 0~25
|   |   |   |-- GT
|   |   |   |   |-- 0~25
```

## 2. Train & Test
- Run `python data/sqc_pathlist.py` to get `sqc_pathlist.npy`
- first stage `python pretraining.py`
- second stage `python main_training.py`
- test `python MyTest.py --load your_model`

