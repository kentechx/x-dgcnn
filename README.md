# DGCNN.pytorch

This is the experimental code for comparing the performance of DGCNN and XDGCNN.
The code is based on the [dgcnn.pytorch](https://github.com/antao97/dgcnn.pytorch).

## Requirements

- Python >= 3.7
- PyTorch >= 1.2
- CUDA >= 10.0
- Package: glob, h5py, sklearn, plyfile, torch_scatter
- x-dgcnn

&nbsp;

## Contents

- [Point Cloud Classification](#point-cloud-classification)
- [Point Cloud Part Segmentation](#point-cloud-part-segmentation)
- [Point Cloud Semantic Segmentation on the S3DIS Dataset](#point-cloud-semantic-segmentation-on-the-s3dis-dataset)
- [Point Cloud Semantic Segmentation on the ScanNet Dataset](#point-cloud-semantic-segmentation-on-the-scannet-dataset)

**Note:** All following commands default use all GPU cards. To specify the cards to use,
add `CUDA_VISIBLE_DEVICES=0,1,2,3` before each command, where the user uses 4 GPU cards with card index `0,1,2,3`. You
can change the card number and indexes depending on your own needs.

&nbsp;

## Point Cloud Classification

### Run the training script:

- 1024 points

```
python main_cls.py --exp_name=cls_1024_dgcnn --num_points=1024 --k=20 --model=xdgcnn_dgcnn
```

- 2048 points

``` 
python main_cls.py --exp_name=cls_2048_dgcnn --num_points=2048 --k=40 --batch_size=32 --model=xdgcnn_dgcnn
```

### Run the evaluation script after training finished:

- 1024 points

``` 
python main_cls.py --exp_name=cls_1024_dgcnn_eval --num_points=1024 --k=20 --eval=True --model_path=outputs/cls_1024_dgcnn/models/model.t7 --model=xdgcnn_dgcnn
```

- 2048 points

``` 
python main_cls.py --exp_name=cls_2048_dgcnn_eval --num_points=2048 --k=40 --eval=True --model_path=outputs/cls_2048_dgcnn/models/model.t7 --model=xdgcnn_dgcnn
```

### Performance:

ModelNet40 dataset (1 NVIDIA 3090 GPU)

Please note that the implementation in [dgcnn.pytorch](https://github.com/antao97/dgcnn.pytorch) consumes more GPU
memory, allowing us to only use a batch size of 16 for 2048 points on a single 3090 GPU. In contrast, our implementation
can handle a batch size of 32 (~17G GPU memory used).

|                                              | Mean Class Acc | Overall Acc |
|:--------------------------------------------:|:--------------:|:-----------:|
|      DGCNN (dgcnn.pytorch, 1024 points)      |      89.3      |    91.9     |
|         DGCNN (x-dgcnn, 1024 points)         |    **90.1**    |  **92.6**   |
| DGCNN (dgcnn.pytorch, 2048 points, batch 16) |      89.1      |    92.6     |
|    DGCNN (x-dgcnn, 2048 points, batch 16)    |      89.3      |    92.7     |
|    DGCNN (x-dgcnn, 2048 points, batch 32)    |    **90.0**    |  **92.7**   |

&nbsp;

## Point Cloud Part Segmentation

**Note:** The training modes **'full dataset'** and **'with class choice'** are different.

- In **'full dataset'**, the model is trained and evaluated in all 16 classes and outputs mIoU 85.2% in this repo. The
  prediction of points in each shape can be any part of all 16 classes.
- In **'with class choice'**, the model is trained and evaluated in one class, for example airplane, and outputs mIoU
  84.5% for airplane in this repo. The prediction of points in each shape can only be one of the parts in this chosen
  class.

### Run the training script:

- Full dataset

``` 
python main_partseg.py --exp_name=partseg 
```

- With class choice, for example airplane

``` 
python main_partseg.py --exp_name=partseg_airplane --class_choice=airplane
```

### Run the evaluation script after training finished:

- Full dataset

```
python main_partseg.py --exp_name=partseg_eval --eval=True --model_path=outputs/partseg/models/model.t7
```

- With class choice, for example airplane

```
python main_partseg.py --exp_name=partseg_airplane_eval --class_choice=airplane --eval=True --model_path=outputs/partseg_airplane/models/model.t7
```

### Performance:

ShapeNet part dataset (4 NVIDIA 3090 GPUs)

|                 | Mean IoU | Airplane |   Bag    |   Cap    |   Car    |  Chair   | Earphone |  Guitar  |  Knife   |   Lamp   |  Laptop  |  Motor   |   Mug    |  Pistol  |  Rocket  | Skateboard |  Table   
|:---------------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:----------:|:--------:| 
|     Shapes      |          |   2690   |    76    |    55    |   898    |   3758   |    69    |   787    |   392    |   1547   |   451    |   202    |   184    |   283    |    66    |    152     |   5271   | 
|      Paper      |   85.2   |   84.0   | **83.4** | **86.7** |   77.8   |   90.6   |   74.7   |   91.2   | **87.5** |   82.8   | **95.7** |   66.3   | **94.9** |   81.1   | **63.5** |    74.5    |   82.6   |
|  dgcnn.pytorch  | **85.4** | **85.2** |   81.9   |   85.1   | **79.6** | **91.2** | **75.0** | **92.1** |   86.9   | **84.1** | **96.1** | **71.2** |   94.7   | **84.0** |   49.1   |  **76.3**  | **82.7** |
| DGCNN (x-dgcnn) |          |          |          |          |          |          |          |          |          |          |          |          |          |          |          |    ****    |          |

## Point Cloud Semantic Segmentation on the S3DIS Dataset

The network structure for this task is slightly different with part segmentation, without spatial transform and
categorical vector. The MLP in the end is changed into (512, 256, 13) and only one dropout is used after 256.

You have to download `Stanford3dDataset_v1.2_Aligned_Version.zip` manually from https://goo.gl/forms/4SoGp4KtH1jfRqEj2
and place it under `data/`

### Run the training script:

This task uses 6-fold training, such that 6 models are trained leaving 1 of 6 areas as the testing area for each model.

- Train in area 1-5

``` 
python main_semseg_s3dis.py --exp_name=semseg_s3dis_6 --test_area=6 
```

### Run the evaluation script after training finished:

- Evaluate in area 6 after the model is trained in area 1-5

``` 
python main_semseg_s3dis.py --exp_name=semseg_s3dis_eval_6 --test_area=6 --eval=True --model_root=outputs/semseg_s3dis/models/
```

- Evaluate in all areas after 6 models are trained

``` 
python main_semseg_s3dis.py --exp_name=semseg_s3dis_eval --test_area=all --eval=True --model_root=outputs/semseg_s3dis/models/
```

### Performance:

Stanford Large-Scale 3D Indoor Spaces Dataset (S3DIS) dataset

|                 | Mean IoU | Overall Acc |
|:---------------:|:--------:|:-----------:|
|      Paper      |   56.1   |    84.1     |
|  dgcnn.pytorch  | **59.2** |  **85.0**   |
| DGCNN (x-dgcnn) |          |             |

## Point Cloud Semantic Segmentation on the ScanNet Dataset

The DGCNN authors do not test on the ScanNet dataset. We try our best to implement the DGCNN model on the dataset.

### Prepare dataset:

You need to change the directory to the `prepare_data/` folder.

```
cd prepare_data/
```

Please download original dataset from [website](http://www.scan-net.org/). You need to place the dataset
under `data/ScanNet/`. The path `data/ScanNet` includes `data/ScanNet/scans/` and `data/ScanNet/scans_test/` folder.

To prepare the Scannet dataset for training and evaluation, run

```
python scannetv2_seg_dataset_rgb21c_pointid.py
```

This will generate four pickle
files: `scannet_train_rgb21c_pointid.pickle`, `scannet_val_rgb21c_pointid.pickle`, `scannet_val_rgb21c_pointid_keep_unanno.pickle`,
and `scannet_test_rgb21c_pointid_keep_unanno.pickle`.

Return to the root directory:

```
cd ..
```

### Run the training script:

```
python main_semseg_scannet.py --exp_name=semseg_scannet
```

To train with both the training split and the validation split, use `--train_val=True`.

You can use [TensorBoard](https://tensorflow.google.cn/tensorboard) to view the training log
under `outputs/semseg_scannet/logs/`.

### Run the evaluation script after training finished:

- Evaluate on the validation set

```
python main_semseg_scannet.py --eval=True --model_path=outputs/semseg_scannet/models/model_200.pth --exp_name=semseg_scannet_val --split=val
```

- Evaluate on the testing set

```
python main_semseg_scannet.py --eval=True --model_path=outputs/semseg_scannet/models/model_200.pth --exp_name=semseg_scannet_test --split=test
```

Since there are no ground-truth labels on the testing set, this script will directly save prediction result. You need to
upload your prediction results to the [website](https://kaldir.vc.in.tum.de/scannet_benchmark/semantic_label_3d) for
evaluation.

### Performance:

The validation set of the ScanNet Dataset

|                 | Mean IoU | wall | floor | cabinet | bed  | chair | sofa | table | door | window | bookshelf | picture | counter | desk | curtain | refrigerator | shower curtain | toilet | sink | bathtub | otherfurniture | 
|:---------------:|:--------:|:----:|:-----:|:-------:|:----:|:-----:|:----:|:-----:|:----:|:------:|:---------:|:-------:|:-------:|:----:|:-------:|:------------:|:--------------:|:------:|:----:|:-------:|:--------------:| 
|  dgcnn.pytorch  |   49.6   | 73.2 | 93.6  |  44.9   | 64.7 | 70.0  | 50.5 | 55.7  | 35.7 |  47.7  |   69.1    |  14.6   |  41.8   | 45.3 |  33.8   |     29.2     |      35.7      |  55.9  | 40.2 |  56.5   |      32.9      | 
| DGCNN (x-dgcnn) |          |      |       |         |      |       |      |       |      |        |           |         |         |      |         |              |                |        |      |         |                |

The testing set of the ScanNet Dataset

|                 | Mean IoU | wall | floor | cabinet | bed  | chair | sofa | table | door | window | bookshelf | picture | counter | desk | curtain | refrigerator | shower curtain | toilet | sink | bathtub | other furniture | 
|:---------------:|:--------:|:----:|:-----:|:-------:|:----:|:-----:|:----:|:-----:|:----:|:------:|:---------:|:-------:|:-------:|:----:|:-------:|:------------:|:--------------:|:------:|:----:|:-------:|:---------------:| 
|  dgcnn.pytorch  |   44.6   | 72.3 | 93.7  |  36.6   | 62.3 | 65.1  | 57.7 | 44.5  | 33.0 |  39.4  |   46.3    |  12.6   |  31.0   | 34.9 |  38.9   |     28.5     |      22.4      |  62.5  | 35.0 |  47.4   |      27.1       |
| DGCNN (x-dgcnn) |          |      |       |         |      |       |      |       |      |        |           |         |         |      |         |              |                |        |      |         |                 |

These is no official results of DGCNN on the ScanNet dataset. You can find our results on
the [website](https://kaldir.vc.in.tum.de/scannet_benchmark/semantic_label_3d) as `DGCNN_reproduce`.
