# SyMFM6D: Symmetry-aware Multi-directional Fusion for Multi-View 6D Object Pose Estimation

This is the official source code for the paper **"SyMFM6D: Symmetry-aware Multi-directional Fusion 
for Multi-View 6D Object Pose Estimation"** by Fabian Duffhauss et al. accepted to the 
IEEE Robotics and Automation Letters (RA-L) 2023.
The code allows the users to reproduce and extend the results reported in the study. 
Please cite the paper when reporting, reproducing, or extending the results.



## Installation

- Install CUDA 10.1
- Create a virtual environment with all required packages:
    ```shell script
    conda create -n SyMFM6D python=3.6
    conda activate SyMFM6D
    conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=10.1 -c pytorch
    conda install -c anaconda -c conda-forge scikit-learn
    conda install pytorch-lightning==1.4.9 torchmetrics==0.6.0 -c conda-forge
    conda install matplotlib einops tensorboardx pandas opencv==3.4.2 -c conda-forge
    pip install opencv-contrib-python==3.4.2.16
    ```

- Compile the [RandLA-Net](https://github.com/QingyongHu/RandLA-Net) operators:
    ```shell script
    cd models/RandLA/
    sh compile_op.sh
    ```

- Download and install [normalSpeed](https://github.com/hfutcgncas/normalSpeed):
    ```shell script
    git clone https://github.com/hfutcgncas/normalSpeed.git
    cd normalSpeed/normalSpeed
    python3 setup.py install
    ```



## Datasets and Models

- The YCB-Video dataset can be downloaded 
[here](https://drive.google.com/file/d/1if4VoEXNx9W3XCn0Y7Fp15B4GpcYbyYi/view?usp=sharing).
- The MV-YCB SymMovCam dataset can be downloaded 
[here](https://drive.google.com/file/d/16p0keTKr_UQnu7wHS8AgFIFe1GGS1qet/view?usp=share_link). 
Using this dataset requires the 3D models of the YCB-Video dataset which can be downloaded
[here](https://drive.google.com/file/d/1gmcDD-5bkJfcMKLZb3zGgH_HUFbulQWu/view?usp=sharing).

After downloading a new dataset, the zip file needs to be extracted and the paths to the 3D models 
and the datasets need to be modified in [common.py](common.py). For training a new model from scratch, 
a ResNet-34 pre-trained on ImageNet is required which can be downloaded 
[here](https://download.pytorch.org/models/resnet34-333f7ec4.pth). 



## Finding Object Symmetries

Our symmetry-aware training procedure requires the rotational symmetry axes of all objects. 
We compute them once in advance of the training by running the script 
[find_symmetries.py](utils/find_symmetries.py) for each object. For objects with multiple 
symmetry axes, the script needs to be ran multiple times with different initial values for 
the symmetry axis, e.g.
```shell script
python utils/find_symmetries.py --obj_name 024_bowl --symmtype rotational
```

We also provide pre-computed symmetry axes for all objects which we used for the paper in 
[symmetries.txt](datasets/ycb/dataset_config/symmetries.txt).



## Training Models

In the following, we give a few examples how to train models with our SyMFM6D approach on the different datasets.


### Single-View Training on YCB-Video
```shell script
python run.py --dataset ycb --workers 4 --run_name YcbVideo_1view_training --epochs 40 --batch_size 9 \
    --sift_fps_kps 1 --symmetry 1 --n_rot_sym 16
```


### Multi-View Training on YCB-Video
```shell script
python run.py --dataset ycb --workers 4 --run_name YcbVideo_3views_training --epochs 10 --batch_size 3 \
    --sift_fps_kps 1 --symmetry 1 --n_rot_sym 16 --multi_view 1 --set_views 3  --checkpoint single_view_checkpoint.ckpt \
    --lr_scheduler reduce --lr 7e-05
```


### Single-View Training on MV-YCB SymMovCam
```shell script
python run.py --dataset SymMovCam --workers 4 --run_name MvYcbSymMovCam_1view_training --epochs 60 \
    --batch_size 3 --sift_fps_kps 1 --symmetry 1 --n_rot_sym 16
```


### Multi-View Training on MV-YCB SymMovCam
```shell script
python run.py --dataset SymMovCam --workers 4 --run_name MvYcbSymMovCam_1view_training --epochs 60 \
    --batch_size 3 --sift_fps_kps 1 --symmetry 1 --n_rot_sym 16 --multi_view 1 --set_views 3
```



## Evaluating Models

A model can be evaluated by specifying a checkpoint using `--checkpoint <name_of_checkpoint>` 
and by adding the argument `--test`, e.g.
```shell script
python run.py --dataset ycb --workers 4 --run_name YcbVideo_3views_evaluation --batch_size 3 --sift_fps_kps 1 \
    --multi_view 1 --set_views 3 --checkpoint YcbVideo_3views_checkpoint.ckpt 
```



## Runtime

We provide the runtime of our SyMFM6D approach for different number of views 
in the following table:

| Number of Views | Network Forward Time | Pose Estimation Time | Total Time |
| :-------------: | :------------------: |:-------------------: |:---------: |
|       1         |         46 ms        |         14 ms        |    60 ms   |
|       2         |         92 ms        |         19 ms        |   111 ms   |
|       3         |        138 ms        |         25 ms        |   163 ms   |
|       4         |        184 ms        |         30 ms        |   214 ms   |
|       5         |        230 ms        |         36 ms        |   266 ms   |

The network forward time includes the 3D keypoint offset prediction, 
the center point offset prediction, and the prediction of semantic labels.
The pose estimation time represents the time for applying the mean shift clustering 
algorithm and the least-squares fitting for computing the 6D pose of a single object. 
Please note that the usage of the symmetry-aware loss does increase the training time 
slightly, but it does not affect the runtime. 
The runtimes are measured using a single GPU of type NVIDIA Tesla V100 with 32 GB of memory.



## Network Parameters

In the following, we show the tensor shapes of our network architecture: 

| Layer | CNN Tensor Shape | PCN Tensor Shape | 
| :---: | :--------------: | :--------------: |       
|   1   |  H/4, W/4,   64  | N, 8             |
|   2   |  H/4, W/4,   64  | N/4, 64          |
|   3   |  H/8, W/8,  128  | N/16, 128        |
|   4   |  H/8, W/8,  512  | N/64, 256        | 
|   5   |  H/8, W/8, 1024  | N/256, 512       |
|   6   |  H/4, W/4,  256  | N/64, 256        |
|   7   |  H/2, W/2,   64  | N/16, 128        |
|   8   |  H/2, W/2,   64  | N/4, 64          |
 
The multi-layer perceptrons (MLP) of our multi-directional fusion modules 
have the following channel sizes, where c<sub>i</sub> and c<sub>p</sub> denote 
the channel sizes of the image and point features of the respective layer:

|       MLP       |         Channel Sizes         | 
| :-------------: | :---------------------------: |    
| MLP<sub>i</sub> |  c<sub>i</sub>, c<sub>p</sub> |
| MLP<sub>fi</sub>| 2c<sub>p</sub>, c<sub>p</sub> |
| MLP<sub>p</sub> |  c<sub>p</sub>, c<sub>i</sub> |
| MLP<sub>fp</sub>| 2c<sub>i</sub>, c<sub>i</sub> |
     
The channel sizes of the MLPs in the second stage of our network in Fig. 2 of our paper
are as followed:

|           MLP          |                         Channel Sizes                          | 
| :--------------------: | :------------------------------------------------------------: |    
| Keypoint Detection     | c<sub>i</sub> + c<sub>p</sub>, 128, 128, 128, n<sub>cls</sub>  |
| Center Point Detection | c<sub>i</sub> + c<sub>p</sub>, 128, 128, 128, 3n<sub>kps</sub> |
| Semantic Segmentation  | c<sub>i</sub> + c<sub>p</sub>, 128, 128, 128, 3                |

For the output of the network architecture we set c<sub>i</sub> = c<sub>p</sub> = 64. 
n<sub>cls</sub> is the number of classes in the used dataset, 
e.g. n<sub>cls</sub> = 22 for YCB-Video and SymMovCam.
n<sub>kps</sub> is the number of keypoints per object, which we set to eight.



## License

The code of SyMFM6D is open-sourced under the AGPL-3.0 license. See the [LICENSE file](LICENSE) 
for details. The MV-YCB SymMovCam dataset is open-sourced under the CC-BY-SA-4.0 license.

For a list of other open source components included in SyMFM6D, see the
file [3rd-party-licenses.txt](3rd-party-licenses.txt).



## Citation

If you use this work please cite:
```
@article{Duffhauss_2023,
    author    = {Duffhauss, Fabian and Koch, Sebastian and Ziesche, Hanna and Vien, Ngo Anh and Neumann, Gerhard},
    title     = {SyMFM6D: Symmetry-aware Multi-directional Fusion for Multi-View 6D Object Pose Estimation},
    year      = {2023},
    journal   = {IEEE Robotics and Automation Letters (RA-L)}
}
```

