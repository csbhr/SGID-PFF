# SGID-PFF

[![LICENSE](https://img.shields.io/badge/license-MIT-green)](https://github.com/csbhr/SGID-PFF/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-0.4.1-%237732a8)](https://pytorch.org/)

#### [Paper](https://ieeexplore.ieee.org/document/9677961) | [Discussion](https://github.com/csbhr/SGID-PFF/issues)
### Self-Guided Image Dehazing Using Progressive Feature Fusion
By [Haoran Bai](https://csbhr.github.io/), [Jinshan Pan](https://jspan.github.io/), Xinguang Xiang, and Jinhui Tang

## Updates
[2022-1-23] Inference logs are available!  
[2022-1-23] Training code is available!  
[2022-1-23] Pre-trained models are available!    
[2022-1-23] Testing code is available!    
[2022-1-23] Paper is available!  

## Abstract
We propose an effective image dehazing algorithm which explores useful information from the input hazy image itself as the guidance for the haze removal. The proposed algorithm first uses a deep pre-dehazer to generate an intermediate result, and takes it as the reference image due to the clear structures it contains. To better explore the guidance information in the generated reference image, it then develops a progressive feature fusion module to fuse the features of the hazy image and the reference image. Finally, the image restoration module takes the fused features as input to use the guidance information for better clear image restoration. All the proposed modules are trained in an end-to-end fashion, and we show that the proposed deep pre-dehazer with progressive feature fusion module is able to help haze removal. Extensive experimental results show that the proposed algorithm performs favorably against state-of-the-art methods on the widely-used dehazing benchmark datasets as well as real-world hazy images.

![overview](https://s4.ax1x.com/2022/01/23/752HGn.png)  

More detailed analysis and experimental results are included in [[Paper]](https://drive.google.com/drive/folders/1QbzoB4rHwIVdyfC80v5ufZc9w4ZF5_SF?usp=sharing) (for academic discussion only).

## Dependencies

- Linux (Tested on Ubuntu 18.04)
- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
- [PyTorch 0.4.1](https://pytorch.org/): `conda install pytorch=0.4.1 torchvision cudatoolkit=9.2 -c pytorch`
- numpy: `conda install numpy`
- matplotlib: `conda install matplotlib`
- opencv: `conda install opencv`
- imageio: `conda install imageio`
- skimage: `conda install scikit-image`
- tqdm: `conda install tqdm`

## Get Started

### Download
- Pretrained models and Datasets can be downloaded from [[Here]](https://drive.google.com/drive/folders/1QbzoB4rHwIVdyfC80v5ufZc9w4ZF5_SF?usp=sharing).
	- If you have downloaded the pretrained models，please put them to './pretrained_models'.
	- If you have downloaded the datasets，please put them to './dataset'.

### Dataset Organization Form
If you prepare your own dataset, please follow the following form:
```
|--dataset  
    |--clear  
        |--image 1 
        |--image 2
            :
        |--image n
    |--hazy
        |--image 1
        |--image 2
        	:
        |--image n
```

### Training
- Download training dataset from [[Here]](https://sites.google.com/view/reside-dehaze-datasets/), or prepare your own dataset like above form.
- Run the following commands to pretrain the deep pre-dehazer:
```
cd ./code
python main.py --template Pre_Dehaze
```
- After deep pre-dehazer pretraining is done, put the trained model to './pretrained_models' and name it as 'pretrain_pre_dehaze_net.pt'.
- Run the following commands to train the Video SR model:
```
python main.py --template ImageDehaze_SGID_PFF
```

### Testing

#### Quick Test
- Download the pretrained models from [[Here]](https://drive.google.com/drive/folders/1QbzoB4rHwIVdyfC80v5ufZc9w4ZF5_SF?usp=sharing).
- Download the testing dataset from [[Here]](https://drive.google.com/drive/folders/1QbzoB4rHwIVdyfC80v5ufZc9w4ZF5_SF?usp=sharing).
- Run the following commands:
```
cd ./code
python inference.py --quick_test SOTS_indoor
	# --quick_test: the results in Paper you want to reproduce, optional: SOTS_indoor, SOTS_outdoor
```
- The dehazed result will be in './infer_results'.

#### Test Your Own Dataset
- Download the pretrained models from [[Here]](https://drive.google.com/drive/folders/1QbzoB4rHwIVdyfC80v5ufZc9w4ZF5_SF?usp=sharing).
- Organize your dataset like the above form.
- Run the following commands:
```
cd ./code
python inference.py --data_path path/to/hazy/images --gt_path path/to/gt/images --model_path path/to/pretrained/model --infer_flag my_test
	# --data_path: the path of the hazy images in your dataset.
	# --gt_path: the path of the gt images in your dataset.
	# --model_path: the path of the downloaded pretrained model.
	# --infer_flag: the flag of this inference.
```
- The dehazed result will be in './infer_results'.

## Citation
```
@article{bai2022self,
    title = {Self-Guided Image Dehazing Using Progressive Feature Fusion},
    author = {Bai, Haoran and Pan, Jinshan and Xiang, Xinguang and Tang, Jinhui},
    journal = {IEEE Transactions on Image Processing},
    volume = {31},
    pages = {1217 - 1229},
    year = {2022},
    publisher = {IEEE}
}
```




