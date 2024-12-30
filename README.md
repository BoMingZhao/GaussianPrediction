# GaussianPrediction [SIGGRAPH2024]
### [Project Page](https://zju3dv.github.io/gaussian-prediction/) | [arXiv](https://arxiv.org/abs/2405.19745) | [Supplementary](https://raw.githubusercontent.com/BoMingZhao/open_access_assets/main/GaussianPrediction/supp.pdf)
This is the official pytorch implementation of **GaussianPrediction: Dynamic 3D Gaussian Prediction for Motion Extrapolation and Free View Synthesis**

## Installation
```bash
git clone https://github.com/BoMingZhao/GaussianPrediction.git --recursive
cd GaussianPrediction

# conda env create -f environment.yml
conda create -n gaussian_prediction python=3.7.13
conda activate gaussian_prediction

bash scripts/utils/env.sh
```

**Note:** We tested our code on PyTorch 1.13.1 with CUDA 11.7 and CUDA 11.8, and no issues were observed. However, we found that when using PyTorch 2.1.0 with CUDA 12.1, the overall reconstruction quality significantly deteriorates. Therefore, please ensure that your installation steps align with the ones we provided. Additionally, if you are aware of the reasons causing this issue, we welcome you to open an issue and share your findings with us.

## Data Preparation
In our paper, We use synthetic dataset [D-NeRF](https://github.com/albertpumarola/D-NeRF?tab=readme-ov-file#download-dataset) and real-world dataset [HyperNeRF](https://github.com/google/hypernerf/releases/tag/v0.1). 

1. Please download the dataset and organize them as follows:
```
GaussianPrediction
├── datasets
│   ├── d-nerf
│   │   │── data
│   │   │   │── bouncingballs
│   │   │   │   │── test
│   │   │   │   │── train
│   │   │   │   │── val
│   │   │   │   │── transforms_test.json
│   │   │   │   │── transforms_train.json
│   │   │   │   │── transforms_val.json
│   │   │   │── mutant
│   │   │   │── trex
│   │   │   │── ...
│   ├── HyperNeRF
│   │   │── cut-lemon
│   │   │   │── rgb
│   │   │   │── camera
│   │   │   │── dataset.json
│   │   │   │── metadata.json
│   │   │   │── scene.json
│   │   │── chickchicken
│   │   │── vrig-3dprinter
│   │   │── torchocolate
```
2. For the HyperNeRF dataset, we follow the [4D-GS](https://github.com/hustvl/4DGaussians) and use the results of COLMAP MVS as the initialization point cloud for the Gaussian Splatting. Therefore, please ensure that COLMAP is installed on your computer and run the following command:
```bash
bash scripts/utils/colmap.sh datasets/HyperNeRF/${Scene}
```


## Usage
You can find all training command in **scripts/train**. You can train a single scene by running:
```bash
bash ./scripts/train/${Dataset}/${Scene}.sh
```
Next, you can train the GCN to obtain prediction results:
```bash
bash ./scripts/predict/${Dataset}/${Scene}.sh
```

or evaluate the performance of NVS:
```bash
bash ./scripts/eval/${Dataset}/${Scene}.sh
```
Alternatively, you can run the following command to train, predict, and evaluate multiple scenes at once:

```bash
# For prediction
bash ./scripts/train/${Dataset}/train_predict.sh
# For evaluation
bash ./scripts/train/${Dataset}/train_eval.sh
```

**Note**: If you want to test the prediction results, please set the `max_time` parameter to `0.8` (this is the value used in our paper, but you can also set it to any value less than 1). If you only want to compare the NVS results, please set max_time to `1.0`.

<details>
  <summary>Some important Command Line Arguments for training</summary>
  <ul>
    <li><strong>max_time</strong> 
    </li>
    Control the length of the visible dataset.
  </ul>


  <ul>
    <li><strong>nearest_num</strong> 
    </li>
    Each 3D Gaussian is influenced by {nearest_num} key points.
  </ul>


  <ul>
    <li><strong>max_keypoints</strong> 
    </li>
    Select one image from every few images of the Reference sequence as the Training view.
  </ul>
  <ul>
    <li><strong>adaptive_points_num</strong> 
    </li>
    The maximum number of key points that can grow.
  </ul>

  <ul>
    <li><strong>feature_amplify</strong> 
    </li>
    Due to the scale difference between motion features and 3D Gaussian positions, feature_amplify is used to scale the feature space.
  </ul>

</details>

## Acknowledgement
We would like to thanks [Guanjun Wu](https://guanjunwu.github.io/) for providing the source code of [4D-GS](https://github.com/hustvl/4DGaussians). We adopted his code and scripts for processing the hypernerf dataset.
We also want to thank [Ziyi Yang](https://github.com/ingra14m) for providing the code and training strategy. Some of our training strategies refer to his paper [Deformable-GS](https://github.com/ingra14m/Deformable-3D-Gaussians).

## Citing
If you find this work helpful in your research. We would appreciate a citation via
```
@inproceedings{zhao2024gaussianprediction,
  title={Gaussianprediction: Dynamic 3d gaussian prediction for motion extrapolation and free view synthesis},
  author={Zhao, Boming and Li, Yuan and Sun, Ziyu and Zeng, Lin and Shen, Yujun and Ma, Rui and Zhang, Yinda and Bao, Hujun and Cui, Zhaopeng},
  booktitle={ACM SIGGRAPH 2024 Conference Papers},
  pages={1--12},
  year={2024}
}
```