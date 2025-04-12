# TriDeFusion denoising method
<!-- 
![GitHub language count](https://img.shields.io/github/languages/top/Biomed-imaging-lab/SpineTool?style=for-the-badge)
[![GNU License](https://img.shields.io/github/license/Biomed-imaging-lab/SpineTool.svg?style=for-the-badge)](LICENSE)
[![Language](https://img.shields.io/badge/python_version-_3.12-green?style=for-the-badge)]()
[![Language](https://img.shields.io/badge/Anaconda-%E2%89%A5_2022.10-green?style=for-the-badge)]()
[![Issues](https://img.shields.io/github/issues/Biomed-imaging-lab/SpineTool?style=for-the-badge)](https://github.com/Biomed-imaging-lab/SpineTool/issues) -->

![GitHub repo size](https://img.shields.io/github/repo-size/shiveshc/TriDeFusion)
![GitHub release (latest by date)](https://img.shields.io/github/v/release/shiveshc/NIDDL)
![Python version](https://img.shields.io/badge/python-v3.12-blue)
![GitHub](https://img.shields.io/github/license/shiveshc/NIDDL)


<br />
<div align="center">
  <a href="https://github.com/Biomed-imaging-lab/TriDeFusion">
    <img src="figures/logo.jpeg" alt="TriDeFusion" width="60%" style="display:block;line-height:0; vertical-align: middle;font-size:0px">
  </a>

  <h2 align="center">TriDeFusion</h2>

  _Tri - three-dimensional images , De - denoising task, Fusion - integration of multiple techniques._

  Dendritic spine analysis tool for dendritic spine image segmentation, dendritic spine morphologies extraction, analysis and clustering.

  <div align="center" style="line-height: 1;">
  <a href="https://www.deepseek.com/" target="_blank" style="margin: 2px;">
    <img alt="Homepage" src="https://github.com/deepseek-ai/DeepSeek-V2/blob/main/figures/badge.svg?raw=true" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://chat.deepseek.com/" target="_blank" style="margin: 2px;">
    <img alt="Chat" src="https://img.shields.io/badge/🤖%20Chat-DeepSeek%20R1-536af5?color=536af5&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://huggingface.co/TriDeFusion" target="_blank" style="margin: 2px;">
    <img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-TriDeFusion-ffc107?color=ffc107&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>

  <img alt="License" src="https://img.shields.io/badge/License-GPL-green" style="display: inline-block; vertical-align: middle;"/>


  <p align="center">
    <br />
    <a href="https://doi.org/10.1109/SIBIRCON63777.2024.10758532"><strong> Explore the research paper »</strong></a>
    <br />
    <a href="#Citation">Cite</a>
    ·
    <a href="https://appliedmath.gitlab.yandexcloud.net/lmn/frt">FRT web service</a>
    ·
    <a href="https://github.com/gerasimenkoab/simple_psf_extractor">FRT Desktop GitHub</a>
    ·
    <a href="https://static-content.springer.com/esm/art%3A10.1038%2Fs41598-023-37406-4/MediaObjects/41598_2023_37406_MOESM1_ESM.pdf">Read Tutorial</a>
    ·
    <a href="mailto:zolin.work@yandex.ru&subject=TriDeFusion_feedback">Connect</a>
  </p>

[![Share](https://img.shields.io/badge/share-1877F2?logo=facebook&logoColor=white)](https://www.facebook.com/sharer/sharer.php?u=https://github.com/Biomed-imaging-lab/TriDeFusion)
[![Share](https://img.shields.io/badge/share-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/shareArticle?mini=true&url=https://github.com/Biomed-imaging-lab/TriDeFusion)
[![Share](https://img.shields.io/badge/share-FF4500?logo=reddit&logoColor=white)](https://www.reddit.com/submit?title=Check%20out%20this%20project%20on%20GitHub:%20https://github.com/Biomed-imaging-lab/TriDeFusion)
[![Share](https://img.shields.io/badge/share-0088CC?logo=telegram&logoColor=white)](https://t.me/share/url?url=https://github.com/Biomed-imaging-lab/TriDeFusion&text=Check%20out%20this%20project%20on%20GitHub)
</div>

[Ivan, Z., Vyacheslav, C., Ekaterina, P. (2025). FRT (Fluorescence Restoration Techniques) - Integrated web and desktop solution for advanced fluorescence microscopy images denoising and deconvolution. 
10.1109/SIBIRCON63777.2024.10758532.](https://doi.org/10.1109/SIBIRCON63777.2024.10758532)

[Ivan, Z., Vyacheslav, C., Ekaterina, P. (2024). TriDeFusion: Enhanced denoising algorithm for 3D fluorescence microscopy images integrating modified Noise2Noise and Non-local means. IEEE International Multi-Conference on Engineering, Computer and Information Sciences (SIBIRCON). 
10.1109/SIBIRCON63777.2024.10758532.](https://doi.org/10.1109/SIBIRCON63777.2024.10758532)

<div align="center" style="display: flex; justify-content: center; align-items: center;">
  <figure style="display: inline-block; margin: 0 10px;">
    <img src="figures/result_dsred.gif" alt="Description of image" width="50%">
    <figcaption>Figure 1: Results of TriDeFusion</figcaption>
  </figure>
  <figure style="display: inline-block; margin: 0 10px;">
    <img src="figures/result_dsred.gif" alt="Description of image" width="50%">
    <figcaption>Figure 2: Results of TriDeFusion</figcaption>
  </figure>
</div>


## Overview

_Fluorescence microscopy is a technique for obtaining images of luminous objects of small size. It is widely used in fields ranging from materials science to neurobiology. Fluorescence microscopy has several advantages over other forms of microscopy, offering high sensitivity and specificity. However, it often results in images with noise and distortions, complicating subsequent analysis. This paper introduces the TriDeFusion algorithm for 3D image denoising, integrating Non-Local Means (NLM) and Modified Noise2Noise (N2N) techniques. Our results show that TriDeFusion significantly improves image quality, particularly in preserving details while reducing noise. In experiments with synthetic data, the combined methods outperformed individual approaches in both Root Mean Square Error (RMSE) and Peak Signal-to-Noise Ratio (PSNR) metrics, achieving up to a 54% reduction in RMSE and a 20% increase in PSNR. For real data, our algorithm demonstrated a significant reduction of noise mean intensity by over 50% and variance by 33%, confirming its robustness and effectiveness across different noise levels and data types._

<div align="center">
<figure>
  <img src="figures/frt_service.png" alt="Description of image">
  <figcaption>Figure 1: Denoising method</figcaption>
</figure>

<figure>
  <img src="figures/frt_service.png" alt="Description of image">
  <figcaption>Figure 2: This is a beautiful caption below the image.</figcaption>
</figure>
</div>

<!-- <div align="center">
  <img src="https://github.com/deepseek-ai/DeepSeek-V2/blob/main/figures/logo.svg?raw=true" width="60%" alt="DeepSeek-V3" />
</div>
<hr>
<div align="center" style="line-height: 1;">
  <a href="https://www.deepseek.com/" target="_blank" style="margin: 2px;">
    <img alt="Homepage" src="https://github.com/deepseek-ai/DeepSeek-V2/blob/main/figures/badge.svg?raw=true" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://chat.deepseek.com/" target="_blank" style="margin: 2px;">
    <img alt="Chat" src="https://img.shields.io/badge/🤖%20Chat-DeepSeek%20R1-536af5?color=536af5&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://huggingface.co/deepseek-ai" target="_blank" style="margin: 2px;">
    <img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-DeepSeek%20AI-ffc107?color=ffc107&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>

[![Share](https://img.shields.io/badge/share-1877F2?logo=facebook&logoColor=white)](https://www.facebook.com/sharer/sharer.php?u=https://github.com/Biomed-imaging-lab/SpineTool)
[![Share](https://img.shields.io/badge/share-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/shareArticle?mini=true&url=https://github.com/Biomed-imaging-lab/SpineTool)
[![Share](https://img.shields.io/badge/share-FF4500?logo=reddit&logoColor=white)](https://www.reddit.com/submit?title=Check%20out%20this%20project%20on%20GitHub:%20https://github.com/Biomed-imaging-lab/SpineTool)
[![Share](https://img.shields.io/badge/share-0088CC?logo=telegram&logoColor=white)](https://t.me/share/url?url=https://github.com/Biomed-imaging-lab/SpineTool&text=Check%20out%20this%20project%20on%20GitHub)

</div> -->


## System requirements

- Python 3.12.8
- PyTorch 2.0
- CUDA 11.8
- cuDNN 8.9
- NVIDIA GPU with at least 8GB VRAM

## Directory structure

<details>
  <summary>Click to unfold the directory tree</summary>

```
TriDeFusion
|---bin # 
|---|---download_dataset.sh 
|---|---download_pretrained.sh
|---|---unzip_dataset.sh
|---config #
|---|---inference_config.yml
|---|---test_config.yml
|---|---train_config.yml
|---dataset #
|---experiments #
|---figures #
|---models #
|---tests #
|---utils #
|---notebooks #
|---app.py #
|---inference.py #

|---|---|---__init__.py
|---|---|---utils.py
|---|---|---network.py
|---|---|---model_3DUnet.py
|---|---|---data_process.py
|---|---|---buildingblocks.py
|---|---|---test_collection.py
|---|---|---train_collection.py
|---|---|---movie_display.py
|---|---notebooks
|---|---|---demo_train_pipeline.ipynb
|---|---|---demo_test_pipeline.ipynb
|---|---|---DeepCAD_RT_demo_colab.ipynb
|---|---datasets
|---|---|---DataForPytorch # project_name #
|---|---|---|---data.tif
|---|---pth
|---|---|---ModelForPytorch
|---|---|---|---model.pth
|---|---|---|---model.yaml
|---|---onnx
|---|---|---ModelForPytorch
|---|---|---|---model.onnx
|---|---results
|---|---|--- # test results#
```

- **DeepCAD_RT_pytorch** contains the Pytorch implementation of DeepCAD-RT (Python scripts, Jupyter notebooks, Colab notebook)
- **DeepCAD_RT_GUI** contains all C++ and Matlab files for the real-time implementation of DeepCAD-RT

</details>

## Installation

0. Before installation, you must install the [Miniconda (Anaconda)](https://docs.anaconda.com/miniconda/install/), [Docker*](https://docs.docker.com/get-docker/) and [Makefile*](docs/make_doc.md).

    `* - Optional`

1. Clone the repository

    ```bash
    git clone https://github.com:Biomed-imaging-lab/TriDeFusion.git
    cd TriDeFusion
    ```

2. Create and activate a conda environment

    ```bash
    make install
    ```


<!-- 3. Install dependencies

### PIP Option

```bash
pip install -r requirements.txt
```

### Conda Option

```bash


conda create -n frt python=3.10
conda activate frt
pip install -r requirements.txt
``` -->



## Dataset

- [Fluorescence microscopy denoising (FMD) dataset](https://drive.google.com/drive/folders/1aygMzSDdoq63IqSk-ly8cMq0_owup8UM)

  - [FMD GitHub](https://github.com/yinhaoz/denoising-fluorescence)

## Training

There are two ways to train the model. First you can train the model using the following command line arguments. Second you can use a config file (`train_config.yml`). The third way is to use a jupyter notebook ([`netrwok_trainer.ipynb`](notebooks/netrwok_trainer.ipynb)).


1. Using a command line.

    ```bash
    python train.py \
        --exp-name="fluoro-msa" \
        --data-root="./dataset" \
        --imsize=256 \
        --chunk-size=128 \
        --offset-size=64 \
        --in-channels=1 \
        --out-channels=1 \
        --transform="center_crop" \
        --epochs=250 \
        --batch-size=32 \
        --loss-params=1.0,0.2,0.1 \ # alpha, beta, gamma for Loss function
        --lr=5e-4 \
        --cuda=0 \
        --test-group=19 \
        --noise-levels-train=1,2,4,8,16 \
        --noise-levels-test=1 \
        --training-type="standard" # "distillation" for distillation training (UNet-Attention)
    ```

2. Using a config file (`train_config.yml`):

    ```
    python train.py --config train_config.yml
    ```

## Validation

## Inference

1. Using a command line

    ```bash
    python inference.py \
        --noisy_img ./test_images/noisy_tubes.tif \
        --denoise_method tri_de_fusion \
        --model_path ./experiments/n2n/models/best_model.pth \
        --output ./test_images/denoised_tubes.tif
    ```

2. Using a config file (`inference_config.yml`):

    ```
    python inference.py --config inference_config.yml
    ```

## Testing

```bash
python test.py \
    --exp_name n2n \
    --exp_dir ./experiments \
    --cuda 0
```

## FastAPI service

### Introduction

### Docker deployment

```bash
make deploy
```

## Citation

If you use this code please cite the companion paper where the original method appeared:

Li, X., Zhang, G., Wu, J. et al. Reinforcing neuron extraction and spike inference in calcium imaging using deep self-supervised denoising. Nat Methods (2021). https://doi.org/10.1038/s41592-021-01225-0

```latex
%FRT, TriDeFusion-v2 citation
@inproceedings{zhang2018poisson,
    title={FRT (Fluorescence Restoration Techniques) - Integrated desktop and web platform solution for advanced fluorescence microscopy denoising and deconvolution techniques},
    author={Ivan Zolin and Alexander Gerasimenko and Vyacheslav Chukanov and Ekaterina Pchitskaya},
    booktitle={CVPR},
    year={2025}
}

%TriDeFusion-v1 citation
@INPROCEEDINGS{10758532,
  author={Zolin, Ivan and Chukanov, Vyacheslav and Pchitskaya, Ekaterina},
  booktitle={2024 IEEE International Multi-Conference on Engineering, Computer and Information Sciences (SIBIRCON)}, 
  title={TriDeFusion: Enhanced denoising algorithm for 3D fluorescence microscopy images integrating modified Noise2Noise and Non-local means}, 
  year={2024},
  volume={},
  number={},
  pages={211-216},
  keywords={Three-dimensional displays;PSNR;Microscopy;Noise;Noise reduction;Fluorescence;Filtering algorithms;Sensitivity and specificity;Root mean square;Synthetic data;fluorescence microscopy;confocal microscopy;denoising;computer vision;deep learning;convolution neural network},
  doi={10.1109/SIBIRCON63777.2024.10758532}}
```

## Baselines

- Noise2Noise
- Non-local Means
- UNet
- CARE
- 3D-RCAN
- DeepCAD-RT


## License

`TriDeFusion` is distributed under the terms of the [GPL-3.0](https://spdx.org/licenses/GPL-3.0-or-later.html) license.


## Acknowledgments

- Thanks to the contributors of the [FRT](https://appliedmath.gitlab.yandexcloud.net/lmn/frt) project for the inspiration and the codebase.
- This project is supported by Laboratory of biomedical imaging and data analysis.

## Project status

If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.

## Contact 

If you have any questions, please raise an issue or contact us at service@deepseek.com.
