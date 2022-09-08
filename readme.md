# dreamfield-torch-colab (WIP)

A colab friendly toolkit to generate 3D mesh model / video / nerf instance / multiview images of colourful 3D objects by text and image prompts input. Edited by [Shengyu Meng (Simon)](https://twitter.com/meng_shengyu)  

Check the colab for the usage.

Dreamfields-3D is modified from [dreamfields-torch](https://github.com/ashawkey/dreamfields-torch) and [dreamfields](https://github.com/google-research/google-research/tree/master/dreamfields), please check the [Credits.md](./notebook/Credits.md) for details.

**Attention: ** Since I am a beginner of coding, no guarantee for the code quality, and welcome to contribute to this repository :smile: 

vide:

## Main Contributions:
- [x] Integrating Video generation.
- [x] Export obj & glb model with vertex colour.
- [x] Export  360° Video of final model.
- [x] Make it running friendly in colab	
  - [x] Visualizing the training progress in colab.
  - [x] Preview the output video in colab
- [x] Improve the generation quality.
  - [x] Implements multiple CLIP model.
  - [x] Improve the pre-process of the images before feeding into CLIP.
- [x] More useful augments options.
- [ ] Release the colab notebook.

## Compatibility:

- About system: 
  - Colab: Pass on goolge Colab (tested on A100/v100/P100 GPU at 08/09/2022)
  - Ubuntu: The previous version (dreamfields-torch) has successfully ran on Ubuntu 18.04 with RTX 3090. Not tested for the dreamfields-3D yet, but mostly should be fined.
  - Windows: It should be work in windows with proper environment, but I failed to build the raymarching in several windows machine. More test will be required.
- About GUI:
  - When it run on local machine, GUI is supported. However, some new features maybe not available in GUI model.

> 👇 Bellow readme from the dreamfields-torch repository.

-------------------------------

# dreamfields-torch (WIP)

A pytorch implementation of [dreamfields](https://github.com/google-research/google-research/tree/master/dreamfields) as described in [Zero-Shot Text-Guided Object Generation with Dream Fields](https://arxiv.org/abs/2112.01455).

An example of a generated neural field by prompt "cthulhu" viewed in real-time:

https://user-images.githubusercontent.com/25863658/158593558-a52fe215-4276-41eb-a588-cf60c9461cf3.mp4

# Install

The code framework is based on [torch-ngp](https://github.com/ashawkey/torch-ngp).

```bash
git clone https://github.com/ashawkey/dreamfields-torch.git
cd dreamfields-torch
```

### Install with pip
```bash
pip install -r requirements.txt
```
###  install customized verion of pymarchingcubes
```bash
bash scripts/install_PyMarchingCubes.sh
```

### Build extension (optional)
By default, we use [`load`](https://pytorch.org/docs/stable/cpp_extension.html#torch.utils.cpp_extension.load) to build the extension at runtime.
However, this may be inconvenient sometimes.
Therefore, we also provide the `setup.py` to build each extension:
```bash
# install all extension modules
bash scripts/install_ext.sh
# if you want to install manually, here is an example:
cd raymarching
python setup.py build_ext --inplace # build ext only, do not install (only can be used in the parent directory)
pip install . # install to python path (you still need the raymarching/ folder, since this only install the built extension.)
```

### Tested environments
* Ubuntu 20 with torch 1.10 & CUDA 11.3 on a TITAN RTX.
* Windows 10 with torch 1.11 & CUDA 11.3 on a RTX 3070.

Currently, `--ff` only supports GPUs with CUDA architecture `>= 70`.
For GPUs with lower architecture, `--tcnn` can still be used, but the speed will be slower compared to more recent GPUs.

# Usage

First time running will take some time to compile the CUDA extensions.

```bash
# text-guided generation
python main_nerf.py --text "cthulhu" --workspace trial --cuda_ray --fp16

# use the GUI
python main_nerf.py --text "cthulhu" --workspace trial --cuda_ray --fp16 --gui

# [experimental] image-guided generation (also use the CLIP loss)
python main_nerf.py --image /path/to/image --workspace trial --cuda_ray --fp16

```

check the `scripts` directory for more provided examples.


# Difference from the original implementation

* Mip-nerf is not implemented, currently only the original nerf is supported.
* Sampling poses with an elevation range in [-30, 30] degrees, instead of fixed at 30 degree.
* Use the origin loss.


# Update Logs
* 5.18: major update.
* 3.16: basic reproduction.


# Acknowledgement

* The great paper and official JAX implementation of [dreamfields](https://ajayj.com/dreamfields):
    ```
    @article{jain2021dreamfields,
        author = {Jain, Ajay and Mildenhall, Ben and Barron, Jonathan T. and Abbeel, Pieter and Poole, Ben},
        title = {Zero-Shot Text-Guided Object Generation with Dream Fields},
        journal = {arXiv},
        month = {December},
        year = {2021},
    }   
    ```

* The GUI is developed with [DearPyGui](https://github.com/hoffstadt/DearPyGui).
