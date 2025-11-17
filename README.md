# About

![CartoonX Example](./imgs/cartoonx_example.png)

This repository contains a python package for the saliency map method *CartoonX*, which is a core part of my  PhD at LMU Munich. CartoonX was initially introduced in the ECCV 2022 paper [**Cartoon Explanations of Image Classifiers**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136720439.pdf) and further improved in the CVPR 2023 paper [**Explaining Image Classifiers with Multiscale Directional Image Representation**](https://openaccess.thecvf.com/content/CVPR2023/papers/Kolek_Explaining_Image_Classifiers_With_Multiscale_Directional_Image_Representation_CVPR_2023_paper.pdf).

CartoonX was designed to extract the **relevant piece-wise smooth part** of an image. We leverage that piece-wise smooth images are **sparse in the wavelet domain**. CartoonX **learns a sparsity-driven mask on the wavelet coefficients** of the image to maximize a target class probability. Spatial energy may also be regularized to black-out unnecessary areas of the image. Wavelets can be replaced by other (directional) multiscale representation systems such as shearlets.


# Setup

Python 3.9.x is supported.

1. Clone the repository via
    ```
    https://github.com/skmda37/CartoonX.git
    ```

2. Navigate to the root of the repo
    ```
    cd CartoonX
    ```
3. Create a virtualenv in the root of the repo via
    ```
    python -m venv venv
    ```
4. Activate the virtualenv via
    ```
    source venv/bin/activate
    ```
1. Install dependecies and the project source code as a local package via
    ```
    pip install -e .
    ```
1. Install [pytorch version](https://pytorch.org/get-started/previous-versions/)  (2.7.1 or newer) that is compatible with your CUDA driver. For instance, if you have cuda 11.8 then you can install
    ```
    pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
    ```

# Content

* [src/cartoonx](src/cartoonx) contains the code for the CartoonX package
* [src/cartoonx/modelling/explainer.py](src/cartoonx/modelling/explainer.py) implements the main CartoonX modules that implement the optimization loop for CartoonX
* [src/cartoonx/pipeline](src/cartoonx/pipeline/factory.py) creates the CartoonX explainer
* [src/cartoonx/utils/torchutils.py](src/cartoonx/utils/torchutils.py) contains PyTorch utilities


# How to run

We have a notebook in [notebooks/example.ipynb](notebooks/example.ipynb) explaining how to run CartoonX with an example.


# Cite
If you use this code please cite

```bibtex
@inproceedings{kolek2022cartoon,
  title={Cartoon explanations of image classifiers},
  author={Kolek, Stefan and Nguyen, Duc Anh and Levie, Ron and Bruna, Joan and Kutyniok, Gitta},
  booktitle={European Conference on Computer Vision},
  pages={443--458},
  year={2022},
  organization={Springer}
}
```

```bibtex
@inproceedings{kolek2023explaining,
  title={Explaining image classifiers with multiscale directional image representation},
  author={Kolek, Stefan and Windesheim, Robert and Andrade-Loarca, Hector and Kutyniok, Gitta and Levie, Ron},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={18600--18609},
  year={2023}
}
```

# License
<div>
<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative Commons Attribution-NonCommercial 4.0 International License</a>.
</div>