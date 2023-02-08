# DiDeCoNN (Diffusion Deblurring with Convolutional Neural Networks)

## Requirements
DiDaCoNN requires Image Reconstruction technique such as Optimization algorithms and corruption Linear Operator, available in GitHub at:

```
git clone https://github.com/devangelista2/IPPy.git
```

## Dataset
The models are trained on a grey-scale version of CelebA faces, in the 64x64 cropped format. The dataset in .npy format we used is available on HuggingFace, and can be downloaded by:

```
git lfs install
git clone https://huggingface.co/datasets/TivoGatto/celeba_grayscale
```
