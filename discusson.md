
## Some assumptions I have taken in implementing which might create problems:

1. In the current architecture a Resnet50 is used as the   spatial encoder and the last two layer (pooling and fc layer) are removed, which makes the spatial dimention of the feature maps 1/32 of the original image, with 2048 channels. 

   So consequently the output of the frequency encoder is also downsampled to the same spatial size of (1/32) of the original image. With 2048 channels. And image mask is also iterpolated to match the dimention. This leads to interpolating at the end to calculate the logits for the mask.

   We might need to find a walkaround to this, if it hampers the performance.

2. The current frequency encoder only takes the fft over each channel of the image, then concatenates the real and imaginary componants and then passed through two conv layers to match the dimensions of the spatial encoder output. I have not tried incorporating different kind of frequency encoders yet.

3. For teh prototype fusion, MHA is used between the spatial and frequency prototypes, where the Q = spatial_prototype, K,V = frequency_prototype. This is an arbirrary choice and different combinations may be tried


## Base Line:

The baseline model run is given [here](notebooks/dual_panet_baseline.ipynb), run on BRIC dataset. Kaggle notebook is [here](https://www.kaggle.com/code/soumyajitghosh1729/few-shot-seg).

- The model was trained with eposodic batches of 10 support and and 1 query image.  
- There were 3 classes in the dataset. We have trained and evaluated independently across 3 folds, in each fold, one of the classed was left out as unseed class to evaluate on and a random episode was selected for each batch for traning from the rest of the two classes.
- The backbone (resnet50) was freezed for initial episodes and then was unfreezed for final episodes. 

- The results are given below for 3 folds:

  - Fold 1 Results -> mIoU: 0.0631, Dice: 0.1080 
  - Fold 2 Results -> mIoU: 0.1080, Dice: 0.1711 
  - Fold 3 Results -> mIoU: 0.0014, Dice: 0.0027


## improving stride:

Notebook can be found [here](notebooks/feature%20s_stride_8.ipynb).
Kaggle run is given [here](https://www.kaggle.com/code/soumyajitghosh1729/few-shot-seg?scriptVersionId=332386701)

- The resnet architecture is modified so that the feature dimension is 1/8 of the original image instead of the previous shrink of 1/32, o that more spatial information of tha image is preserved.

- results for the 3 folds:

   - Fold 1 Results -> mIoU: 0.0982, Dice: 0.1717 
   - Fold 2 Results -> mIoU: 0.0503, Dice: 0.0929
   - Fold 3 Results -> mIoU: 0.0525, Dice: 0.0937

## DWT based frequency encoder:

Notebook can be found [here](notebooks/LeGall53DWT2D_encoder.ipynb). The kaggle run is [here](https://www.kaggle.com/code/soumyajitghosh1729/few-shot-seg?scriptVersionId=332499409)

- The frequency encoder, earlier fft, is replaced by [LeGall 5/3 Discrete Wavelet Transform](https://arxiv.org/pdf/2205.03898).

- results for the 3 folds:
   - Fold 1 Results -> mIoU: 0.0727, Dice: 0.1230
   - Fold 2 Results -> mIoU: 0.0851, Dice: 0.1271
   - Fold 3 Results -> mIoU: 0.1356, Dice: 0.2068

## Enabling PAR training:

Notebook is found [here](notebooks/par_training.ipynb)
. The kaggle run is given [here](https://www.kaggle.com/code/soumyajitghosh1729/few-shot-seg?scriptVersionId=332502554)

- A  Prototype alignment regularization (PAR) training method is used as per the [PANet paper](https://arxiv.org/pdf/1908.06391).

- Results for the 3 folds:
   - Fold 1 Results -> mIoU: 0.3388, Dice: 0.4259
   - Fold 2 Results -> mIoU: 0.3422, Dice: 0.4488
   - Fold 3 Results -> mIoU: 0.2497, Dice: 0.3398



