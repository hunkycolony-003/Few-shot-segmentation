
## Some assumptions I have taken in implementing which might create problems:

1. In the current architecture a Resnet50 is used as the   spatial encoder and the last two layer (pooling and fc layer) are removed, which makes the spatial dimention of the feature maps 1/32 of the original image, with 2048 channels. 

   So consequently the output of the frequency encoder is also downsampled to the same spatial size of (1/32) of the original image. With 2048 channels. And image mask is also iterpolated to match the dimention. This leads to interpolating at the end to calculate the logits for the mask.

   We might need to find a walkaround to this, if it hampers the performance.

2. The current frequency encoder only takes the fft over each channel of the image, then concatenates the real and imaginary componants and then passed through two conv layers to match the dimensions of the spatial encoder output. I have not tried incorporating different kind of frequency encoders yet.

3. For teh prototype fusion, MHA is used between the spatial and frequency prototypes, where the Q = spatial_prototype, K,V = frequency_prototype. This is an arbirrary choice and different combinations may be tried


## Base Line results:

The baseline model run is given [here](notebooks/dual_panet_baseline.ipynb), run on BRIC dataset.

- The model was trained with eposodic batches of 10 support and and 1 query image.  
- There were 3 classes in the dataset. We have trained and evaluated independently across 3 folds, in each fold, one of the classed was left out as unseed class to evaluate on and a random episode was selected for each batch for traning from the rest of the two classes.
- The backbone (resnet50) was freezed for initial episodes and then was unfreezed for final episodes. 

- The results are given below for 3 folds:

  - Fold 1 Results -> mIoU: 0.0631, Dice: 0.1080 
  - Fold 2 Results -> mIoU: 0.1080, Dice: 0.1711 
  - Fold 3 Results -> mIoU: 0.0014, Dice: 0.0027