
## Some assumptions I have taken in implementing which might create problems:

1. In the current architecture a Resnet50 is used as the   spatial encoder and the last two layer (pooling and fc layer) are removed, which makes the spatial dimention of the feature maps 1/32 of the original image, with 2048 channels. 

   So consequently the output of the frequency encoder is also downsampled to the same spatial size of (1/32) of the original image. With 2048 channels. And image mask is also iterpolated to match the dimention. This leads to interpolating at the end to calculate the logits for the mask.

   We might need to find a walkaround to this, if it hampers the performance.

2. The current frequency encoder only takes the fft over each channel of the image, then concatenates the real and imaginary componants and then passed through two conv layers to match the dimensions of the spatial encoder output. I have not tried incorporating different kind of frequency encoders yet.

3. For teh prototype fusion, MHA is used between the spatial and frequency prototypes, where the Q = spatial_prototype, K,V = frequency_prototype. This is an arbirrary choice and different combinations may be tried
