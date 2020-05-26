## Demosaicing
This is an implementation of non-Deep Learning demosaicing algorithm in OpenCV (C++)

## Implement Method
+ Smooth-hue Interpolation
+ Laplacian-corrected Linear Filter
+ Gradient-Based Threshold-Free (GBTF)
+ Residual interpolation (RI)

## Reference
+ Bayer 
    + Wiki:
        + https://en.wikipedia.org/wiki/Demosaicing
        + https://en.wikipedia.org/wiki/Bayer_filter

+ OpenCV
    + https://github.com/opencv/opencv/blob/master/modules/imgproc/src/demosaicing.cpp
        + Bilinear
        + Edge-Aware
        + Variable Number of Gradients
+ Matlab
    + High quality linear interpolation for demosaicing of Bayer-patterned color images (2004)
    + https://www.mathworks.com/help/images/ref/demosaic.html
    + https://www.ipol.im/pub/art/2011/g_mhcd/article.pdf (review)
+ Bilinear interpolation 
    + https://en.wikipedia.org/wiki/Bilinear_interpolation
+ Smooth-hue Interpolation
    + Demosaicking methods for Bayer color arrays (2002)
    + https://pdfs.semanticscholar.org/28c8/99ab34b6dd91d10474b5635eec6a97b8e3fa.pdf 
+ Laplacian-corrected Linear Filter
    + Same as above Matlab default demosaicing algorithm
+ Gradient-Based Threshold-Free (GBTF)
    + Gradient-Based Threshold-Free Color Filter Array Interpolation (2010)
    + https://ieeexplore.ieee.org/document/5654327
    + https://github.com/RayXie29/GBTF_Color_Interpolation (other's implementation)
+ Residual interpolation (RI)
    + Residual interpolation for color image demosaicking (2013)
    + http://www.ok.sc.e.titech.ac.jp/res/DM/RI.pdf
    + include Guided Image Filtering (2010)
    + http://kaiminghe.com/eccv10/
