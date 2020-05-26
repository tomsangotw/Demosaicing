#pragma once

#include <opencv2/opencv.hpp>
#include <string>

// Bayer Patterm:
// 1:
//  RG
//  GB
// 2:
//  GR
//  BG
// 3:
//  BG
//  GR
// 4:
//  GB
//  RG

// Demosaicing Method:
//  Smooth_Hue          1
//  Laplacian_Corrected 2
//  GBTF                3
//  RI                  4

// assume input raw image is 10 bit image encoded in 16 bit format
void Demosaicing(std::string &BayerFileName, cv::Mat &Dst, int rows, int cols, int BayerPatternFlag=1, int DemosaicingMethod=4);
