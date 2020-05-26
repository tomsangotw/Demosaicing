#include <cstdio>
#include <iostream>
#include <fstream>
#include <cstring>
#include <vector>
#include <opencv2/opencv.hpp>
#include "Demosaicing.hpp"

using namespace std;
using namespace cv;

/* Type look Up
https://stackoverflow.com/questions/10167534/how-to-find-out-what-type-of-a-mat-object-is-with-mattype-in-opencv
+--------+----+----+----+----+------+------+------+------+
|        | C1 | C2 | C3 | C4 | C(5) | C(6) | C(7) | C(8) |
+--------+----+----+----+----+------+------+------+------+
| CV_8U  |  0 |  8 | 16 | 24 |   32 |   40 |   48 |   56 |
| CV_8S  |  1 |  9 | 17 | 25 |   33 |   41 |   49 |   57 |
| CV_16U |  2 | 10 | 18 | 26 |   34 |   42 |   50 |   58 |
| CV_16S |  3 | 11 | 19 | 27 |   35 |   43 |   51 |   59 |
| CV_32S |  4 | 12 | 20 | 28 |   36 |   44 |   52 |   60 |
| CV_32F |  5 | 13 | 21 | 29 |   37 |   45 |   53 |   61 |
| CV_64F |  6 | 14 | 22 | 30 |   38 |   46 |   54 |   62 |
+--------+----+----+----+----+------+------+------+------+
*/

// Split 1 channel image into 3 channels according to bayer pattern
//  R  G
//  G  B
// 
//  m*n*1 -> m*n*3
void bayer_split(cv::Mat &Bayer,cv::Mat &Dst){
	if(Bayer.channels() != 1){
		std::cerr << "bayer_split allow only 1 channel raw bayer image " << std::endl;
		return;
	}  
	Dst = cv::Mat::zeros(Bayer.rows, Bayer.cols, CV_8UC3); 
	//cout << "SIZE: " << Dst.size() << " Channels: " << Dst.channels() << endl;
	int channelNum;

	for(int row = 0; row < Bayer.rows; row++){
		for(int col = 0; col < Bayer.cols; col++){
			if(row % 2 == 0){ // opencv: BGR
				//even rows and even cols = R = channel:2
				//even rows and  odd cols = G = channel:1 
				channelNum = (col % 2 == 0) ? 2 : 1;
			}else{
				//odd rows and even cols = G = channel:1
				//odd rows and  odd cols = B = channel:0 
				channelNum = (col % 2 == 0) ? 1 : 0;
			}
			Dst.at<Vec3b>(row, col).val[channelNum] = Bayer.at<uchar>(row, col);
		}
	}
	return;
}
// using bayer_split to get bayer mask according to bayer pattern
//  R  G
//  G  B
// val = 1 denote existing
void bayer_mask(cv::Mat &Bayer,cv::Mat &Dst){
	Mat temp = cv::Mat::ones(Bayer.size(), CV_8U);
	bayer_split(temp, Dst);
}



// Downsampling to 3 Channel Bayer according to `RGGB` pattern
//  R  G
//  G  B
// 
void ConvertToThreeChannelBayerBG(Mat &BGRImage){
	if(BGRImage.channels() != 3){
		std::cerr << "ConvertToThreeChannelBayerBG allow only 3 channel bayer rgb image " << std::endl;
		return;
	}  
	Mat BayerImage(BGRImage.rows, BGRImage.cols, CV_8UC1);
	int channel;
	for (int row = 0; row < BGRImage.rows; row++){
		for (int col = 0; col < BGRImage.cols; col++){
			if (row % 2 == 0){
				//even columns and even rows = red 
				//even columns and odd rows = green 
				channel = (col % 2 == 0) ? 0 : 1;
			}else{
				//odd columns and even rows = green 
				//odd columns and odd rows = blue
				channel = (col % 2 == 0) ? 1 : 2;
			}
			for(int i = 0; i < 3; i++){
				if(channel == i){
					continue;
				}
				BGRImage.at<Vec3b>(row, col)[i] = 0;
			}
		}
	}
}

// Downsampling to 1 Channel Bayer according to `RGGB` pattern
//  R  G
//  G  B
// 
void toSingleChannel(cv::Mat &src,cv::Mat &dst){ //checked
	if(src.channels() != 3){
		std::cerr << "to_SingleChannel need 3 channel image" << std::endl;
		return;
	}  
	/*
	dst = cv::Mat::zeros(src.rows, src.cols, CV_8UC1); 
	int channelNum;
	for(int row = 0; row < src.rows; row++){
		for(int col = 0; col < src.cols; col++){
			if(row % 2 == 0){ // opencv: BGR
				//even rows and even cols = R = channel:2
				//even rows and  odd cols = G = channel:1 
				channelNum = (col % 2 == 0) ? 2 : 1;
			}else{
				//odd rows and even cols = G = channel:1
				//odd rows and  odd cols = B = channel:0 
				channelNum = (col % 2 == 0) ? 1 : 0;
			}
			dst.at<uchar>(row, col) = src.at<Vec3b>(row, col).val[channelNum];
		}
	}*/

	// faster way
	// https://answers.opencv.org/question/3120/how-to-sum-a-3-channel-matrix-to-a-one-channel-matrix/
	//https://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html?highlight=transform#transform
	transform(src, dst, cv::Matx13f(1,1,1));
	return;
}




/* ==================== Smooth-hue Interpolation ====================
#       https://patents.google.com/patent/US4642678A/en
#       https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=1207407
#		http://www.sfu.ca/~gchapman/e895/e895l11.pdf
#
# The algorithm
#   - interpolate `G`
#   - compute hue for `R`,`B` channels at subsampled locations
#   - interpolate hue for all pixels in `R`,`B` channels
#   - determine chrominance `R`,`B` from hue 
*/
void demosaic_smooth_hue(cv::Mat &Bayer,cv::Mat &Dst){
	cv::Mat Src = Bayer.clone();

	if(Bayer.channels() == 1){ //input 1 channel -> 3 channel Bayer
		bayer_split(Bayer, Src);
	}

	//kernels
	Mat K_G =( Mat_<float>(3,3) << 0, 1, 0, 1, 4, 1, 0, 1, 0); 
	K_G *= (1.0/4.0);
	Mat K_B = (Mat_<float>(3,3) << 1, 2, 1, 2, 4, 2, 1, 2, 1);
	K_B *= (1.0/4.0);
	Mat K_R = K_B;

	//split channel to BGR 
	Src.convertTo(Src, CV_32F, 1.0 / 255.0); //to float
	vector<Mat> bgr(3);   
	split(Src, bgr);

	// interpolate luminance G
	filter2D(bgr[1], bgr[1], CV_32F, K_G);

	// compute hue (B-G), (R-G)
	// R G
	// G B
	for(int row = 0; row < bgr[1].rows; row++){
		int col = 0;
		if(row % 2 == 1){
			col = 1;
		}
		for(; col < bgr[1].cols; col += 2){
			if(row % 2 == 0){ //red
				bgr[2].at<float>(row, col) -= bgr[1].at<float>(row, col);
			}else{ //blue
				bgr[0].at<float>(row, col) -= bgr[1].at<float>(row, col);
			}
		}
	}

	// interpolate hue
	filter2D(bgr[0], bgr[0], CV_32F, K_B);
	filter2D(bgr[2], bgr[2], CV_32F, K_R);

	// Compute chrominance B,R
	bgr[0] = bgr[0] + bgr[1];
	bgr[2] = bgr[2] + bgr[1];

	merge(bgr, Dst);
	Dst.convertTo(Dst, CV_8U, 255.0);
	return;
}



/* ==================== Laplacian-corrected linear filter (MATLAB's demosaic) ====================
#       https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=1326587
#       https://www.ipol.im/pub/art/2011/g_mhcd/article.pdf
*/
void demosaic_laplacian_corrected(cv::Mat &Bayer,cv::Mat &Dst, float alpha = 1.0/2, float beta = 5.0/8, float gamma = 3.0/4){
	cv::Mat Src = Bayer.clone();

	if(Bayer.channels() == 1){ //input 1 channel -> 3 channel Bayer
		bayer_split(Bayer, Src);
	}

	// kernels
	Mat K_L = ( Mat_<float>(5,5) << 
		0, 0, -1, 0,  0, 
		0, 0,  0, 0,  0, 
		-1, 0,  4, 0, -1,
		0, 0,  0, 0,  0,
		0, 0, -1, 0,  0); 
	K_L *= (1.0/4.0);
	Mat K_G =( Mat_<float>(3,3) << 0, 1, 0, 1, 4, 1, 0, 1, 0); 
	K_G *= (1.0/4.0);
	Mat K_B = (Mat_<float>(3,3) << 1, 2, 1, 2, 4, 2, 1, 2, 1);
	K_B *= (1.0/4.0);
	Mat K_R = K_B;

	// split channel to BGR 
	Src.convertTo(Src, CV_32F, 1.0 / 255.0); //to float
	vector<Mat> bgr(3);
	vector<Mat> laplacian(3);
	split(Src, bgr);
	split(Src, laplacian);

	// interpolate luminance R, G, B
	filter2D(bgr[0], bgr[0], CV_32F, K_B);
	filter2D(bgr[1], bgr[1], CV_32F, K_G);
	filter2D(bgr[2], bgr[2], CV_32F, K_R);

	// Compute discrete laplacian in 5x5 neighborhood
	filter2D(laplacian[0], laplacian[0], CV_32F, K_L);
	filter2D(laplacian[1], laplacian[1], CV_32F, K_L);
	filter2D(laplacian[2], laplacian[2], CV_32F, K_L);

	// Laplacian correction
	// R G
	// G B
	for(int row = 0; row < Bayer.rows; row++){
		for(int col = 0; col < Bayer.cols; col++){
			if(row % 2 == 0 && col % 2 == 0){ //Red
				//Blue @ Red
				bgr[0].at<float>(row, col) += gamma * laplacian[2].at<float>(row, col);
				//Green @ Red
				bgr[1].at<float>(row, col) += alpha * laplacian[2].at<float>(row, col);
			}else if(row % 2 == 1 && col % 2 == 1){ //Blue
				//Green @ Blue
				bgr[1].at<float>(row, col) += alpha * laplacian[0].at<float>(row, col);
				//Red @ Blue
				bgr[2].at<float>(row, col) += beta * laplacian[0].at<float>(row, col);
			}else{
				//Red @ Green
				bgr[2].at<float>(row, col) += beta * laplacian[1].at<float>(row, col);
				//Blue @ Green
				bgr[0].at<float>(row, col) += gamma * laplacian[1].at<float>(row, col);
			}
		}
	}

	merge(bgr, Dst);
	Dst.convertTo(Dst, CV_8U, 255.0);
	return;
}



/* ==================== Gradient based threshold free ====================
#       https://ieeexplore.ieee.org/document/5654327 (2010)
*/
void demosaic_GBTF(cv::Mat &Bayer,cv::Mat &Dst){
	cv::Mat Src = Bayer.clone();
	if(Bayer.channels() == 1){ //input 1 channel -> 3 channel Bayer
		bayer_split(Bayer, Src);
	}

	// split channel to BGR 
	Src.convertTo(Src, CV_32F, 1.0 / 255.0); //to float
	vector<Mat> bgr(3);
	vector<Mat> finalBGR(3);
	split(Src, bgr);
	split(Src, finalBGR);

	// 2.1. Green Channel Interpolation
	// Hamilton and Adams’ interpolation for B', G', R'
	float G_H, G_V;
	float R_H, R_V;
	float B_H, B_V;
	// for interpolation purpose
	copyMakeBorder(bgr[0], bgr[0], 2, 2, 2, 2, cv::BORDER_DEFAULT);
	copyMakeBorder(bgr[1], bgr[1], 2, 2, 2, 2, cv::BORDER_DEFAULT);
	copyMakeBorder(bgr[2], bgr[2], 2, 2, 2, 2, cv::BORDER_DEFAULT);
	// horizontal and vertical color difference
	Mat V_Diff(Src.size(), CV_32F, cv::Scalar(0));
	Mat H_Diff(Src.size(), CV_32F, cv::Scalar(0));
	// [Bayer]		[Horizontal]			[Vertical]
	// R G R G		D_gr D_gr D_gr D_gr		D_gr D_gb D_gr D_gb
	// G B G B  ->	D_gb D_gb D_gb D_gb  &	D_gr D_gb D_gr D_gb
	// R G R G		D_gr D_gr D_gr D_gr		D_gr D_gb D_gr D_gb
	// G B G B		D_gb D_gb D_gb D_gb		D_gr D_gb D_gr D_gb
	for(int row = 0; row < Src.rows; row++){
		for(int col = 0; col < Src.cols; col++){
			int i = row + 2;
			int j = col + 2;
			if(row % 2 == 0 && col % 2 == 0){ //Red
				G_H = (bgr[1].at<float>(i, j - 1) + bgr[1].at<float>(i, j + 1)) * 0.5
					+ (2 * bgr[2].at<float>(i, j) - bgr[2].at<float>(i, j - 2) - bgr[2].at<float>(i, j + 2)) * 0.25;
				G_V = (bgr[1].at<float>(i - 1, j) + bgr[1].at<float>(i + 1, j)) * 0.5
					+ (2 * bgr[2].at<float>(i, j) - bgr[2].at<float>(i - 2, j) - bgr[2].at<float>(i + 2, j)) * 0.25;
				
				// cal G' - R difference
				V_Diff.at<float>(row, col) = G_V - bgr[2].at<float>(i, j);
				H_Diff.at<float>(row, col) = G_H - bgr[2].at<float>(i, j);
			}else if(row % 2 == 1 && col % 2 == 1){ //Blue
				G_H = (bgr[1].at<float>(i, j - 1) + bgr[1].at<float>(i, j + 1)) * 0.5
					+ (2 * bgr[0].at<float>(i, j) - bgr[0].at<float>(i, j - 2) - bgr[0].at<float>(i, j + 2)) * 0.25;
				G_V = (bgr[1].at<float>(i - 1, j) + bgr[1].at<float>(i + 1, j)) * 0.5
					+ (2 * bgr[0].at<float>(i, j) - bgr[0].at<float>(i - 2, j) - bgr[0].at<float>(i + 2, j)) * 0.25;
				
				// cal G' - B difference
				V_Diff.at<float>(row, col) = G_V - bgr[0].at<float>(i, j);
				H_Diff.at<float>(row, col) = G_H - bgr[0].at<float>(i, j);
			}else if(row % 2 == 1 && col % 2 == 0){ //Green with Red Vertical / Blue Horizontal
				R_V = (bgr[2].at<float>(i - 1, j) + bgr[2].at<float>(i + 1, j)) * 0.5
					+ (2 * bgr[1].at<float>(i, j) - bgr[1].at<float>(i - 2, j) - bgr[1].at<float>(i + 2, j)) * 0.25;
				B_H = (bgr[0].at<float>(i, j - 1) + bgr[0].at<float>(i, j + 1)) * 0.5
					+ (2 * bgr[1].at<float>(i, j) - bgr[1].at<float>(i, j - 2) - bgr[1].at<float>(i, j + 2)) * 0.25;

				// cal G - R', G - B' difference
				V_Diff.at<float>(row, col) = bgr[1].at<float>(i, j) - R_V;
				H_Diff.at<float>(row, col) = bgr[1].at<float>(i, j) - B_H;
			}else{ //Green with Red Horizontal / Blue Vertical
				R_H = (bgr[2].at<float>(i, j - 1) + bgr[2].at<float>(i, j + 1)) * 0.5
					+ (2 * bgr[1].at<float>(i, j) - bgr[1].at<float>(i, j - 2) - bgr[1].at<float>(i, j + 2)) * 0.25;
				B_V = (bgr[0].at<float>(i - 1, j) + bgr[0].at<float>(i + 1, j)) * 0.5
					+ (2 * bgr[1].at<float>(i, j) - bgr[1].at<float>(i - 2, j) - bgr[1].at<float>(i + 2, j)) * 0.25;

				// cal G - R', G - B' difference
				H_Diff.at<float>(row, col) = bgr[1].at<float>(i, j) - R_H;
				V_Diff.at<float>(row, col) = bgr[1].at<float>(i, j) - B_V;
			}
		}
	}

	// Final difference estimation for the target pixel
	copyMakeBorder(V_Diff, V_Diff, 4, 4, 4, 4, cv::BORDER_DEFAULT); // now V_Diff need shift row and col by 4
	copyMakeBorder(H_Diff, H_Diff, 4, 4, 4, 4, cv::BORDER_DEFAULT);
	Mat gr_Diff(Src.size(), CV_32F, cv::Scalar(0));
	Mat gb_Diff(Src.size(), CV_32F, cv::Scalar(0));
	// use gradients of color differences to come up with weights for each direction.
	Mat V_diff_gradient;
	Mat H_diff_gradient;
	float VHkernel[3] = {-1,0,1}; //(central difference form)
	cv::Mat HK(1, 3, CV_32F, VHkernel);
	cv::Mat VK(3, 1, CV_32F, VHkernel);
	cv::filter2D(V_Diff, V_diff_gradient, -1, VK);
	cv::filter2D(H_Diff, H_diff_gradient, -1, HK);
	V_diff_gradient = cv::abs(V_diff_gradient); // V_diff_gradient need shift row and col by 4
	H_diff_gradient = cv::abs(H_diff_gradient);

	float Weight[4]; //four direction, N, S, W, E
	int startPoint[4][2] = {{-4, -2}, {0, -2}, {-2, -4}, {-2, 0}}; //weight startpoint
	float W_total;
	int a, b;
	for(int row = 0; row < Src.rows; row++){
		int col = 0;
		if(row % 2 == 1){
			col = 1;
		}
		for(; col < Src.cols; col += 2){
			// calculate Weight
			Weight[0] = Weight[1] = Weight[2] = Weight[3] = W_total = 0.0;
			for(int dir = 0; dir < 4; dir++){// N, S, W, E
				a = startPoint[ dir ][0];
				b = startPoint[ dir ][1];
				for(int i = 0; i < 5; i++){
					for(int j = 0; j < 5; j++){
						if(dir < 2){ // N, S -> Vertical
							Weight[ dir ] += V_diff_gradient.at<float>(4 + row + a + i, 4 + col + b + j); // shift 4 due to copyMakeBorder
						}else{ // W, E -> Horizontal
							Weight[ dir ] += H_diff_gradient.at<float>(4 + row + a + i, 4 + col + b + j);
						}
					}
				}
				Weight[ dir ] *= Weight[ dir ] ;
				Weight[ dir ] = 1.0/Weight[ dir ];
				W_total += Weight[ dir ];
			}

			// calculate gr_Diff/gb_Diff & finalGreen 
			int i = row + 4;
			int j = col + 4;
			if(row % 2 == 0){ //Green @ Red
				gr_Diff.at<float>(row, col) =
					( Weight[0] * 0.2 * (V_Diff.at<float>(i - 4, j) + V_Diff.at<float>(i - 3, j) + V_Diff.at<float>(i - 2, j) + V_Diff.at<float>(i - 1, j) + V_Diff.at<float>(i, j))
					+ Weight[1] * 0.2 * (V_Diff.at<float>(i, j) + V_Diff.at<float>(i + 1, j) + V_Diff.at<float>(i + 2, j) + V_Diff.at<float>(i + 3, j) + V_Diff.at<float>(i + 4, j))
					+ Weight[2] * 0.2 * (H_Diff.at<float>(i, j - 4) + H_Diff.at<float>(i, j - 3) + H_Diff.at<float>(i, j - 2) + H_Diff.at<float>(i, j - 1) + H_Diff.at<float>(i, j))
					+ Weight[3] * 0.2 * (H_Diff.at<float>(i, j) + H_Diff.at<float>(i, j + 1) + H_Diff.at<float>(i, j + 2) + H_Diff.at<float>(i, j + 3) + H_Diff.at<float>(i, j + 4))
					) / W_total;

				finalBGR[1].at<float>(row, col) = finalBGR[2].at<float>(row, col) + gr_Diff.at<float>(row, col); // R + gb_Diff
			}else{ //Green @ Blue
				gb_Diff.at<float>(row, col) =
					( Weight[0] * 0.2 * (V_Diff.at<float>(i - 4, j) + V_Diff.at<float>(i - 3, j) + V_Diff.at<float>(i - 2, j) + V_Diff.at<float>(i - 1, j) + V_Diff.at<float>(i, j))
					+ Weight[1] * 0.2 * (V_Diff.at<float>(i, j) + V_Diff.at<float>(i + 1, j) + V_Diff.at<float>(i + 2, j) + V_Diff.at<float>(i + 3, j) + V_Diff.at<float>(i + 4, j))
					+ Weight[2] * 0.2 * (H_Diff.at<float>(i, j - 4) + H_Diff.at<float>(i, j - 3) + H_Diff.at<float>(i, j - 2) + H_Diff.at<float>(i, j - 1) + H_Diff.at<float>(i, j))
					+ Weight[3] * 0.2 * (H_Diff.at<float>(i, j) + H_Diff.at<float>(i, j + 1) + H_Diff.at<float>(i, j + 2) + H_Diff.at<float>(i, j + 3) + H_Diff.at<float>(i, j + 4))
					) / W_total;

				finalBGR[1].at<float>(row, col) = finalBGR[0].at<float>(row, col) + gb_Diff.at<float>(row, col); // B + gb_Diff
			}
		}
	}

	// 2.2. Red and Blue Channel Interpolation
	float PrbData[49] = {
		0, 0, -0.03125, 0, -0.03125, 0, 0,
		0,0,0,0,0,0,0,
		-0.03125,0,0.3125,0,0.3125,0,-0.03125,
		0,0,0,0,0,0,0,
		-0.03125,0,0.3125,0,0.3125,0,-0.03125,
		0,0,0,0,0,0,0,
		0,0,-0.03125,0,-0.03125,0,0
	};
	cv::Mat Prb(7, 7, CV_32FC1, PrbData);

	copyMakeBorder(gr_Diff, gr_Diff, 3, 3, 3, 3, cv::BORDER_DEFAULT);
	copyMakeBorder(gb_Diff, gb_Diff, 3, 3, 3, 3, cv::BORDER_DEFAULT);
	// Red pixel values at blue locations and blue pixel values at redlocations
	// R G
	// G B
	for(int row = 0; row < Bayer.rows; row++){
		int col = 0;
		if(row % 2 == 1){
			col = 1;
		}
		//https://stackoverflow.com/questions/21874774/sum-of-elements-in-a-matrix-in-opencv
		for(; col < Bayer.cols; col += 2){
			if(row % 2 == 0){ //Red
				//Blue @ Red
				finalBGR[0].at<float>(row, col) = finalBGR[1].at<float>(row, col) - cv::sum( gb_Diff(cv::Range(3 + row - 3, 3 + row + 3 + 1), cv::Range( 3 + col - 3, 3 + col + 3 + 1)).mul(Prb) )[0];
			}else{ //Blue
				//Red @ Blue
				finalBGR[2].at<float>(row, col) = finalBGR[1].at<float>(row, col) - cv::sum( gr_Diff(cv::Range(3 + row - 3, 3 + row + 3 + 1), cv::Range( 3 + col - 3, 3 + col + 3 + 1)).mul(Prb) )[0];
			}
		}
	}

	// For red and blue pixels at green locations, we use bilinearinterpolation over the closest four neighbors
	// R G
	// G B
	copyMakeBorder(finalBGR[0], bgr[0], 1, 1, 1, 1, cv::BORDER_DEFAULT);
	copyMakeBorder(finalBGR[1], bgr[1], 1, 1, 1, 1, cv::BORDER_DEFAULT);
	copyMakeBorder(finalBGR[2], bgr[2], 1, 1, 1, 1, cv::BORDER_DEFAULT);
	for(int row = 0; row < Src.rows; row++){
		int col = 0;
		if(row % 2 == 0){
			col = 1;
		}
		for(; col < Src.cols; col += 2){ //Green
			int i = row + 1;
			int j = col + 1;
			// Red
			finalBGR[2].at<float>(row, col) = finalBGR[1].at<float>(row, col) 
				- (bgr[1].at<float>(i - 1, j) - bgr[2].at<float>(i - 1, j)) / 4.0
				- (bgr[1].at<float>(i + 1, j) - bgr[2].at<float>(i + 1, j)) / 4.0
				- (bgr[1].at<float>(i, j - 1) - bgr[2].at<float>(i, j - 1)) / 4.0
				- (bgr[1].at<float>(i, j + 1) - bgr[2].at<float>(i, j + 1)) / 4.0;
			// Blue
			finalBGR[0].at<float>(row, col) = finalBGR[1].at<float>(row, col)
				- (bgr[1].at<float>(i - 1, j) - bgr[0].at<float>(i - 1, j)) / 4.0
				- (bgr[1].at<float>(i + 1, j) - bgr[0].at<float>(i + 1, j)) / 4.0
				- (bgr[1].at<float>(i, j - 1) - bgr[0].at<float>(i, j - 1)) / 4.0
				- (bgr[1].at<float>(i, j + 1) - bgr[0].at<float>(i, j + 1)) / 4.0;
		}
	}

	merge(finalBGR, Dst);
	Dst.convertTo(Dst, CV_8U, 255.0);
	return;
}


// =============== Guided Filter ===================
// https://medium.com/@gary1346aa/%E5%B0%8E%E5%90%91%E6%BF%BE%E6%B3%A2%E7%9A%84%E5%8E%9F%E7%90%86%E4%BB%A5%E5%8F%8A%E5%85%B6%E6%87%89%E7%94%A8-78fdf562e749
// https://github.com/atilimcetin/guided-filter
// I: imput
// r: (radius)
Mat box_filter(const Mat &I, int r){
	Mat result;
	// = call boxFiltr (all average)
	// boxFilter(src, dst, src.type(), anchor, true, borderType).
	blur(I, result, Size(2 * r + 1, 2 * r + 1)); //create Mat data by cv function so can return (not user allocated data)
	return result;
}
// p: origin img (assume CV_8U or 0~255)
// I: guided img
// r: local window radius
// eps: regularization parameter (0.1)^2, (0.2)^2...
Mat guided_filter(const Mat &originP, const Mat &originI, int r = 2, float eps=0.0){
	Mat p, I;
	originP.convertTo(p, CV_32F, 1.0 / 255.0); //(a * (i,j) + b)
	originI.convertTo(I, CV_32F, 1.0 / 255.0);

	// step: 1
	Mat mean_I = box_filter(I, r);
	Mat mean_p = box_filter(p, r);
	Mat corr_I = box_filter(I.mul(I), r); //mul: element wise mul
	Mat corr_Ip = box_filter(I.mul(p), r);
	// step: 2
	Mat var_I = corr_I - mean_I.mul(mean_I);
	Mat cov_Ip = corr_Ip - mean_I.mul(mean_p);
	// step: 3
	Mat a;
	if (var_I.channels() == 3){
		a = cov_Ip / (var_I + Scalar(eps, eps, eps)); //otherwise only 1 channel get added
	}else{
		a = cov_Ip / (var_I + eps);
	}
	Mat b = mean_p - a.mul(mean_I);

	// step: 4
	Mat mean_a = box_filter(a, r);
	Mat mean_b = box_filter(b, r);
	// step: 5
	Mat q = mean_a.mul(I) + mean_b;
	Mat res;

	q.convertTo(q, CV_8U, 255.0);
	return q;
}



// =============== Guided Filter Modified ===================
// I: imput
// h, v: local window radius
Mat box_filter_modified(const Mat &I, int h, int v){
	Mat result;
	// = call boxFiltr (all average)
	// boxFilter(src, dst, src.type(), anchor, true, borderType).
	blur(I, result, Size(2 * h + 1, 2 * v + 1)); //width * height
	return result * (2 * h + 1) * (2 * v + 1);
}
// p: origin img (assume float or CV_32F, 0.0 ~ 1.0)
// I: guided img
// M: binary mask
// h, v: local window radius
// eps: regularization parameter (0.1)^2, (0.2)^2...
Mat guided_filter_modified(const Mat &originP, const Mat &originI,  const Mat &M, int h = 2, int v = 2, float eps=0.0){
	Mat p, I;
	p = originP.clone();
	I = originI.clone();

	// seems need mul(M) for all I if I will not be zero outside mask

	//The number of the sammpled pixels in each local patch
	Mat N = box_filter_modified(M, h, v);
	Mat temp = (N == 0);
	temp.convertTo(temp, CV_32F, 1.0/255.0);
	N = N + temp;
	// The size of each local patch; N=(2h+1)*(2v+1) except for boundary pixels.
	Mat N2 = box_filter_modified(Mat::ones(I.rows, I.cols, CV_32F), h, v);

	// step: 1
	Mat mean_I = box_filter_modified(I.mul(M), h, v);
	divide(mean_I, N, mean_I);
	Mat mean_p = box_filter_modified(p, h, v);
	divide(mean_p, N, mean_p);
	Mat corr_I = box_filter_modified(I.mul(I).mul(M), h, v); //mul: element wise mul
	divide(corr_I, N, corr_I);
	Mat corr_Ip = box_filter_modified(I.mul(p), h, v);
	divide(corr_Ip, N, corr_Ip);

	// step: 2
	Mat var_I = corr_I - mean_I.mul(mean_I);
	//threshold parameter
	float th = 0.00001;
	for(int row = 0; row < var_I.rows; row++){
		float *p = var_I.ptr<float>(row); 
		for(int col = 0; col < var_I.cols; col++){
			if(p[col] < th){ 
				p[col] = th;
			}
		}
	}

	Mat cov_Ip = corr_Ip - mean_I.mul(mean_p);
	// step: 3
	Mat a;
	if (var_I.channels() == 3){
		a = cov_Ip / (var_I + Scalar(eps, eps, eps)); //otherwise only 1 channel get added
	}else{
		a = cov_Ip / (var_I + eps);
	}
	Mat b = mean_p - a.mul(mean_I);
	// step: 4
	Mat mean_a = box_filter_modified(a, h, v);
	divide(mean_a, N2, mean_a);
	Mat mean_b = box_filter_modified(b, h, v);
	divide(mean_b, N2, mean_b);
	// step: 5
	Mat q = mean_a.mul(I) + mean_b;
	Mat res;

	//q.convertTo(q, CV_8U, 255.0);
	return q;
}


// =============== Residual Interpolation ===================
//	http://www.ok.sc.e.titech.ac.jp/res/DM/RI.pdf (2013)
//	sigma: standard deviation of gaussian filter
void demosaic_residual(cv::Mat &Bayer,cv::Mat &Dst, float sigma = 1.0){
	cv::Mat Src = Bayer.clone();
	if(Bayer.channels() == 1){ //input 1 channel -> 3 channel Bayer
		bayer_split(Bayer, Src);
	}

	Mat Src1ch;
	toSingleChannel(Src, Src1ch); //cv8U
	Src1ch.convertTo(Src1ch, CV_32F, 1.0 / 255.0); //normalize

	Mat tempMask;
	bayer_mask(Src, tempMask);
	tempMask.convertTo(tempMask, CV_32F); //normalize
	vector<Mat> mask(3);
	split(tempMask, mask);
    tempMask.release();

	// split channel to BGR 
	Src.convertTo(Src, CV_32F, 1.0 / 255.0); //normalize
	vector<Mat> bgr(3);
	vector<Mat> finalBGR(3);
	split(Src, bgr);
	split(Src, finalBGR);

	// ==== 1.Green interpolation ===
	// get green mask, care it depend on bayer type
	// R G -> 0 1 -> 0 0
	// G B    0 0    1 0
	// see horizontal: Gr or Gb
	Mat maskGr = Mat::zeros(Src.rows, Src.cols, CV_32F); 
	Mat maskGb = Mat::zeros(Src.rows, Src.cols, CV_32F); 
	for(int row = 0; row < Src.rows; row++){
		int col = 0;
		float *targetGr = maskGr.ptr<float>(row);
		float *targetGb = maskGb.ptr<float>(row);
		if(row % 2 == 0){
			col = 1;
		}
		for(; col < Src.cols; col += 2){
			if(row % 2 == 0){ // R G
				targetGr[col] = 1.0;
			}else{ //G B
				targetGb[col] = 1.0;
			}
		}
	}

	float VHkernel[3] = {0.5, 0, 0.5}; //bilinear interpolation at 1D
	cv::Mat HK(1, 3, CV_32F, VHkernel);
	cv::Mat VK(3, 1, CV_32F, VHkernel);
	Mat rawH, rawV;
	filter2D(Src1ch, rawH, -1, HK); //original matlab all using 'replicate' for filter
	filter2D(Src1ch, rawV, -1, VK);

	// Guide image
	Mat GuideG_H = bgr[1] + rawH.mul(mask[2]) + rawH.mul(mask[0]);//Gh @ R: 1/2(Gh left + Gh right), Gh @ B: 1/2(Gh left + Gh right)
	Mat GuideR_H = bgr[2] + rawH.mul(maskGr); //R pixel + r nearby rawH(= red bilinear part)
	Mat GuideB_H = bgr[0] + rawH.mul(maskGb); //B pixel + b nearby rawH(= blue bilinear part)
	Mat GuideG_V = bgr[1] + rawV.mul(mask[2]) + rawV.mul(mask[0]);//bilinear interpolation
	Mat GuideR_V = bgr[2] + rawV.mul(maskGb); //vertical need change Gr <-> Gb
	Mat GuideB_V = bgr[0] + rawV.mul(maskGr);

	// Tentative image
	int h = 5; //horizontal
	int v = 0; //vertical
	float eps = 0;
	Mat tentativeR_H = guided_filter_modified(bgr[2], GuideG_H, mask[2], h, v, eps);
	Mat tentativeGr_H = guided_filter_modified(bgr[1].mul(maskGr), GuideR_H, maskGr, h, v, eps);// need mul(mask) because Green has two location
	Mat tentativeGb_H = guided_filter_modified(bgr[1].mul(maskGb), GuideB_H, maskGb, h, v, eps);
	Mat tentativeB_H = guided_filter_modified(bgr[0], GuideG_H, mask[0], h, v, eps);
	// vertical part
	Mat tentativeR_V = guided_filter_modified(bgr[2], GuideG_V, mask[2], v, h, eps);
	Mat tentativeGr_V = guided_filter_modified(bgr[1].mul(maskGb), GuideR_V, maskGb, v, h, eps); //Gr <-> Gb
	Mat tentativeGb_V = guided_filter_modified(bgr[1].mul(maskGr), GuideB_V, maskGr, v, h, eps);
	Mat tentativeB_V = guided_filter_modified(bgr[0], GuideG_V, mask[0], v, h, eps);

    //relsease Guided Image
	GuideG_H.release();
	GuideR_H.release();
	GuideB_H.release();
	GuideG_V.release();
	GuideR_V.release();
	GuideB_V.release();

	// Residual
	Mat residualGr_H = (bgr[1] - tentativeGr_H).mul(maskGr);
	Mat residualGb_H = (bgr[1] - tentativeGb_H).mul(maskGb);
	Mat residualR_H =  (bgr[2] - tentativeR_H).mul(mask[2]);
	Mat residualB_H =  (bgr[0] - tentativeB_H).mul(mask[0]);
	Mat residualGr_V = (bgr[1] - tentativeGr_V).mul(maskGb); 
	Mat residualGb_V = (bgr[1] - tentativeGb_V).mul(maskGr);
	Mat residualR_V =  (bgr[2] - tentativeR_V).mul(mask[2]);
	Mat residualB_V =  (bgr[0] - tentativeB_V).mul(mask[0]);

	// Residual interpolation
	filter2D(residualGr_H, residualGr_H, -1, HK); //original matlab using 'replicate'
	filter2D(residualGb_H, residualGb_H, -1, HK);
	filter2D(residualR_H, residualR_H, -1, HK);
	filter2D(residualB_H, residualB_H, -1, HK);
	// verical part
	filter2D(residualGr_V, residualGr_V, -1, VK);
	filter2D(residualGb_V, residualGb_V, -1, VK);
	filter2D(residualR_V, residualR_V, -1, VK);
	filter2D(residualB_V, residualB_V, -1, VK);

	// Add tentative image
	Mat Gr_H = ( tentativeGr_H + residualGr_H ).mul(mask[2]);
	Mat Gb_H = ( tentativeGb_H + residualGb_H ).mul(mask[0]);
	Mat R_H = ( tentativeR_H + residualR_H ).mul(maskGr);
	Mat B_H = ( tentativeB_H + residualB_H ).mul(maskGb);
	Mat Gr_V = ( tentativeGr_V + residualGr_V ).mul(mask[2]);
	Mat Gb_V = ( tentativeGb_V + residualGb_V ).mul(mask[0]);
	Mat R_V = ( tentativeR_V + residualR_V ).mul(maskGb);
	Mat B_V = ( tentativeB_V + residualB_V ).mul(maskGr);

	// Vertical and horizontal color difference 
	Mat dif_H = bgr[1] + Gr_H + Gb_H - bgr[2] - bgr[0] - R_H - B_H;
	Mat dif_V = bgr[1] + Gr_V + Gb_V - bgr[2] - bgr[0] - R_V - B_V;

    //release not used Mat for memory useage
    tentativeR_H.release();
	tentativeGr_H.release();
	tentativeGb_H.release();
	tentativeB_H.release();
	tentativeR_V.release();
	tentativeGr_V.release();
	tentativeGb_V.release();
	tentativeB_V.release();

    residualGr_H.release();
	residualGb_H.release();
	residualR_H.release();
	residualB_H.release();
	residualGr_V.release(); 
	residualGb_V.release();
	residualR_V.release();
	residualB_V.release();

    Gr_H.release();
	Gb_H.release();
	R_H.release(); 
	B_H.release(); 
	Gr_V.release(); 
	Gb_V.release(); 
	R_V.release(); 
	B_V.release(); 

	// Combine Vertical and Horizontal Color Differences
	// color difference gradient
	float difkernel[3] = {1, 0, -1}; 
	Mat dif_H_K(1, 3, CV_32F, difkernel);
	Mat dif_V_K(3, 1, CV_32F, difkernel);
	Mat V_diff_gradient, H_diff_gradient;
	filter2D(dif_H, H_diff_gradient, -1, dif_H_K);
	filter2D(dif_V, V_diff_gradient, -1, dif_V_K);
	//filter2D(dif_H, H_diff_gradient, -1, dif_H_K, Point(-1,-1), 0.0, BORDER_REPLICATE );
	//filter2D(dif_V, V_diff_gradient, -1, dif_V_K, Point(-1,-1), 0.0, BORDER_REPLICATE );
	H_diff_gradient = cv::abs(H_diff_gradient);
	V_diff_gradient = cv::abs(V_diff_gradient);

	// Directional weight (four direction)
	Mat K = Mat::ones(5, 5, CV_32F);
	Mat hWeightSum, vWeightSum; 
	filter2D(H_diff_gradient, hWeightSum, -1, K); //Add up 5x5 weight patch first
	filter2D(V_diff_gradient, vWeightSum, -1, K);
	Mat Wkernel = (Mat_<float>(1, 5) << 1, 0, 0, 0, 0); //shift kernal
	Mat Ekernel = (Mat_<float>(1, 5) << 0, 0, 0, 0, 1);
	Mat Nkernel = (Mat_<float>(5, 1) << 1, 0, 0, 0, 0);
	Mat Skernel = (Mat_<float>(5, 1) << 0, 0, 0, 0, 1);
	Mat wWeight, eWeight, nWeight, sWeight;
	filter2D(hWeightSum, wWeight, -1, Wkernel); //shift
	filter2D(hWeightSum, eWeight, -1, Ekernel);
	filter2D(vWeightSum, nWeight, -1, Nkernel);
	filter2D(vWeightSum, sWeight, -1, Skernel);
	divide(1.0, (wWeight.mul(wWeight) + 1e-32), wWeight); //divide(float scale, InputArray src2, OutputArray dst, int dtype=-1)
	divide(1.0, (eWeight.mul(eWeight) + 1e-32), eWeight);
	divide(1.0, (nWeight.mul(nWeight) + 1e-32), nWeight);
	divide(1.0, (sWeight.mul(sWeight) + 1e-32), sWeight);

	// combine directional color differences
	Mat weightedF = getGaussianKernel(9, sigma, CV_32F); //(int ksize, float sigma, int ktype=CV_64F). return [ksize x 1] matrix
    Nkernel = (Mat_<float>(9, 1) << 1, 1, 1, 1, 1, 0, 0, 0, 0);
	Nkernel = Nkernel.mul(weightedF);
	float s = sum(Nkernel)[0];
	Nkernel /= s;
	Skernel = (Mat_<float>(9, 1) << 0, 0, 0, 0, 1, 1, 1, 1, 1);
	Skernel = Skernel.mul(weightedF) / s;
	transpose(Nkernel, Wkernel);
	transpose(Skernel, Ekernel);
	Mat fMulGradientSum_N, fMulGradientSum_S, fMulGradientSum_W, fMulGradientSum_E;
	filter2D(dif_V, fMulGradientSum_N, -1, Nkernel); 
	filter2D(dif_V, fMulGradientSum_S, -1, Skernel);
	filter2D(dif_H, fMulGradientSum_W, -1, Wkernel);
	filter2D(dif_H, fMulGradientSum_E, -1, Ekernel);
	Mat totalWeight = nWeight + eWeight + wWeight + sWeight;
	Mat diff;
	divide(nWeight.mul(fMulGradientSum_N) + sWeight.mul(fMulGradientSum_S) + wWeight.mul(fMulGradientSum_W) + eWeight.mul(fMulGradientSum_E), totalWeight, diff);//(InputArray src1, InputArray src2, OutputArray dst, float scale=1, int dtype=-1)

	// Calculate Green by adding bayer raw data 
	finalBGR[1] = diff + Src1ch; //raw CFA data
	Mat imask = (mask[1] == 0); //[0, 1, 1] -> [255, 0, 0]
	imask.convertTo(imask, CV_32F, 1.0/255.0);
	finalBGR[1] = finalBGR[1].mul(imask) + bgr[1]; //avoid original green value been modified

	// clip to 0~1
	//https://docs.opencv.org/master/db/d8e/tutorial_threshold.html
	//(InputArray src, OutputArray dst, float thresh, float maxval, int type)
	threshold(finalBGR[1], finalBGR[1], 1.0, 1.0, THRESH_TRUNC); // > 1 to 1
	threshold(finalBGR[1], finalBGR[1], 0.0, 0.0, THRESH_TOZERO); // < 0 to 0
	//https://docs.opencv.org/master/d3/d63/classcv_1_1Mat.html#adf88c60c5b4980e05bb556080916978b
	//although converTo() wil auto clamped to min/max

	// === 2.Red and Blue ===
	h = 5; //horizontal
	v = 5; //vertical
	eps = 0.0;

	// R interpolation
	Mat tentativeR = guided_filter_modified(bgr[2], finalBGR[1], mask[2], h, v, eps);
	Mat residualR = (bgr[2] - tentativeR).mul(mask[2]);
	Mat bilinearKernel = (Mat_<float>(3, 3) << 0.25, 0.5, 0.25, 0.5, 1.0, 0.5, 0.25, 0.5, 0.25);
	filter2D(residualR, residualR, -1, bilinearKernel);
	finalBGR[2] = residualR + tentativeR;

	// B interpolation
	Mat tentativeB = guided_filter_modified(bgr[0], finalBGR[1], mask[0], h, v, eps);
	Mat residualB = (bgr[0] - tentativeB).mul(mask[0]);
	filter2D(residualB, residualB, -1, bilinearKernel);
	finalBGR[0] = residualB + tentativeB;

	// Merge to single 3 channel Img
	merge(finalBGR, Dst);
	Dst.convertTo(Dst, CV_8U, 255.0);
}


void Demosaicing(std::string &BayerFileName, cv::Mat &Dst, int rows, int cols, int BayerPatternFlag, int DemosaicingMethod){
    Mat RawImage; 
    Mat BayerImage;

    RawImage.create(rows, cols, CV_16UC1); //16 bit short

	//https://stackoverflow.com/questions/15138353/how-to-read-a-binary-file-into-a-vector-of-unsigned-chars
	ifstream inFile;
	inFile.open(BayerFileName, ios::binary);
	if (!inFile.is_open()){
		cout << "Unable to open file" << endl;
        return;
	}

	//16 bit -> short
	// Stop eating new lines in binary mode!!!
	inFile.unsetf(std::ios::skipws);
	// get its size:
	std::streampos fileSize;
	inFile.seekg(0, std::ios::end);
	fileSize = inFile.tellg();
	inFile.seekg(0, std::ios::beg);
	//read data
	inFile.read((char *)RawImage.data, rows * cols * sizeof(short)); //<-- 16 bit short
	inFile.close();

    //cout << "rows: " << RawImage.rows << endl;
	//cout << "cols: " << RawImage.cols << endl;
	//cout << "size: " << RawImage.size() << endl;
	//cout << "dept: " << RawImage.depth() << endl; //0~4
	//cout << "type: " << RawImage.type() << endl;
	//cout << "chal: " << RawImage.channels() << endl

    RawImage.convertTo(BayerImage, CV_32FC1);
	BayerImage.convertTo(BayerImage, CV_8UC1, 255.0 / 1023.0); // divide depend on img bit, care opencv saturate_case

    // R G R
    // G B G
    // R G R
    //copyMakeBorder(src, dst, int top, int bottom, int left, int right, int borderType, const Scalar& value=Scalar() )
    switch(BayerPatternFlag){ //turn all to RGGB by copyMakeBorder
        case 1: //RGGB
            break;
        case 2: //GRBG
            // R|GR|G
            // G|BG|B
            copyMakeBorder(BayerImage, BayerImage, 0, 0, 1, 1, cv::BORDER_REFLECT_101);
            break;
        case 3: //BGGR
            // R|GR|G
            // ------
            // G|BG|B
            // R|GR|G
            // ------
            // G|BG|B
            copyMakeBorder(BayerImage, BayerImage, 1, 1, 1, 1, cv::BORDER_REFLECT_101);
            break;
        case 4: //GBRG
            // RG
            // --
            // GB
            // RG
            // --
            // GB
            copyMakeBorder(BayerImage, BayerImage, 1, 1, 0, 0, cv::BORDER_REFLECT_101);
            break;
        default:
            std::cerr << "Wrong Bayer Pattern Flag" << endl;
            return;
    }
    switch(DemosaicingMethod){
        case 1:
            demosaic_smooth_hue(BayerImage, Dst);
            break;
        case 2:
            demosaic_laplacian_corrected(BayerImage, Dst);
            break;
        case 3:
            demosaic_GBTF(BayerImage, Dst);
            break;
        case 4:
            demosaic_residual(BayerImage, Dst);
            break;
        default:
            std::cerr << "Wrong Demosaicing Method Index" << endl;
            return;
    }

    //Rect (x, y, width, height);
    switch(BayerPatternFlag){ //turn all back from RGGB to Dst
        case 1: //RGGB
            break;
        case 2: //GRBG
            // R|GR|G
            // G|BG|B
            Dst = Dst(Rect(1, 0, BayerImage.cols - 2, BayerImage.rows));
            break;
        case 3: //BGGR
            // R|GR|G
            // ------
            // G|BG|B
            // R|GR|G
            // ------
            // G|BG|B
            Dst = Dst(Rect(1, 1, BayerImage.cols - 2, BayerImage.rows - 2));
            break;
        case 4: //GBRG
            // RG
            // --
            // GB
            // RG
            // --
            // GB
            Dst = Dst(Rect(0, 1, BayerImage.cols, BayerImage.rows - 2));
            break;
    }

    return;
}