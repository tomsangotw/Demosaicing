#include <cstdio>
#include <iostream>
#include <fstream>
#include <cstring>
#include <cstring>
#include <vector>
#include <opencv2/opencv.hpp>
#include "GBTF.hpp"

using namespace std;
using namespace cv;
/*
// old raw files
int rows = 2464;
int columns = 3280;
*/

//new raw files
int rows = 3072;
int columns = 4096;

//char fileName[] = "NB_10MA.raw";
//char fileName[] = "./1/raw16_4096X3072.16_rggb_raw";
//char fileName[] = "./2/raw16_4096X3072.16_rggb_raw";
char fileName[] = "./3/raw16_4096X3072.16_rggb_raw";

Mat RawImage; //Mat->Matrix 宣告一個圖片變數
Mat BayerImage;
Mat Image;
Mat CvDe;
Mat CvDe_VNG;
Mat CvDe_EA;


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
void toSingleChannel(cv::Mat &src,cv::Mat &dst){
    if(src.channels() != 3){
        std::cerr << "to_SingleChannel need 3 channel image" << std::endl;
        return;
    }  
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
    }
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

	//kernals
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

	// kernals
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
#       https://ieeexplore.ieee.org/document/5654327
*/
void demosaic_GBTF(cv::Mat &Bayer,cv::Mat &Dst){
    cv::Mat Src = Bayer.clone();
    if(Bayer.channels() == 1){ //input 1 channel -> 3 channel Bayer
        bayer_split(Bayer, Src);
    }
	// for calculate gradients
	Mat src1ch;
	toSingleChannel(Src, src1ch);
	src1ch.convertTo(src1ch, CV_32F, 1.0 / 255.0);

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
	float VHkernal[3] = {-1,0,1}; //(central difference form)
    cv::Mat HK(1, 3, CV_32F, VHkernal);
    cv::Mat VK(3, 1, CV_32F, VHkernal);
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
							Weight[ dir ] += V_diff_gradient.at<float>(4 + row + a + i, col + b + j); // shift 4 due to copyMakeBorder
						}else{ // W, E -> Horizontal
							Weight[ dir ] += H_diff_gradient.at<float>(4 + row + a + i, col + b + j);
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

	cout << "Test" << endl;

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
//https://medium.com/@gary1346aa/%E5%B0%8E%E5%90%91%E6%BF%BE%E6%B3%A2%E7%9A%84%E5%8E%9F%E7%90%86%E4%BB%A5%E5%8F%8A%E5%85%B6%E6%87%89%E7%94%A8-78fdf562e749
//https://github.com/atilimcetin/guided-filter
// I: imput
// r: radius
Mat box_filter(const Mat &I, int r){
	Mat result;
	// 等同 call boxFiltr (全部取平均)
	// boxFilter(src, dst, src.type(), anchor, true, borderType).
	blur(I, result, Size(2 * r + 1, 2 * r + 1));
	return result;
}
// p: origin img
// I: guided img
// r: local window radius
// eps: regularization parameter (0.1)^2, (0.2)^2...
Mat guided_filter(const Mat &originP, const Mat &originI, int r, double eps){
	Mat p, I;
	originP.convertTo(p, CV_64F, 1.0 / 255.0); //(a * (i,j) + b)
	originI.convertTo(I, CV_64F, 1.0 / 255.0);

	//step: 1
	Mat mean_I = box_filter(I, r);
	Mat mean_p = box_filter(p, r);
	Mat corr_I = box_filter(I.mul(I), r); //mul: element wise mul
	Mat corr_Ip = box_filter(I.mul(p), r);
	//step: 2
	Mat var_I = corr_I - mean_I.mul(mean_I);
	Mat cov_Ip = corr_Ip - mean_I.mul(mean_p);
	//step: 3
	Mat a;
	if (var_I.channels() == 3){
		a = cov_Ip / (var_I + Scalar(eps, eps, eps)); //otherwise only 1 channel get added
	}else{
		a = cov_Ip / (var_I + eps);
	}
	Mat b = mean_p - a.mul(mean_I);
	imwrite("cvar_I.png", (var_I + eps) * 255.0);
	imwrite("ca.png", a * 255.0);
	//step: 4
	Mat mean_a = box_filter(a, r);
	Mat mean_b = box_filter(b, r);
	//step: 5
	Mat q = mean_a.mul(I) + mean_b;
	Mat res;

	q *= 255;

	return q;
}

int main(){
	// read raw byte files
	//http://www.cplusplus.com/doc/tutorial/files/
	//https://stackoverflow.com/questions/36658734/c-get-all-bytes-of-a-file-in-to-a-char-array
	//https://stackoverflow.com/questions/21662520/reading-a-dat-file-two-bytes-at-a-time
	// read in vector
	//https://stackoverflow.com/questions/15138353/how-to-read-a-binary-file-into-a-vector-of-unsigned-chars
	RawImage.create(rows, columns, CV_16UC1);
	Image.create(rows, columns, CV_8UC3);

	//https://stackoverflow.com/questions/15138353/how-to-read-a-binary-file-into-a-vector-of-unsigned-chars
	ifstream inFile;
	inFile.open(fileName, ios::binary);
	if (!inFile.is_open()){
		cout << "Unable to open file" << endl;
	}

	//16 bit -> short
	// Stop eating new lines in binary mode!!!
	inFile.unsetf(std::ios::skipws);
	// get its size:
	std::streampos fileSize;
	inFile.seekg(0, std::ios::end);
	fileSize = inFile.tellg();
	inFile.seekg(0, std::ios::beg);
	cout << fileSize << endl;
	//read data
	inFile.read((char *)RawImage.data, rows * columns * sizeof(short));
	inFile.close();

	/*
    // File C style
    FILE *fp = fopen("NB_10MA.raw", "rb");
    if(fp == NULL){
        puts("File Error");
        exit(1);
    }*/

	//image = imread( argv[1], 1 );//讀圖
	//(檔案名稱, flag < 0原圖; flag=0 灰階; flag>0 BGR)
	cout << "rows: " << RawImage.rows << endl;
	cout << "cols: " << RawImage.cols << endl;
	cout << "size: " << RawImage.size() << endl;
	cout << "dept: " << RawImage.depth() << endl; //0~4
	cout << "type: " << RawImage.type() << endl;
	cout << "chal: " << RawImage.channels() << endl;

	for (int i = 16; i < 32; i++){
		for (int j = 0; j < 8; j++){
			cout << RawImage.at<ushort>(i, j) << " ";
		}
		cout << endl;
	}

	RawImage.convertTo(BayerImage, CV_64FC1);
	BayerImage = BayerImage.mul(255.0 / 1023.0); //element wise
	BayerImage.convertTo(BayerImage, CV_8UC1);

/*
	//OpenCV's demosaicing 
	// -bilinear
	// -edge-aware
	// -variable number of gradients
    cvtColor(BayerImage, CvDe, COLOR_BayerBG2BGR);
    cvtColor(BayerImage, CvDe_EA, COLOR_BayerBG2BGR_EA);
    cvtColor(BayerImage, CvDe_VNG, COLOR_BayerBG2BGR_VNG); //github opencv issue 15011
	// = demosaicing(BayerImage, CvDe_VNG, COLOR_BayerBG2BGR_VNG);
    imwrite("CvDe.bmp", CvDe);
    imwrite("CvDe_EA.bmp", CvDe_EA);
    imwrite("CvDe_BG_VNG.bmp", CvDe_VNG);
*/


/*
	//Guided filter
	Mat lena = imread("lena.jpg", IMREAD_COLOR);
	Mat res = guided_filter(lena, lena, 8, 0.05*0.05);
	imwrite("GF.png", res);
*/
/*
	//GBTF
	// https://github.com/RayXie29/GBTF_Color_Interpolation
	Mat dst;
	Mat threeChannel;
	cvtColor(BayerImage, threeChannel, COLOR_GRAY2BGR);
	imwrite("threeChannel.bmp", threeChannel);
	ConvertToThreeChannelBayerBG(threeChannel);
	//Mat src = imread("bayer_pattern_img.bmp", IMREAD_COLOR);
	GBTF_CFAInterpolation(threeChannel, dst, 3); //3: BGGR
	cvtColor(dst, dst, COLOR_RGB2BGR);
	imwrite("GBTF.bmp", dst);
*/
	Mat dst;
	//demosaic_smooth_hue(BayerImage, dst);
	//demosaic_laplacian_corrected(BayerImage, dst);
	demosaic_GBTF(BayerImage, dst);
	//BayerImage
	imwrite("cppGBTF.bmp", dst);

	cout << CV_VERSION << endl;
	//cout << cv::getBuildInformation() << endl;
	waitKey(0);
	return 0;
}
