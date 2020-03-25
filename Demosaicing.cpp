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

void ConvertToThreeChannelBayerBG(Mat &BGRImage){
	/*
	Assuming a Bayer filter that looks like this:
	# // 0  1  2  3
	///////////////
	0 // R  G  R  G
	1 // G  B  G  B
	2 // R  G  R  G 
	3 // G  B  G  B

	GR
	BG

	*/
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

//https://medium.com/@gary1346aa/%E5%B0%8E%E5%90%91%E6%BF%BE%E6%B3%A2%E7%9A%84%E5%8E%9F%E7%90%86%E4%BB%A5%E5%8F%8A%E5%85%B6%E6%87%89%E7%94%A8-78fdf562e749
//https://github.com/atilimcetin/guided-filter
// I: imput
// r: radius
Mat box_filter(const Mat &I, int r)
{
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



	cout << CV_VERSION << endl;
	//cout << cv::getBuildInformation() << endl;
	waitKey(0);
	return 0;
}
