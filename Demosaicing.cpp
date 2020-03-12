#include <cstdio>
#include <iostream>
#include <fstream>
#include <cstring>
#include <cstring>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int rows = 2464;
int columns = 3280;

Mat RawImage; //Mat->Matrix 宣告一個圖片變數
Mat BayerImage;
Mat Image;
Mat CvDe;
Mat CvDe_VNG;
Mat CvDe_EA;

/*
void mySharpen(){
	for(int i = 0; i < image.rows; i++){
		for(int j = 0; j < image.cols; j++){
			if(i == 0 || i == image.rows - 1 || j == 0 || j == image.cols - 1){
				result1.at<Vec3b>(i, j)[0] = 0;
				result1.at<Vec3b>(i, j)[1] = 0;
				result1.at<Vec3b>(i, j)[2] = 0;
			}else{
				int temp;
				for(int k = 0; k < 3; k++){
					temp = 5 * int(image.at<Vec3b>(i, j)[k]);
					temp -= image.at<Vec3b>(i-1, j)[k];
					temp -= image.at<Vec3b>(i, j-1)[k];
					temp -= image.at<Vec3b>(i+1, j)[k];
					temp -= image.at<Vec3b>(i, j+1)[k];
					if(temp < 0){
						result1.at<Vec3b>(i, j)[k] = 0;
					}else if(temp > 255){
						result1.at<Vec3b>(i, j)[k] = 255;
					}else{
						result1.at<Vec3b>(i, j)[k] = temp;
					}
				}
			}
			//cout << image.at<Vec3b>(i, j);
		}
		//cout << endl;
	}
}*/


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
    inFile.open("NB_10MA.raw", ios::binary);
    if(!inFile.is_open()){
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
	cout << "dept: " << RawImage.depth() << endl;//0~4
	cout << "type: " << RawImage.type() << endl;
	cout << "chal: " << RawImage.channels() << endl;

	for(int i = 16; i < 32; i++){
		for(int j = 0; j < 8; j++){
			cout << RawImage.at<ushort>(i, j) << " ";
		}
		cout << endl;
	}

    RawImage.convertTo(BayerImage, CV_64FC1);
    BayerImage = BayerImage.mul(255.0/1023.0);
    BayerImage.convertTo(BayerImage, CV_8UC1);
    
    cvtColor(BayerImage, CvDe, COLOR_BayerBG2BGR);
    cvtColor(BayerImage, CvDe_EA, COLOR_BayerBG2BGR_EA);
    cvtColor(BayerImage, CvDe_VNG, COLOR_BayerBG2BGR_VNG); //github opencv issue 15011


	//namedWindow("Display Image", WINDOW_AUTOSIZE ); //創建一個 window 並給名稱
	//namedWindow("Sharpen Image", WINDOW_AUTOSIZE ); 

	//imshow("RawImage Image", BayerImage); //給定一個 window，並給要顯示的圖片
	//imshow("Image", Image);
    imwrite("CvDe.bmp", CvDe);
    imwrite("CvDe_EA.bmp", CvDe_EA);
    imwrite("CvDe_VNG.bmp", CvDe_VNG);


    cout << CV_VERSION << endl;
	waitKey(0);
	return 0; 
}
