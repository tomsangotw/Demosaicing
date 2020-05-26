#include <cstdio>
#include <iostream>
#include <string>
#include <fstream>
#include <cstring>
#include <vector>
#include <opencv2/opencv.hpp>
#include "Demosaicing.hpp"
#include "GBTF.hpp"

using namespace std;
using namespace cv;

int main(){

    // old raw files
    int rows = 2464;
    int columns = 3280;

    /*
    //new raw files
    int rows = 3072;
    int columns = 4096;
    */
    string fileName = "NB_10MA.raw";
    //string fileName = "./1/raw16_4096X3072.16_rggb_raw";
    //string fileName = "./2/raw16_4096X3072.16_rggb_raw";
    //string fileName = "./3/raw16_4096X3072.16_rggb_raw";

	Mat dst;

    for(int i = 1; i <= 4; i++){
        for(int j = 1; j <= 4; j++){
            Demosaicing(fileName, dst, rows, columns, i, j);
            imwrite("cpp" + to_string((i-1)*4 + j) + ".bmp", dst);
        }
    }

	cout << CV_VERSION << endl;
	//cout << cv::getBuildInformation() << endl;
	waitKey(0);
	return 0;
}