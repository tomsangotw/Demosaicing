// This code is modified from 
// https://github.com/RayXie29/GBTF_Color_Interpolation

#include "GBTF.hpp"
#include <opencv2/opencv.hpp>
#include <fstream>


void to_SingleChannel(cv::Mat &src,cv::Mat &dst)
{
    dst = cv::Mat(src.size(),CV_8UC1,cv::Scalar(0));
    for(int i=0;i<src.rows;++i)
    {
        uchar *sptr = src.ptr<uchar>(i);
        uchar *dptr = dst.ptr<uchar>(i);
        for(int j=0;j<src.cols;++j)
        {
            int baseJ = j*3;
            dptr[j] = sptr[baseJ] + sptr[baseJ+1] + sptr[baseJ+2];
        }
    }
}

void CalHVcolordiff(cv::Mat &src1ch,cv::Mat &HCDMap,cv::Mat &VCDMap)
{
    float vhkenel[5] = {-0.25,0.5,0.5,0.5,-0.25};
    
    cv::Mat HK(1,5,CV_32F,vhkenel);
    cv::Mat VK(5,1,CV_32F,vhkenel);
    src1ch.convertTo(src1ch, CV_32F);
    HCDMap = cv::Mat(src1ch.size(),CV_32F);
    VCDMap = cv::Mat(src1ch.size(),CV_32F);
    
    cv::filter2D(src1ch, HCDMap, -1, HK);
    cv::filter2D(src1ch, VCDMap, -1, VK);
    
    float tempArr[2][2] = {{1,-1},{-1,1}};
    cv::Mat tempMat(2,2,CV_32F,tempArr);
    cv::copyMakeBorder(tempMat, tempMat, 0, src1ch.rows-2, 0, src1ch.cols-2, cv::BORDER_REFLECT_101);
    
    src1ch = src1ch.mul(tempMat);
    tempMat *= -1;
    HCDMap = HCDMap.mul(tempMat);
    VCDMap = VCDMap.mul(tempMat);
    
    cv::add(HCDMap,src1ch,HCDMap);
    cv::add(VCDMap,src1ch,VCDMap);
}

void CalHVcolordiffGrad(cv::Mat &HCDMap,cv::Mat &VCDMap,cv::Mat &HGradientMap, cv::Mat &VGradientMap)
{
    float vhkenel2[3] = {-1,0,1};
    cv::Mat HK(1,3,CV_32F,vhkenel2);
    cv::Mat VK(3,1,CV_32F,vhkenel2);
    
    cv::filter2D(HCDMap, HGradientMap, -1, HK);
    cv::filter2D(VCDMap, VGradientMap, -1, VK);
    HGradientMap = cv::abs(HGradientMap);
    VGradientMap = cv::abs(VGradientMap);
}

void CalWeightingTable(std::vector<float> &WeightTable,cv::Mat &HGradientMap, cv::Mat &VGradientMap)
{
    int i,j,k;
    int NSEW[4][2] = { {-4,-2}, {0,-2} , {-2,-4}, {-2,0} };
    for (i = 5, k = 0; i < HGradientMap.rows - 5; ++i)
    {
        j = i%2? 6 : 5;
        for (; j < HGradientMap.cols - 5; j+=2)
        {
            //R & B pixel location
            for(int dir=0;dir<4;++dir,++k)
            {
                for(int a=i+NSEW[dir][0] ; a<=i+NSEW[dir][0]+4 ; ++a)
                {
                    float *VGMPtr = VGradientMap.ptr<float>(a);
                    float *HGMPtr = HGradientMap.ptr<float>(a);
                    for(int b=j+NSEW[dir][1]; b<=j+NSEW[dir][1]+4 ;++b)
                    {
                        if(dir<2) { WeightTable[k] += VGMPtr[b]; }
                        else { WeightTable[k] += HGMPtr[b]; }
                    }
                }
                if(WeightTable[k]) { WeightTable[k] = 1.0 / (WeightTable[k]*WeightTable[k]); }
            }
        }
    }
}

void GBTF_CFAInterpolation(cv::Mat &Bayer,cv::Mat &Dst,int BayerPatternFlag = 0)
{
    cv::Mat Src = Bayer.clone();
    if (BayerPatternFlag == 1) { cv::copyMakeBorder(Src, Src, 0, 0, 1, 1, cv::BORDER_REFLECT_101); }
    else if (BayerPatternFlag == 2) { cv::copyMakeBorder(Src, Src, 1, 1, 1, 1, cv::BORDER_REFLECT_101); }
    else if (BayerPatternFlag == 3) { cv::copyMakeBorder(Src, Src, 1, 1, 0, 0, cv::BORDER_REFLECT_101); }
    else if (BayerPatternFlag)
    {
        std::cerr << "Please select the right Bayer Pattern ,  default(0) ->GRBG , 1->RGGB , 2->GBRG , 3->BGGR" << std::endl;
        return;
    }
    
    Dst = Src.clone();
    cv::Mat src1ch;
    to_SingleChannel(Src, src1ch);
    int i,j,k;
    int channels = 3;
    int width = Src.cols, height = Src.rows;
    
    
    //Calculate Horizontal and vertical color difference maps
    cv::Mat HCDMap,VCDMap;
    CalHVcolordiff(src1ch, HCDMap, VCDMap);
    
    //Calculate Horizontal and vertical color difference gradients
    cv::copyMakeBorder(HCDMap, HCDMap, 5, 5, 5, 5, cv::BORDER_CONSTANT, cv::Scalar(0));
    cv::copyMakeBorder(VCDMap, VCDMap, 5, 5, 5, 5, cv::BORDER_CONSTANT, cv::Scalar(0));
    
    cv::Mat HGradientMap,VGradientMap;
    CalHVcolordiffGrad(HCDMap, VCDMap, HGradientMap, VGradientMap);
    
    //weight table calculation
    int WeightTableSize = height*(int)(width / 2);
    std::vector<float> WeightTable(WeightTableSize * 4);
    CalWeightingTable(WeightTable, HGradientMap, VGradientMap);
    
    //Target Pixel Gradient Map
    cv::Mat TPdiff(height, width, CV_32FC1, cv::Scalar(0));
    
    //Interpolate green value at blue and red pixel location
    for(i=5, k=0; i<HCDMap.rows-5;++i)
    {
        j = i%2? 6 : 5;
        float *TPPtr = TPdiff.ptr<float>(i-5);
        uchar *DstPtr = Dst.ptr<uchar>(i-5);
        for(;j<HCDMap.cols-5; j+=2, k+=4 )
        {
            float N = WeightTable[k], S = WeightTable[k+1], E = WeightTable[k+2] , W = WeightTable[k+3] ;
            float totalWeight = N+S+E+W;
            if(!totalWeight) { TPPtr[j-5] = 0; }
            else
            {
                float NSKernel[9][9] = { 0 };
                NSKernel[4][4] = (N+S)*0.2;
                NSKernel[0][4] = NSKernel[1][4] = NSKernel[2][4] = NSKernel[3][4] = 0.2*N;
                NSKernel[5][4] = NSKernel[6][4] = NSKernel[7][4] = NSKernel[8][4] = 0.2*S;
     
                float WEKernel[9][9] = { 0 };
                WEKernel[4][4] = (W+E)*0.2;
                WEKernel[4][0] = WEKernel[4][1] = WEKernel[4][2] = WEKernel[4][3] = E*0.2;
                WEKernel[4][5] = WEKernel[4][6] = WEKernel[4][7] = WEKernel[4][8] = W*0.2;
     
                cv::Mat Vroi = VCDMap(cv::Rect(j-4,i-4,9,9));
                cv::Mat Hroi = HCDMap(cv::Rect(j-4,i-4,9,9));
                
                for(int idx=0;idx<9;++idx)
                {
                    TPPtr[j-5] += WEKernel[4][idx] * Hroi.at<float>(4,idx) + NSKernel[idx][4] * Vroi.at<float>(idx,4);
                }
                TPPtr[j-5] /= totalWeight;
            }
            DstPtr[(j-5)*channels+1]= cv::saturate_cast<uchar>(DstPtr[(j-5)*channels+i%2+!(j%2)] + TPPtr[j-5]);
        }
    }
    
    // Prb.csv reading
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
    //PrbMat(Prb);
    
    cv::copyMakeBorder(TPdiff, TPdiff, 3, 3, 3, 3, cv::BORDER_CONSTANT, cv::Scalar(0));
    
    //Interpolate R,B channel in blue and red pixel locations
    for(i=3;i<TPdiff.rows-3;++i)
    {
        j = i%2?4:3;
        uchar *DPtr = Dst.ptr<uchar>(i-3);
        for(;j<TPdiff.cols-3;j+=2)
        {
            float sum = 0;
            cv::Mat TProi = TPdiff(cv::Rect(j-3,i-3,7,7));
            
            for(int x=0;x<7;x+=2)
            {
                float *PrbPtr = Prb.ptr<float>(x);
                float *TProiPtr = TProi.ptr<float>(x);
                for(int y=0;y<7;y+=2) { sum += PrbPtr[y]* TProiPtr[y]; }
            }
            DPtr[(j-3)*channels + !(i%2)+j%2 ] = cv::saturate_cast<uchar>(DPtr[(j-3)*channels + 1 ] - sum);
        }
    }
    
    
    //Interpolate R,B channel on G pixel
    int NSEW[4][2] = { {-1,0} , {1,0}, {0,-1}, {0,1} };
    for(i=0;i<height;++i)
    {
        uchar *DstPtr = Dst.ptr<uchar>(i);
        j = i%2?1:0;
        for(;j<width;j+=2)
        {
            uchar G = DstPtr[j*channels+1];
            float GBdiff=0,GRdiff=0;
            for(int dir = 0; dir<4;++dir)
            {
                int x = i+NSEW[dir][0];
                int y = j+NSEW[dir][1];
                if(x>=0 && x<height && y>=0 && y<width)
                {
                    GBdiff += Dst.at<cv::Vec3b>(x,y)[1] - Dst.at<cv::Vec3b>(x,y)[0];
                    GRdiff += Dst.at<cv::Vec3b>(x,y)[1] - Dst.at<cv::Vec3b>(x,y)[2];
                }
            }
            DstPtr[j*channels+2] = cv::saturate_cast<uchar>(G - GRdiff/4);
            DstPtr[j*channels] = cv::saturate_cast<uchar>(G-GBdiff/4);
        }
    }
    
    if (BayerPatternFlag == 1){ Dst = Dst(cv::Rect(1, 0, width - 2, height));   }
    else if (BayerPatternFlag == 2) {   Dst = Dst(cv::Rect(1, 1, width - 2, height - 2)); }
    else if (BayerPatternFlag == 3) {   Dst = Dst(cv::Rect(0, 1, width, height - 2)); }
}

