#include <stdio.h>
#include <opencv2/opencv.hpp>

#include "ArUcoMarker/ArUcoMarker.h"
#include "DR/PixMix/PixMixMarkerHiding.h"
#include "CameraCalibration/Calibration.h"

using namespace std;
using namespace cv;

Mat frame;
Mat background;
Mat final_output;
Mat unity_tex;
bool is_Reset;

unsigned char* GetCurrImage(){
    if (unity_tex.cols > 1){
        cvtColor(unity_tex, unity_tex, COLOR_RGB2RGBA);
        return unity_tex.data;
    }
    cvtColor(frame, frame, COLOR_RGB2RGBA);
    return frame.data;
}

extern "C" {

    void ResetPixMix(bool reset){
        if (reset){
            is_Reset = true;
        }
        else{
            is_Reset = false;
        }
    }

    void PixMixImage(unsigned char* bytes, int width, int height, bool isReset){
        
        setUseOptimized(true);
        
        Size imageSize;
        Mat cameraMatrix, distCoeffs;
        io::ReadIntrinsics(string("data/ip.xml"), imageSize, cameraMatrix, distCoeffs);
    
        ArUcoMarker marker(23, 0.036f, 0.02f);
        
        //process incoming stream before we can use it
        frame = Mat(height, width, CV_8UC3, static_cast<void*>(bytes));
        
        dr::PixMixMarkerHiding pmMk(marker, true);
        
        if (!frame.empty()){
            
            Mat color, inpainted, viz;
            color = frame.clone();
            flip(color, color, 1);
            
            vector<Point2f> corners;
            marker.DetectMarkers(color);
            //marker.EstimatePoseSingleMarkers(cameraMatrix, distCoeffs);
            marker.GetCorners(corners);
            
            // inpainting
            if (corners.size() > 0 && (is_Reset)) //r (reset) key
            {
                Mat reset_text = Mat::zeros(color.rows, color.cols, color.type());
                putText(reset_text, String("corners"), Point(50, 75), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));
                flip(reset_text, reset_text, -1);
                color += reset_text;
                
                //dr::det::PixMixParams params;
                //params.alpha = 0.5f;
                //params.maxItr = 10;

                //pmMk.Reset(color, corners, params);
            }
            else if (corners.size() > 0 && pmMk.IsInitiated())
            {
                Mat inpaint_text = Mat::zeros(color.rows, color.cols, color.type());
                putText(inpaint_text, String("inpainted"), Point(50, 95), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
                flip(inpaint_text, inpaint_text, -1);
                color += inpaint_text;
                
                //dr::det::PixMixParams params;
                //params.alpha = 0.0f;
                //params.maxItr = 1;

                //pmMk.Run(color, inpainted, corners, params);
            }

            if (!inpainted.empty())
            {
                //marker.DrawAxis(inpainted, viz, cameraMatrix, distCoeffs, 0.05f);
                viz = inpainted.clone();
            }
            else
            {
                viz = color.clone();
                Mat normal_text = Mat::zeros(color.rows, color.cols, color.type());
                putText(normal_text, String("normal"), Point(50, 55), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0));
                flip(normal_text, normal_text, -1);
                viz += normal_text;
            }
            viz.copyTo(unity_tex);
        }
    }
}

/*
void SetBackground(unsigned char* bytes, int width, int height, bool mirror, bool rotate){
    Mat temp = Mat(height, width, CV_8UC3, static_cast<void*>(bytes));
    if (mirror){
        flip(temp, temp, 1);
    }
    if (rotate){
        cv::rotate(temp, temp, ROTATE_90_CLOCKWISE);
    }
    resize(temp, background, Size(frame.cols,frame.rows), 0, 0, INTER_LINEAR);
}

void RecieveImage(unsigned char* bytes, int width, int height, bool isGreen){
    
    //process incoming stream before we can use it
    frame = Mat(height, width, CV_8UC3, static_cast<void*>(bytes));
    
    if (!frame.empty()){
        
        if (background.cols > 1){
            
            //Converting image from BGR to HSV color space.
            Mat hsv;
            cvtColor(frame, hsv, COLOR_RGB2HSV);
            
            Mat mask1,mask2;
            // Creating masks to detect the upper and lower bounds of color
            
            if (isGreen){
                //green
                inRange(hsv, Scalar(35, 40, 20), Scalar(100, 255, 255), mask1);
                inRange(hsv, Scalar(290, 100, 70), Scalar(300, 100, 100), mask2);
            } else {
                //red
                inRange(hsv, Scalar(0, 180, 70), Scalar(10, 255, 255), mask1);
                inRange(hsv, Scalar(170, 120, 70), Scalar(180, 255, 255), mask2);
            }
            
            // Generating the final mask
            mask1 = mask1 + mask2;
            
            Mat kernel = Mat::ones(3,3, CV_32F);
            morphologyEx(mask1,mask1,cv::MORPH_OPEN,kernel);
            morphologyEx(mask1,mask1,cv::MORPH_DILATE,kernel);
            
            // creating an inverted mask to segment out the cloth from the frame
            bitwise_not(mask1,mask2);
            Mat res1, res2, final_output;
            
            // Segmenting the cloth out of the frame using bitwise and with the inverted mask
            bitwise_and(frame,frame,res1,mask2);
            
            // creating image showing static background frame pixels only for the masked region
            bitwise_and(background,background,res2,mask1);
            
            // Generating the final augmented output.
            addWeighted(res1,1,res2,1,0,final_output);
            final_output.copyTo(unity_tex);
        }
    }
}
*/


