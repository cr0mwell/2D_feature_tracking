/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <list>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "matching2D.hpp"

using namespace std;


enum OptionType {Detector, Descriptor};
enum class Detectors {AKAZE=1, BRISK, FAST, Harris, ORB, ShiTomasi, SIFT, SURF};
enum class Descriptors {AKAZE, BRIEF, BRISK, FREAK, ORB, SIFT, SURF};

template<typename T>
void processInput(T &option) {
    cin >> option;
    cin.clear();
    cin.ignore(numeric_limits<streamsize>::max(), '\n');
}

void processInput(string &option) {
    cin >> option;
    transform(option.begin(), option.end(), option.begin(), [](unsigned char c){ return tolower(c);});
    cin.clear();
    cin.ignore(numeric_limits<streamsize>::max(), '\n');
}


void printDetectorOptions() {
    cout << "* the detector type (1-7):" << endl;
    cout << "1. AKAZE" << endl;
    cout << "2. BRISK" << endl;
    cout << "3. FAST" << endl;
    cout << "4. Harris" << endl;
    cout << "5. ORB" << endl;
    cout << "6. ShiTomasi" << endl;
    cout << "7. SIFT" << endl;
    cout << "8. SURF" << endl;
}


void printDescriptorOptions() {
    cout << "* the descriptor type (1-5):" << endl;
    cout << "1. BRIEF" << endl;
    cout << "2. BRISK" << endl;
    cout << "3. FREAK" << endl;
    cout << "4. ORB" << endl;
    cout << "5. SIFT" << endl;
    cout << "6. SURF" << endl;
}


bool getBoolOption(string message) {
    char drawOption;
    cout << message;
    processInput(drawOption);
    
    while (tolower(drawOption) != 'y' && tolower(drawOption) != 'n') {
        cout << "\nInvalid value entered (enter 'y' or 'n') ";
        processInput(drawOption);
    }
    
    return (drawOption == 'y');
}


int getIntOption(string message, int limit=10000) {
    int option;
    cout << message << limit << "]: ";
    processInput(option);
    
    while (!isdigit(option) && (option == 0 || option > limit)) {
        cout << "\nInvalid value entered. " << message << limit << "]: ";
        processInput(option);
    }
    
    return option;
}


int getEnumOption(int optionType) {
    int option;
    vector<int> validOptions;
    void (*printFunc)() = NULL;
    
    if (optionType == Detector) {
        validOptions = {1, 2, 3, 4, 5, 6, 7, 8};
        printFunc = &printDetectorOptions;
    } else {
        validOptions = {1, 2, 3, 4, 5, 6};
        printFunc = &printDescriptorOptions;
    }
    
    printFunc();
    processInput(option);
    
    while (find(validOptions.begin(), validOptions.end(), option) == validOptions.end()) {
        cout << "\nInvalid value entered" << endl;
        printFunc();
        processInput(option);
    }
    
    return option;
}


/* MAIN PROGRAM */
int main(int argc, const char *argv[])
{
    cout << "Welcome to 2D feature tracking application!" << endl;
    cout << "Please set all necessary runtime options to proceed:" << endl;
    
    /* INIT VARIABLES AND DATA STRUCTURES */

    // camera
    string imgBasePath = "../media/";
    string imgPrefix = "KITTI/2011_09_26/image_00/data/000000"; // left camera, color
    string imgFileType = ".png";
    int imgStartIndex = 0;  // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 9;    // last file index to load
    int imgFillWidth = 4;   // no. of digits which make up the file index (e.g. img-0001.png)

    // misc
    int dataBufferSize = 2; // no. of images which are held in memory (ring buffer) at the same time
    list<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time
    bool bVis = false;  // visualize results
    
    // SET RUNTIME VARIABLES
    Detectors detectorType = static_cast<Detectors>(getEnumOption(Detector));
    Descriptors descriptorType;
    if (detectorType != Detectors::AKAZE)
        descriptorType = static_cast<Descriptors>(getEnumOption(Descriptor));
    else
        descriptorType = Descriptors::AKAZE;
    
    // Keep keypoints on the preceding vehicle only
    bool bFocusOnVehicle = getBoolOption("* track features on the preceeding vehicle only? (y/n): ");
    
    // Limit number of keypoints (helpful for debugging and learning)
    bool bLimitKpts = getBoolOption("* set the limit for the key points detection? (y/n): ");
    int maxKeypoints;
    if (bLimitKpts)
        maxKeypoints = getIntOption("* max key points number to process? [0-");
    
    // Use crosscheck for BF matching? (For HOG descriptors only)
    bool crossCheck {false};
    if (descriptorType == Descriptors::SIFT || descriptorType == Descriptors::SURF)
        crossCheck = getBoolOption("* use crosscheck matching for the descriptor? (y/n): ");
    
    string selectorType = "SEL_NN";
    int k {2};
    // No way of using KNN if crosscheck was chosen as an option
    if (!crossCheck)
    {
        selectorType = getBoolOption("* use NN matching filtering algorithm (y) or KNN (n)?: ") ? "SEL_NN" : "SEL_KNN";
        if (selectorType.compare("SEL_KNN") == 0)
            k = getIntOption("* max number of nearest neighbours to process for each keypoint matching (default 2)? [1-", 5);
    }
    
    /* MAIN LOOP OVER ALL IMAGES */

    for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++)
    {
        /* LOAD IMAGE INTO BUFFER */

        // assemble filenames for current index
        ostringstream imgNumber;
        imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
        string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

        // load image from file and convert to grayscale
        cv::Mat img, imgGray;
        img = cv::imread(imgFullFilename);
        cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

        // push image into data frame buffer
        DataFrame frame;
        frame.cameraImg = imgGray;
        dataBuffer.push_back(frame);
        
        if (dataBuffer.size() > dataBufferSize)
            dataBuffer.pop_front();

        cout << "#1: LOAD IMAGE INTO BUFFER done. Number of elements in the buffer: " << dataBuffer.size() << endl;

        /* DETECT IMAGE KEYPOINTS */

        // extract 2D keypoints from current image
        vector<cv::KeyPoint> keypoints;
        
        switch (detectorType)
        {
            case Detectors::AKAZE:
                detKeypointsModern<cv::AKAZE>(keypoints, imgGray, bVis);
                break;
            case Detectors::BRISK:
                detKeypointsModern<cv::BRISK>(keypoints, imgGray, bVis);
                break;
            case Detectors::FAST:
                detKeypointsFast(keypoints, imgGray, bVis);
                break;
            case Detectors::Harris:
                detKeypointsHarris(keypoints, imgGray, bVis);
                break;
            case Detectors::ORB:
                detKeypointsModern<cv::ORB>(keypoints, imgGray, bVis);
                break;
            case Detectors::ShiTomasi:
                detKeypointsShiTomasi(keypoints, imgGray, bVis);
                break;
            case Detectors::SIFT:
                detKeypointsModern<cv::SIFT>(keypoints, imgGray, bVis);
                break;
            case Detectors::SURF:
                detKeypointsModern<cv::xfeatures2d::SURF>(keypoints, imgGray, bVis);
                break;
        }
        
        // Limit the detected keypoints by the area of the vehicle in front of us
        if (bFocusOnVehicle)
        {
            cv::Rect obstacle(535, 180, 180, 150);
            vector<cv::KeyPoint> filtered_keypoints;
            copy_if(keypoints.begin(), keypoints.end(), back_inserter(filtered_keypoints), [obstacle](cv::KeyPoint k){return obstacle.contains(k.pt);} );
            cout << "Total detected keypoints size: " << keypoints.size() << ", size of the keypoints on the preceeding vehicle: " << filtered_keypoints.size() << endl;
            keypoints = filtered_keypoints;
        }
        
        if (bLimitKpts)
        {
            if (detectorType == Detectors::ShiTomasi)
            { // there is no response info, so keep the first 50 as they are sorted in descending quality order
                keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
            }
            cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
            cout << "Note: keypoints size have been limited to " << maxKeypoints << endl;
        }

        // push keypoints and descriptor for current frame to end of data buffer
        (next(dataBuffer.end(), -1))->keypoints = keypoints;
        cout << "#2: DETECT KEYPOINTS done" << endl;

        /* EXTRACT KEYPOINT DESCRIPTORS */
        
        cv::Mat descriptors;
        switch (descriptorType)
        {
            case Descriptors::AKAZE:
                descKeypoints<cv::AKAZE>((next(dataBuffer.end(), -1))->keypoints, (next(dataBuffer.end(), -1))->cameraImg, descriptors, "AKAZE");
                break;
            case Descriptors::BRIEF:
                descKeypoints<cv::xfeatures2d::BriefDescriptorExtractor>((next(dataBuffer.end(), -1))->keypoints, (next(dataBuffer.end(), -1))->cameraImg, descriptors, "BRIEF");
                break;
            case Descriptors::BRISK:
                descKeypoints<cv::BRISK>((next(dataBuffer.end(), -1))->keypoints, (next(dataBuffer.end(), -1))->cameraImg, descriptors, "BRISK");
                break;
            case Descriptors::FREAK:
                descKeypoints<cv::xfeatures2d::FREAK>((next(dataBuffer.end(), -1))->keypoints, (next(dataBuffer.end(), -1))->cameraImg, descriptors, "FREAK");
                break;
            case Descriptors::ORB:
                descKeypoints<cv::ORB>((next(dataBuffer.end(), -1))->keypoints, (next(dataBuffer.end(), -1))->cameraImg, descriptors, "ORB");
                break;
            case Descriptors::SIFT:
                descKeypoints<cv::SIFT>((next(dataBuffer.end(), -1))->keypoints, (next(dataBuffer.end(), -1))->cameraImg, descriptors, "SIFT");
                break;
            case Descriptors::SURF:
                descKeypoints<cv::xfeatures2d::SURF>((next(dataBuffer.end(), -1))->keypoints, (next(dataBuffer.end(), -1))->cameraImg, descriptors, "SURF");
                break;
        }
        
        // push descriptors for current frame to end of data buffer
        (next(dataBuffer.end(), -1))->descriptors = descriptors;

        cout << "#3: EXTRACT DESCRIPTORS done" << endl;

        if (dataBuffer.size() > 1) // wait until at least two images have been processed
        {

            /* MATCH KEYPOINT DESCRIPTORS */

            vector<cv::DMatch> matches;
            string matcherType = (descriptorType == Descriptors::SIFT || descriptorType == Descriptors::SURF) ? "MAT_BF" : "MAT_FLANN";
            string descriptorStructType = (descriptorType == Descriptors::SIFT || descriptorType == Descriptors::SURF) ? "DES_HOG" : "DES_BINARY";
            
            matchDescriptors((next(dataBuffer.end(), -2))->keypoints, (next(dataBuffer.end(), -1))->keypoints,
                             (next(dataBuffer.end(), -2))->descriptors, (next(dataBuffer.end(), -1))->descriptors,
                             matches, descriptorStructType, matcherType, selectorType, crossCheck, k);

            // store matches in current data frame
            (next(dataBuffer.end(), -1))->kptMatches = matches;

            cout << "#4: MATCH KEYPOINT DESCRIPTORS done" << endl;

            // visualize matches between current and previous image
            bVis = true;
            if (bVis)
            {
                cv::Mat matchImg = ((next(dataBuffer.end(), -1))->cameraImg).clone();
                cv::drawMatches((next(dataBuffer.end(), -2))->cameraImg, (next(dataBuffer.end(), -2))->keypoints,
                                (next(dataBuffer.end(), -1))->cameraImg, (next(dataBuffer.end(), -1))->keypoints,
                                matches, matchImg,
                                cv::Scalar::all(-1), cv::Scalar::all(-1),
                                vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

                string windowName = "Matching keypoints between two camera images";
                cv::namedWindow(windowName, 2);
                cv::imshow(windowName, matchImg);
                cout << "Press key to continue to next image" << endl;
                cv::waitKey(0); // wait for key to be pressed
            }
            bVis = false;
        }

    } // eof loop over all images

    return 0;
}