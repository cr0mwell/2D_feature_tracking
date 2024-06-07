# 2D Feature Tracking

### (based on [Sensor Fusion Nanodegree Program](https://www.udacity.com/course/sensor-fusion-engineer-nanodegree--nd313))

<img src="media/example.gif" width="1490" height="300" />

The application is an interim project for the more advanced collision detection system.
Using the interactive CLI it allows to:
* Select a keypoint detector among AKAZE, BRISK, FAST, Harris, ORB, ShiTomasi SIFT and SURF;
* Select a descriptor from HOG and binary descriptors list (AKAZE, BRIEF, BRISK, FREAK, ORB, SIFT and SURF);
* Choose whether to track the keypoints on the whole image or at the preceding vehicle only;
* Limit the number of keypoints being tracked;
* Choose whether to use cross-check matching approach for SIFT descriptor; 
* Choose between NN and KNN matching algorithms to filter false positive tracking (option is available when cross-check is disabled);

The initial environment was developed by [Udacity](https://github.com/udacity/SFND_2D_Feature_Tracking/tree/master).
In the current application were implemented lots of additional features, like:
* Processing buffer optimization (only 2 subsequent images are stored for the memory optimization);
* Integrated lots of detector and descriptor algorithms from OpenCV library;
* Implemented brute force and FLANN matching approaches;
* Implemented KNN filtering algorithm to reduce the number of false positive matches;
* Added interactive CLI runtime options selection;

## Environment prerequisites
1. Ubuntu 22.04
2. C++ standard v14
3. gcc >= 11.4
4. cmake >= 3.22
5. make >= 4.3
6. OpenCV >= 4.9
* NOTE: this must be compiled from source using the `-D OPENCV_EXTRA_MODULES_PATH=<opencv_contrib>modules` and `-D OPENCV_ENABLE_NONFREE=ON` cmake flags for testing the SIFT and SURF detectors/descriptors. Refer to the [official installation instructions](https://docs.opencv.org/master/df/d65/tutorial_table_of_content_introduction.html).

## Basic Build Instructions
1. Clone this repo
    ```shell
   cd ~
   git clone https://github.com/cr0mwell/2D_feature_tracking.git
   ```
2. Make a build directory in the top level directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./2D_feature_tracking`.
