# Self Driving Car Project 3: Behavioral Cloning

**Project Outline:**
* Driving data (centre camera images and steering angles) was collected from the SDC simulator
* A convolutional neural network was trained on this data to predict steering angles
* The trained model was used to autonomously drive the car in the simulator
* Additional data was collected where the car would go off-course
* The final model successfully drove the car around the first track

[//]: # (Image References)

[image1]: ./examples/resized1.png "Resized Image 1"
[image2]: ./examples/resized2.png "Resized Image 2"
[image3]: ./examples/resized3.png "Resized Image 3"
[image4]: ./examples/center1.jpg "Center Driving 1"
[image5]: ./examples/center2.jpg "Center Driving 2"
[image6]: ./examples/center3.jpg "Center Driving 3"
[image7]: ./examples/recovery1.jpg "Recovery Driving 1"
[image8]: ./examples/recovery2.jpg "Recovery Driving 2"
[image9]: ./examples/recovery3.jpg "Recovery Driving 3"
[image10]: ./examples/recovery4.jpg "Recovery Driving 4"

## Rubric Points
### Below the [rubric points](https://review.udacity.com/#!/rubrics/432/view) are addressed individually

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

Included files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and included drive.py and model.h5 files, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline used for training and validating the model with explanatory comments.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The CNN implementation can be found in model.py (lines 80 to 97) and is described in a later section below. It is composed of 4 convolutional layers and 4 dense layers. A smaller model would have been sufficient for track 1. This particular model was chosen to be able to drive on track 2; however, those attempts were not successful.

#### 2. Attempts to reduce overfitting in the model

Dropout layers were employed at various points in the network to reduce overfitting. However using dropout with longer training (more epochs) did not improve the loss in any significant manner. More training data with early termination provided better results, over fewer epochs. Dropout layers were removed in the final CNN model.

#### 3. Model parameter tuning

The model uses an adam optimizer, thus learning rate was not tuned manually (model.py line 101). MSE was used to calculate loss since this is a linear regression model (model.py line 101).

#### 4. Appropriate training data

Training data was collect using the Udacity provided simulator. The mouse was used to steer the car for analog input. Most of the data was based on center driving. Recovery data was collected from sections of the track where the car drove off.

See next section for details on training strategy.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

To begin a simple model was used to test the entire process of data collection, model training, and autonomous driving. The car was driven for 3 laps (2 froward and 1 reverse track direciton). Images from the simulated car's center camera were used for training. The model simply flattened these images and fed into a single output node to predict steering angles. Using this approach the model was able to effectively drive the car in a straight line.

Images from the car's camera were shrunk and cropped to be 70x70 (rows x columns) before they were fed into the network. This was done in both model.py (lines 60 and 69) and drive.py (lines 66 and 67). Originally images were 160x320. The top 65 and bottom 15 rows contained unrelated scenery and were cropped. The 320 image columns were resized to 70 to improve the networks performance during training and autonomous driving. Below are some examples of resized images used in training:

![alt text][image1]
![alt text][image2]
![alt text][image3]

The CNN architecture was improved incrementally by adding convolutional, maxpooling, and dense layers. The model was fine-tuned to minimize loss and avoid overfitting. Originally dropout layers were added to address overfitting (which tended to occur after 5 epochs ). These were removed in the final implementation because additional training data provided better results over fewer epochs. The final model architecture can be found below.

#### 2. Final Model Architecture

The employed CNN can be found in model.py (lines 80 to 97). It is composed of 4 convolutional layers and 4 dense layers:
   1. Lambda layer: for pixel normalization between -0.5 to 0.5
   2. Conv layer: 8 filters @ 5x5 with RELU activation
   3. Conv layer: 16 filters @ 5x5 with RELU activation
   4. Maxpool
   5. Conv layer: 24 filters @ 5x5 with RELU activation
   6. Maxpool
   7. Conv layer: 32 filters @ 5x5 with RELU activation
   8. Maxpool
   9. Dense layer: 320 outputs, no activation
  10. Dense layer: 64 outputs, no activation
  11. Dense layer: 16 outputs, no activation
  12. Dense layer: 1 output, no activation, final layer with predicted steering angle

#### 3. Creation of the Training Set & Training Process

The capture good driving behavior, the car was driven on the center of the track for 2 laps. Then the car was center-driven for 1 lap in the opposite direction to help the model generalize driving behavior. Below are some images of center driving:

![alt text][image4]  
![alt text][image5]  
![alt text][image6]  

The model was trained on these images and used to drive the car autonomously. The model functioned well with this data on track 1; however, at certian points on the track the car would drive off the road. To correct this behavior additional 'recovery' data was collected. For recovery data the car was driven close to the edge where it would drive off autonomously, and then it was corrected to return to the center of the road. Data was only collected during the actual recovery part as shown below:

![alt text][image7]  
![alt text][image8]  
![alt text][image9]  
![alt text][image10]  

This process was repeated for all sections of the track where the model did not perform as expected. In the end the model was able to successfully drive the car through the whole track. An example of a successful lap completed autonmously by the model can be found in run1.mp4.
