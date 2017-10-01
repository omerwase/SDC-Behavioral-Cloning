''' 
SDC Project 3: Behavioral Cloning
By: Omer Waseem
Description: The model.py file trains the model on images from the 'data' folder to
predict the car's steering angle. The trained model is saved in 'model.h5' and later
used to drive the car autonomously with 'drive.py'
'''

import os
import csv
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, MaxPooling2D, Dropout

# Hyperparameters
epochs = 7
batch_size = 32

# flipped = True indicates each image and angle will be flipped for additional data
flipped = True
flip_factor = 1

# Reads CSV file for image names and associated angles
samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader) # skip header line
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)
train_len = len(train_samples)
validation_len = len(validation_samples)
if flipped:
    train_len *= 2
    validation_len *= 2
    flip_factor = 2
print("training samples:", train_len)
print("validation samples:", validation_len)

# Data generator for use during training and validation
# Avoids loading all images in memory at once
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './data/IMG/'+batch_sample[0].split('/')[-1]
                # cv2.imread reads image as BGR, converting to RGB (same as images from drive.py)
                temp_image = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB)
                # Image is resized to have 70 columns
                center_image = cv2.resize(temp_image, (70, 160))
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                # Append flipped image
                images.append(np.fliplr(center_image))
                angles.append(center_angle*-1.0)

            # Trim image to only see section with road
            X_train = np.array(images)[:,65:135,:,:]
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# Compiles and trains the model using the generator functions
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

# Trimmed image format
rows, cols, chs = 70, 70, 3  

# Model architecture
model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: (x / 255.0) - 0.5,
        input_shape=(rows, cols, chs),
        output_shape=(rows, cols, chs)))
model.add(Convolution2D(8,5,5,activation='relu'))
model.add(Convolution2D(16,5,5,activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(24,5,5,activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(32,5,5,activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(320))
model.add(Dense(64))
model.add(Dense(16))
model.add(Dense(1))

# Train model
# flip_factor equals 2 if flipped images are used, otherwise its 1
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch=len(train_samples)*flip_factor, 
                    validation_data=validation_generator, nb_val_samples=len(validation_samples)*flip_factor, 
                    nb_epoch=epochs)

# Save model parameters for use with 'drive.py'
model.save('model.h5')
print("Model saved")
