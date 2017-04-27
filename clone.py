import csv
import cv2
import numpy as np



lines = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
images = []
measurements = []

# Ignore the header
# Take out the first line in lines
ignoreHeader = False
temp = lines[1]
temp1 = temp[0]

# Store all of the image and driving data.
root = r"E:\Udacity\self-driving-engineering\CarND-Behavioral-Cloning-P3\data\IMG"
for line in lines:
    if(ignoreHeader):
        for i in range(3):
            source_path = line[i]
            filename = source_path.split('/')[-1]
            image = cv2.imread(root +'/' +  filename)
            images.append(image)
            
            # Add a correction value to the right and left camera images
            measurement = float(line[3])
            if i == 1:
                measurement += 0.2
            elif i == 2:
                measurement -=  0.2
            measurements.append(measurement)
    ignoreHeader = True

# Augment the data by fliping it
augmented_images, augmented_measurements = [], []
for image,measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1.0)
    
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

    
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()

# Normalize pixels
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape = (160,320,3)))
# Remove extra background noise from the images
model.add(Cropping2D(cropping=((70,25), (0,0))))

# The architecture
model.add(Convolution2D(24,5,5, subsample = (2,2), activation = "relu"))
model.add(Convolution2D(36,5,5, subsample = (2,2), activation = "relu"))
model.add(Convolution2D(48,5,5, subsample = (2,2), activation = "relu"))
model.add(Convolution2D(64,3,3, activation = "relu"))
model.add(Convolution2D(64,3,3, activation = "relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# Use the adam optimizer
model.compile(loss='mse', optimizer='adam')
print()
print("Fitting the model")

# Start training
history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=3)
print("model fitted")
import matplotlib.pyplot as plt



### Print the keys contained in the history object
print(history_object.history.keys())

### Plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
model.save('model.h5')