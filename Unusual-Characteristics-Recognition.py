import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import gc
import os
import argparse
from sklearn.metrics import f1_score
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
import csv
from sklearn.datasets import load_files
from glob import glob
from tensorflow.contrib.distributions.python.ops.bijectors import inline
train = pd.read_csv('train.csv')


# Function to extract frames
def frameCapture(path):
    # Path to video file
    vidObj = cv2.VideoCapture(path)
    # Used as counter variable
    count = 0
    # checks whether frames were extracted
    success = 1
    while success:
        # vidObj object calls read
        # function extract frames
        success, image = vidObj.read()
        # Saves the frames with frame-count
        if count == 1423:
            # if the number of frames/images generated from the video go beyond this number, break the loop
            break
        else:
            cv2.imwrite("C:\\Users\\Adi\\Desktop\\Python Projects\\FeatureExtract\\testdataset\\%.8d.jpg" % count, image)
            with open('test1.csv', mode='a', newline='') as employee_file:
                employee_writer = csv.writer(employee_file)
                employee_writer.writerow(["%.8d.jpg"%count])
                employee_file.close()
        count += 1
# Driver Code
if __name__ == '__main__':
# Calling the function
    frameCapture("C:\\Users\\Adi\\Desktop\\Python Projects\\FeatureExtract\\sheep videos\\Sheep.mp4")
# Save the result in test1.csv file
test = pd.read_csv('test1.csv')
print("Result saved Successfully")
print('Training dataset consists of {} images with {} attributes'.format(train.shape[0], train.shape[1]-1))
print('Testing dataset consists of {} images.'.format(test.shape[0]))
print('Columns in the dataset:\n', train.columns)
# Now we visualise our data #
cols = list(train.columns)
cols.remove('Image_name')
cols.sort()
print("Only Attributes are taken:\n",cols)
count_labels = train[cols].sum()
count_labels.sort_values(inplace=True)
print("The attributes that are found in the images:\n",count_labels)
# To plot the attributes in a graph and to display #
plt.figure(figsize=(18, 8))
ax = sns.barplot(x=count_labels.index, y=count_labels.values)
ax.set_xticklabels(labels=count_labels.index,rotation=90, ha='right')
ax.set_ylabel('Count')
ax.set_xlabel('Attributes/ Labels')
ax.title.set_text('Label/ Attribute distribution')
plt.tight_layout()
# plt.show(ax) #
# Compute the cooccurrence matrix for the labels
label_data = np.array(train[cols])
cooccurrence_matrix = np.dot(label_data.transpose(), label_data)
print('\n Co-occurence matrix: \n', cooccurrence_matrix)
# to compute the co-occurence matrix in percentage
cooccurrence_matrix_diagonal = np.diagonal(cooccurrence_matrix)
with np.errstate(divide = 'ignore', invalid='ignore'):
    cooccurrence_matrix_percentage = np.nan_to_num(np.true_divide(cooccurrence_matrix, cooccurrence_matrix_diagonal))
print('\n Co-occurrence matrix percentage: \n', cooccurrence_matrix_percentage)
# To see which labels usually occur together #
ax = plt.figure(figsize=(18, 12))
sns.set(style='white')
     # Generate a custom diverging colormap
cmap = sns.diverging_palette(200, 10, as_cmap=True)
     # Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(cooccurrence_matrix_percentage, cmap=cmap, center=0, square=True, linewidths=0.15, cbar_kws={"shrink": 0.5})
plt.title('Co-occurrence Matrix of the Labels')
# plt.show(sns) # # To display the graphs #
# To set the paths to the dataset containing images #
TRAIN_PATH = 'trainingdataset/'
TEST_PATH = 'testdataset/'
img_path = TRAIN_PATH + str(train.Image_name[0])
# Image.open(img_path).show()
# to read the images in the form of array #
img = cv2.imread(img_path)
print("The colours that can be read of the various images are: \n", img)
# Extracting label columns
label_cols = list(set(train.columns)-set(['Image_name']))
label_cols.sort()
# Extracting labels corresponding to image at the zeroth index of the training dataset.
labels = train.iloc[0][1:].index[train.iloc[0][1:] == 1]
# We plot the Animal and the attributes/ labels corresponding to it.
txt = 'Labels/ Attributes: ' + str(labels.values)
axl = plt.figure(figsize=(10, 10))
axl.text(.5, .05, txt, ha='center')
# plt.show(plt.imshow(img))
# we need to pre-process our data before sending it to the training model
# The below function reads an image and resizes it to 128 x 128 dimensions and returns it.


def read_img(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (128, 128))
    return img


#temp = train.sample(frac=0.3)
#train = temp.reset_index(drop=True)
train_img = []
for img_path in tqdm(train.Image_name.values):
    train_img.append(read_img(TRAIN_PATH + img_path))
# Convert the image data into an array.
# Since the range of color(RGB) is in the range of (0-255).
# Hence by dividing each image by 255, we convert the range to (0.0 - 1.0)

X_train = np.array(train_img, np.float32) / 255.
# Next, we will calculate the mean and standard deviation.
mean_img = X_train.mean(axis=0)
std_dev = X_train.std(axis=0)
# Next, we will normalize the image data using the following formula:
X_norm = (X_train - mean_img)/ std_dev
print(X_norm.shape)
del X_train
y = train[label_cols].values
# Finally, we create the training and validation sets.
Xtrain, Xvalid, ytrain, yvalid = train_test_split(X_norm, y, test_size=0.25, random_state=90)
del X_norm
gc.collect()
# We will be using the Keras framework to create our model

# We will use a Sequential model, which is a linear stack of layers to build this model.
gc.collect()
model = Sequential()
model.add(BatchNormalization(input_shape=Xtrain.shape[1:]))
model.add(Conv2D(32, kernel_size=(3, 3), activation= 'relu'))

model.add(Conv2D(64, kernel_size=(3, 3), activation= 'relu', padding='same'))
model.add(Conv2D(64, kernel_size=(3, 3), activation= 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(6, activation='sigmoid'))
# to check the model summary
model.summary()
# we define our loss function, the optimizer and metrics for our model.
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
checkpointer = ModelCheckpoint(filepath='weights.best.eda.hdf5', verbose=1, save_best_only=True)
# Finally, we train our model.
model.fit(Xtrain, ytrain, validation_data=(Xvalid, yvalid), epochs=4, batch_size=10, callbacks=[checkpointer], verbose=1)
# Now testing time

test_img = []
for img_path in tqdm(test.Image_name.values):
    test_img.append(read_img(TEST_PATH + img_path))
X_test = np.array(test_img, np.float32) / 255.
# test images are normalised below
mean_img = X_test.mean(axis=0)
std_dev = X_test.std(axis=0)
X_norm_test = (X_test - mean_img)/std_dev
# To predict the labels on the test images
model.load_weights('weights.best.eda.hdf5')
pred_test = model.predict(X_norm_test).round()
pred_test = pred_test.astype(np.int)
print(pred_test)
# Save the result in csv file
subm = pd.DataFrame()
subm['Image_name'] = test.Image_name
label_df = pd.DataFrame(data=pred_test, columns=label_cols)
subm = pd.concat([subm, label_df], axis=1)
subm.to_csv('result.csv', index=True)
print("Result saved Successfully")

# detect, then generate a set of bounding box colors for each class

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable","dog", "horse",
           "motorbike", "person", "pottedplant",
           "sheep","sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt","MobileNetSSD_deploy.caffemodel")

# load the input image and construct an input blob for the image
# by resizing to a fixed 300x300 pixels and then normalizing it
# (note: normalization is done via the authors of the MobileNet SSD
# implementation)

data = pd.read_csv('result.csv')
c = 0
k = 0
for img_path in tqdm(test.Image_name.values):
    image1 = cv2.imread(TEST_PATH + img_path)
    (h, w) = image1.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image1, (300, 300)), 0.007843, (300, 300), 127.5)
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--confidence", type=float, default=0.2,help="minimum probability to filter weak detections")
    args = vars(ap.parse_args())
# pass the blob through the network and obtain the detections and
# predictions
    print("[INFO] computing object detections...")
    net.setInput(blob)
    detections = net.forward()

# loop over the detections
    for i in np.arange(0, detections.shape[2]):
    # extract the confidence (i.e., probability) associated with the
    # prediction
        confidence = detections[0, 0, i, 2]

        if confidence > args["confidence"]:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
# display the prediction
            d1 = data.loc[c:c, ['attrib_01', 'attrib_02', 'attrib_03', 'attrib_04', 'attrib_05', 'attrib_06']]
            for m in d1.values:
                k = 1
                for j in m:
                    if j == 1:
                        break
                    k += 1
            if k == 1:
                labelz = 'SKINNY'
            elif k == 2:
                labelz = 'AGGRESSIVE'
            elif k == 3:
                labelz = 'REDNESS'
            elif k == 4:
                labelz = 'NON-AGGRESSIVE'
            elif k == 5:
                labelz = 'FORGAGING'
            elif k == 6:
                labelz = 'BEEFY'
            else:
                labelz = 'NONE'
            label = "{}: {}: {:.2f}%".format(labelz, CLASSES[idx], confidence * 100)
            print("[INFO] {} {}".format(label, labelz))
            cv2.rectangle(image1, (startX, startY), (endX, endY),COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(image1, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

# show the output image
    cv2.imwrite("C:\\Users\\Adi\\Desktop\\Python Projects\\FeatureExtract\\frames with bounds\\%.8d.jpg" %c, image1)
    c += 1
# To convert images frames into video
from os.path import isfile, join


def convert_frames_to_video(pathIn, pathOut, fps):
    frame_array = []
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]

    # for sorting the file names properly
    files.sort()

    for i in range(len(files)):
        filename = pathIn + files[i]
        # reading each files
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        print(filename)
        # inserting the frames into an image array
        frame_array.append(img)

    out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'MP4V'), fps, size)

    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()


def main():
    pathIn = 'frames with bounds/'
    pathOut = 'Result Video/resultvideo.mp4'
    fps = 17.0
    convert_frames_to_video(pathIn, pathOut, fps)


if __name__ == "__main__":
    main()

print("Project compiled successfully")