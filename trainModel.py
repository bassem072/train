import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Flatten, MaxPool2D, Conv2D, Dense, Reshape, Dropout
from emnist import extract_training_samples
from emnist import extract_test_samples
from keras.preprocessing.image import ImageDataGenerator
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing import image
from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers.convolutional import ZeroPadding2D, Convolution2D, MaxPooling2D
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, RMSprop, Adam
from keras.layers.advanced_activations import ELU, LeakyReLU, ThresholdedReLU, ReLU
import cv2
from keras.callbacks import ProgbarLogger, ModelCheckpoint
from keras.layers import add

from PIL import Image

target_size = (256, 256)
grayscale = True

# Relative path for the train, test, and submission file
train_path = 'train.csv'
test_path = 'test.csv'
submission_path = 'sample_submission.csv'
submission_output = 'submission.csv'

def load_image(id):
    img_path = 'images/%d.jpg' % (id, )
    img = image.load_img(img_path,
                         grayscale=grayscale)
    img.thumbnail(target_size)
    bg = Image.new('L', target_size, (0,))
    bg.paste(
        img, (int((target_size[0] - img.size[0]) / 2), int((target_size[1] - img.size[1]) / 2))
    )
    img_arr = image.img_to_array(bg)
    
    return img_arr
train_data = pd.read_csv(train_path)
# load the ids in the training data set
x_ids = train_data.iloc[:, 0]
x_images = list()
for i in x_ids:
    x_images.append(load_image(i))
x_images = np.array(x_images)
cv2.imwrite("out.png",x_images[0].squeeze())
plt.imshow(x_images[0].squeeze())
plt.show()
print('Shape of images', x_images[0].shape)
# Ignore the first column (id) and the second column (species)
x_features = train_data.iloc[:, 2:].values
print('Number of features', x_features.shape[1])

# Convert the species to category type
y = train_data['species']
yy=y.copy()
# Get the corresponding categories list for species
le = LabelEncoder()
le.fit(y)
y = le.transform(y)
mapp=[]
for i in range(len(y)):
    mapp[y[i]]=y[i]
    
nb_classes = len(le.classes_)
print('Number of classes', nb_classes)
print('Number of instances', len(y))

plt.hist(y, bins=nb_classes)
plt.title('Number of instances in each class')
plt.xlabel('Class id')
plt.ylabel('Number of instances')
plt.show()

# convert a class vectors (integers) to a binary class matrix
y = np_utils.to_categorical(y)

# Load testing data
test_data = pd.read_csv(test_path)
test_ids = test_data.iloc[:, 0]
test_images = list()
for i in test_ids:
    test_images.append(load_image(i))
test_images = np.array(test_images)

# Load submission file
submission_data = pd.read_csv(submission_path)
sss = StratifiedShuffleSplit(10, 0.2, random_state=15)
for train_index, test_index in sss.split(x_images, y):
	x_train_images, x_test_images, x_train_features, x_test_features = x_images[train_index], x_images[test_index], x_features[train_index], x_features[test_index]
	y_train, y_test = y[train_index], y[test_index]
# The folds are made by preserving the percentage of samples for each class
sss = StratifiedShuffleSplit(10, 0.2, random_state=15)
for train_index, test_index in sss.split(x_images, y):
	x_train_images, x_test_images, x_train_features, x_test_features = x_images[train_index], x_images[test_index], x_features[train_index], x_features[test_index]
	y_train, y_test = y[train_index], y[test_index]

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(256,256,1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes, activation='softmax'))


model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train_images, y_train, validation_data=(x_test_images, y_test), epochs=100, batch_size=10, verbose=1)
scores = model.evaluate(x_test_images,y_test, verbose=0)

#********save model*****
model_yaml = model.to_yaml()
with open("model5.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
# serialize weights to HDF5
model.save_weights("model5.h5")