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

# Ignore the first column (id) and the second column (species)
x_features = train_data.iloc[:, 2:].values
print('Number of features', x_features.shape[1])
test_data = pd.read_csv(test_path)
test_ids = test_data.iloc[:, 0]
test_images = list()
for i in test_ids:
    test_images.append(load_image(i))
# Convert the species to category type
y = train_data['species']
yy=y.copy()
# Get the corresponding categories list for species
le = LabelEncoder()
le.fit(y)
y = le.transform(y)
mapp={}
for i in range(len(y)):
    mapp[y[i]]=yy[i]
from keras.models import model_from_yaml
yaml_file = open('model5.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
loaded_model = model_from_yaml(loaded_model_yaml)
# load weights into new model

loaded_model.load_weights("model5.h5")
 


for i in range(len(test_images)):
    x = test_images[i]
    x=cv2.resize(x,(256,256))
    x=np.reshape(x, [1,256,256,1])
    rs=loaded_model.predict(x)
    font = cv2.FONT_HERSHEY_SIMPLEX
    x=np.reshape(x,[256,256])
    x=cv2.cvtColor(x,cv2.COLOR_GRAY2RGB)
    cv2.putText(x, mapp[np.argmax(rs)], (0,20), font, .5, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imwrite('model_Result/'+str(i)+'.png', x)

# evaluate loaded model on test data




