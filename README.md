# Food
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense, Dropout, Activation
from google.colab import drive

from google.colab import drive
drive.mount('/content/drive')

drive.mount('/content/drive',force_remount=True)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
training_set=train_datagen.flow_from_directory('/content/drive/MyDrive/CNN/FO/training_set',
                                               target_size=(256,256), batch_size=32, class_mode ='categorical')
test_set=train_datagen.flow_from_directory('/content/drive/MyDrive/CNN/FO/test_set',
                                               target_size=(256,256), batch_size=32, class_mode ='categorical')
                                               
drive.mount('/content/drive')

model=Sequential()
model.add(Conv2D(128,(3,3),activation='relu',kernel_initializer='he_uniform',padding='same',input_shape=(256,256,3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64,(3,3),activation='relu',kernel_initializer='he_uniform',padding='same'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(32,(3,3),activation='relu',kernel_initializer='he_uniform',padding='same'))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(128,activation='relu',kernel_initializer = 'he_uniform'))
model.add(Dense(10,activation='Softmax'))
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping
model.compile(optimizer = 'adam', loss ='categorical_crossentropy',metrics = ['accuracy'])
callbacks=[EarlyStopping(monitor='val_loss',patience=100)]
history=model.fit(training_set, steps_per_epoch=len(training_set), batch_size = 64, epochs=100,
                  validation_data=test_set, validation_steps=len(test_set), callbacks=callbacks, verbose = 1)
                  
score = model.evaluate(test_set,verbose=0)
print('Sai số kiểm tra là: ',score[0])
print('Độ chính xác kiểm tra là: ',score[1])

model.save('model_fruit.h5')
from tensorflow.keras.models import load_model
model=load_model('model_fruit.h5')

from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np

img_0 = load_img('/content/drive/MyDrive/CNN/FO/prediction/Banh-xeo.jpg', target_size=(256,256))
img_1 = load_img('/content/drive/MyDrive/CNN/FO/prediction/BanhBao.jpg', target_size=(256,256))
img_2 = load_img('/content/drive/MyDrive/CNN/FO/prediction/Bun-ca.jpg', target_size=(256,256))
img_3 = load_img('/content/drive/MyDrive/CNN/FO/prediction/Cha.jpg', target_size=(256,256))
img_4 = load_img('/content/drive/MyDrive/CNN/FO/prediction/Com.jpg', target_size=(256,256))
img_5 = load_img('/content/drive/MyDrive/CNN/FO/prediction/Hamburger.jpg', target_size=(256,256))
img_6 = load_img('/content/drive/MyDrive/CNN/FO/prediction/Hotdog.jpg', target_size=(256,256))
img_7 = load_img('/content/drive/MyDrive/CNN/FO/prediction/Mi.jpg', target_size=(256,256))
img_8 = load_img('/content/drive/MyDrive/CNN/FO/prediction/Pho.jpg', target_size=(256,256))
img_9 = load_img('/content/drive/MyDrive/CNN/FO/prediction/Pizza.jpg', target_size=(256,256))

img = [img_0,img_1,img_2,img_3,img_4,img_5,img_6,img_7,img_8,img_9]
food = ['Bánh xèo','Bánh bao','Bún','Chả','Cơm','Hamburger','Hot dog','Mì','Phở','Pizza']

for i in range(10):
  plt.imshow(img[i])
  imga = img_to_array(img[i])
  imga = imga/255
  imga = np.expand_dims(imga,axis=0)
  result = model.predict(imga)

  if round(result[0][i])==1: prediction = food[i]
  print(prediction)
  plt.show()
