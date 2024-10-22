import warnings
warnings.filterwarnings('ignore')
import matplotlib as plt
import IPython.display as ipd
import librosa
import librosa.display
import pandas as pd
import numpy as np
import os
  
from keras import models
from keras import layers
from tensorflow.compat.v1 import ConfigProto  # type: ignore
from tensorflow.compat.v1 import InteractiveSession  # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.utils import to_categorical # type: ignore
from sklearn.preprocessing import StandardScaler
data = pd.read_csv('D:\\ML PROJECT\\dataset.csv')
X = data.drop('target', axis=1)  # Replace 'target' with your actual target column name
y = data['target'] 
encoder = OneHotEncoder()
y = encoder.fit_transform(np.array(y).reshape(-1,1)).toarray()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=0, shuffle=True)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train.shape, y_train.shape, X_test.shape, y_test.shape
def plotter(history):
  plt.figure()
  plt.plot(history.history['loss'],label='train loss')
  plt.plot(history.history['val_loss'],label='test loss')
  plt.xlabel('iterations')
  plt.ylabel('losses')
  plt.legend()
  plt.figure()
  plt.plot(history.history['accuracy'],label='train accuracy')
  plt.plot(history.history['val_accuracy'],label='test accuracy')
  plt.xlabel('iterations')
  plt.ylabel('accuracy')
  plt.legend()


config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


model = models.Sequential()
model.add(layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(y_train.shape[1], activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

history = model.fit(X_train,y_train,batch_size=256,epochs = 1000,verbose=1,validation_data=(X_test,y_test))

plotter(history)

model.summary()
result = model.evaluate(X_test,y_test)
print(result)
def mlppredict(extracted_features):
    data = pd.read_csv('D:\\ML PROJECT\\dataset.csv')
    X = data.drop('target', axis=1)  # Replace 'target' with your actual target column name
    y = data['target'] 
    encoder = OneHotEncoder()
    y = encoder.fit_transform(np.array(y).reshape(-1,1)).toarray()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=0, shuffle=True)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_train.shape, y_train.shape, X_test.shape, y_test.shape
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    model = models.Sequential()
    model.add(layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(y_train.shape[1], activation='softmax'))

    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

    history = model.fit(X_train,y_train,batch_size=256,epochs = 1000,verbose=1,validation_data=(X_test,y_test))
    plt.figure()
    plt.plot(history.history['loss'],label='train loss')
    plt.plot(history.history['val_loss'],label='test loss')
    plt.xlabel('iterations')
    plt.ylabel('losses')
    plt.legend()
    plt.figure()
    plt.plot(history.history['accuracy'],label='train accuracy')
    plt.plot(history.history['val_accuracy'],label='test accuracy')
    plt.xlabel('iterations')
    plt.ylabel('accuracy')
    plt.legend()
    model.summary()
    result = model.evaluate(X_test,y_test)
    print(result)
    a= np.array(extracted_features).reshape(1, -1)
    scaled_input =  scaler.transform(a,scaler)
    prediction = model.predict(scaled_input)
    emotions = {
        '01': 'neutral',
        '02': 'calm',
        '03': 'happy',
        '04': 'sad',
        '05': 'angry',
        '06': 'fearful',
        '07': 'disgust',
        '08': 'surprised'
    }
    predicted_class = np.argmax(prediction, axis=1)[0]
    emotion_code = f"{predicted_class + 1:02}"  # Ensure the key is in '01' format
    predicted_emotion = emotions.get(emotion_code, "Unknown emotion")
    return predicted_emotion