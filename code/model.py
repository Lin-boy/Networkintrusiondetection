# ----------------------------------------
# Written by Xiao Feng
# ----------------------------------------
import tensorflow as tf
import numpy as np
from config import cfg


LSTMModel = tf.keras.Sequential()
LSTMModel.add(tf.keras.layers.LSTM(64,input_dim=cfg.IN_DIM, return_sequences=True))
LSTMModel.add(tf.keras.layers.Dropout(0.1))
LSTMModel.add(tf.keras.layers.LSTM(128,return_sequences=True))
LSTMModel.add(tf.keras.layers.Dropout(0.1))
LSTMModel.add(tf.keras.layers.LSTM(128,return_sequences=True))
LSTMModel.add(tf.keras.layers.Dropout(0.1))
LSTMModel.add(tf.keras.layers.LSTM(128,return_sequences=False))
LSTMModel.add(tf.keras.layers.Dropout(0.1))
LSTMModel.add(tf.keras.layers.Dense(cfg.MODEL_NUM_CLASSES))
LSTMModel.add(tf.keras.layers.Activation('softmax'))


FCNModel = tf.keras.Sequential()
FCNModel.add(tf.keras.layers.Dense(1024,input_dim=cfg.IN_DIM,activation='relu'))
FCNModel.add(tf.keras.layers.Dropout(0.01))
FCNModel.add(tf.keras.layers.Dense(512,activation='relu'))
FCNModel.add(tf.keras.layers.Dropout(0.01))
FCNModel.add(tf.keras.layers.Dense(512,activation='relu'))
FCNModel.add(tf.keras.layers.Dropout(0.01))
FCNModel.add(tf.keras.layers.Dense(256,activation='relu'))
FCNModel.add(tf.keras.layers.Dropout(0.01))
FCNModel.add(tf.keras.layers.Dense(128,activation='relu'))
FCNModel.add(tf.keras.layers.Dropout(0.01))
FCNModel.add(tf.keras.layers.Dense(cfg.MODEL_NUM_CLASSES, activation='softmax'))


CNNModel = tf.keras.Sequential()
CNNModel.add(tf.keras.layers.Convolution1D(64, 3, padding='same',activation="relu",input_shape=(cfg.IN_DIM, 1)))
CNNModel.add(tf.keras.layers.Convolution1D(64, 3, padding='same',activation="relu"))
CNNModel.add(tf.keras.layers.MaxPooling1D(pool_size=(2)))
CNNModel.add(tf.keras.layers.Convolution1D(128, 3, padding='same', activation="relu"))
CNNModel.add(tf.keras.layers.Convolution1D(128, 3, padding='same', activation="relu"))
CNNModel.add(tf.keras.layers.MaxPooling1D(pool_size=(2)))
CNNModel.add(tf.keras.layers.Flatten())
CNNModel.add(tf.keras.layers.Dense(128, activation="relu"))
CNNModel.add(tf.keras.layers.Dropout(0.5))
CNNModel.add(tf.keras.layers.Dense(cfg.MODEL_NUM_CLASSES, activation="softmax"))
