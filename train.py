# ----------------------------------------
# Written by Xiao Feng
# ----------------------------------------
import os
import sys
import tensorflow as tf
import numpy as np
from config import cfg
from model import LSTMModel, FCNModel, CNNModel
from utils import *
import matplotlib.pyplot as plt

def train(model):
    X_train,label_train,X_test, label_test, label_true= data_load_process(cfg.DATA)

    model.compile(loss=cfg.LOSSFUNCTION,
                  optimizer=cfg.OPTIMIZER,
                  metrics=['accuracy'])
    checkpointer = tf.keras.callbacks.ModelCheckpoint(
        filepath=cfg.MODEL_SAVE_DIR+"/checkpoint-{epoch:02d}.hdf5",
        verbose=1,
        save_best_only=True,
        monitor='val_accuracy',
        mode='max')
    csv_logger = tf.keras.callbacks.CSVLogger(
        (cfg.LOG_DIR+"/"+cfg.MODEL_NAME+"_training_set_iranalysis.csv"),
        separator=',',
        append=False)

    history = model.fit(X_train, label_train,
        batch_size=cfg.TRAIN_BATCHES,
        epochs=cfg.TRAIN_EPOCHS,
        validation_data=(X_test, label_test),
        callbacks=[checkpointer, csv_logger])

    model.save(cfg.MODEL_SAVE_DIR+"/"+cfg.MODEL_NAME+"_model.hdf5")

    loss, accuracy = model.evaluate(X_test, label_test)
    print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))
    # y_pred = model.predict_classes(X_test)
    y_pred = np.argmax(model.predict(X_test), axis=-1)
    np.savetxt((cfg.LOG_DIR+'/'+cfg.MODEL_NAME+".txt"),np.hstack((label_true.reshape(-1, 1), y_pred.reshape(-1, 1))), fmt='%01d')

    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(cfg.LOG_DIR+"/"+cfg.MODEL_NAME+"_accuracy.png")

    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(cfg.LOG_DIR+"/"+cfg.MODEL_NAME+"_loss.png")
    print("Finish Modeling")

def test(model):
    _,_,X_test, label_test,label_true= data_load_process(cfg.DATA)
    model.load_weights(cfg.MODEL_SAVE_DIR+"/"+cfg.MODEL_NAME+"_model.hdf5")
    y_pred = cnn.predict_classes(X_test)
    np.savetxt((cfg.LOG_DIR+'/'+cfg.MODEL_NAME+".txt"),np.hstack((label_true.reshape(-1, 1), y_pred.reshape(-1, 1))), fmt='%01d')
    model.compile(loss=cfg.LOSSFUNCTION,
                  optimizer=cfg.OPTIMIZER,
                  metrics=['accuracy'])
    loss, accuracy = model.evaluate(X_test, label_test)
    print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))


if __name__ == '__main__':
    if cfg.MODEL_NAME == 'LSTMModel':
        train(LSTMModel)
    elif cfg.MODEL_NAME =='CNNModel':
        train(CNNModel)
    elif cfg.MODEL_NAME == "FCNModel":
        train(FCNModel)
