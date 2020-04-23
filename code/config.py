# ----------------------------------------
# Written by Xiao Feng
# ----------------------------------------
import argparse
import os
import sys

class Configuration():
	def __init__(self):
		self.ROOT_DIR = os.getcwd()
		self.IN_DIM = 61
		self.MODEL_NAME ='LSTMModel' #'LSTMModel' #'CNNModel' #'FCNModel'
		self.MODEL_NUM_CLASSES = 5
		self.MODEL_SAVE_DIR = os.path.join(self.ROOT_DIR,self.MODEL_NAME,'model')

		self.TRAIN_BATCHES = 128
		self.TRAIN_MINEPOCH = 0
		self.TRAIN_EPOCHS = 5

		self.DATA = '/content/sample_data/event.csv'

		self.OPTIMIZER = 'adam'
		self.LOSSFUNCTION = 'categorical_crossentropy'

		self.LOG_DIR = os.path.join(self.ROOT_DIR,self.MODEL_NAME,'log')

		self.__check()

	def __check(self):
		if not os.path.isdir(self.LOG_DIR):
			os.makedirs(self.LOG_DIR)
		if not os.path.isdir(self.MODEL_SAVE_DIR):
			os.makedirs(self.MODEL_SAVE_DIR)



cfg = Configuration()
