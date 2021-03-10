
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Activation, Dropout
from keras import optimizers

import tensorflow as tf
import numpy as np
import pandas as pd

from sklearn.metrics import classification_report
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer

# Sets GPU to use
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"


#############
# Terminology
# AV = Action Vector
# AE = Audio Embedding
#############



# Matches each AE to its corresponding AV
# Reduces dimensionality of AEs
def preprocessData(av_filename, ae_filename):

	av_training_data = pd.read_csv(av_filename) # AV CSV
	categories = av_training_data['category']
	audio_files = av_training_data['filename']
	av_training_data = av_training_data.drop(columns = ['filename',
													'fold',
													'target',
													'category'])
	x_actionvectors = av_training_data.to_numpy(dtype='float') # AV data


	# Converts string class-names to numerical(integer) classes
	le = LabelEncoder()
	le.fit(categories)
	y_train = le.transform(categories)

	# One-Hot Encodes each integer class
	lb = LabelBinarizer()
	lb.fit(y_train)
	y_train = lb.transform(y_train)

	embeddings_dict = {} # Audio filename to embedding mapping
	raw_embeddings = np.load(ae_filename, allow_pickle=True) # Imports AEs

	# Iterates through AEs
	# Creates filename to embedding table
	# Takes average across time for each embedding
	for i in range(len(raw_embeddings)):
		fname = raw_embeddings[i][0]
		embedding = raw_embeddings[i][1]
		embeddings_dict[fname] = embedding.mean(axis=0) # Takes average across timesteps

	# Concatenates corresponding AV to each AE
	x_train_stacked = []
	for i in range(len(audio_files)):
		fname = audio_files[i] # Audio filename
		av = np.array(x_actionvectors[i]) #corresponding AV
		embedding = embeddings_dict[fname]
		x_train_stacked.append(np.concatenate((embedding,av)))
		
	# Converts to Numpy Array
	x_train_stacked = np.array(x_train_stacked)
	return (x_train_stacked, y_train, le)




def train_model(all_x_train, all_y_train, le):
	# Will collect statistics
	cf_report = []
	accuracy = []

	# Iterate over 5 folds
	# Folds designated by ESC50 dataset
	for fold in range(5):
		print('')



		# Splits data into 80/20 train/test sections based on fold
		x_train = np.concatenate((all_x_train[0:fold*400],
								all_x_train[(fold+1)*400:]))
		x_test = all_x_train[fold*400:(fold+1)*400]
		y_train = np.concatenate((all_y_train[0:fold*400],
								all_y_train[(fold+1)*400:]))
		y_test = all_y_train[fold*400:(fold+1)*400]

		# Normalize the data
		normalize = Normalizer(norm='l2').fit(x_train)
		x_train = normalize.transform(x_train)
		x_test = normalize.transform(x_test)




		# Setting up model structure
		model = Sequential()
		model.add(Dense(units=800,input_dim=len(x_train[0])))
		model.add(BatchNormalization())
		model.add(Activation('tanh'))
		model.add(Dropout(0.5))
		model.add(Dense(units=500))
		model.add(BatchNormalization())
		model.add(Activation('tanh'))
		model.add(Dropout(0.5))
		model.add(Dense(units=200))
		model.add(BatchNormalization())
		model.add(Activation('tanh'))
		model.add(Dropout(0.5))
		model.add(Dense(units=50, activation='softmax'))

		opt = optimizers.SGD(lr=0.008)
		model.compile(loss='categorical_crossentropy',
							optimizer=opt,
							metrics=['accuracy'])
		model.fit(x_train,
					y_train,
					epochs=70,
					batch_size=32)


		_, test_accuracy = model.evaluate(x_test, y_test)
		accuracy.append(test_accuracy)

		# Setting up classification_report
		Y_test = np.argmax(y_test,axis=1)
		y_pred = model.predict_classes(x_test)
		cf_report.append(classification_report(Y_test, y_pred))





	# Prints a Classification Report for each fold
	print("Classification Reports")
	for r in cf_report:
		print(r)

	# Prints accuracy for each fold
	print("Accuracy per FOLD:")
	for a in range(len(accuracy)):
		print("Fold " + str(a) + " Accuracy:", accuracy[a])

	# Prints aggregate statistics
	print('')
	print("Avg Accuracy:", sum(accuracy) / 5)
	print("Standard Deviation:", np.std(accuracy))







# Imports, modifies, and stacks AE to AV features
av_filename = '../embeddings_fifty/actionvector_one_per_audiofile.csv'
ae_filename = '../embeddings_fifty/ESC-50_openl3_music_mel256_6144.npy'
(all_x_train, all_y_train, le) = preprocessData(av_filename, ae_filename)

# Runs training on model
train_model(all_x_train, all_y_train, le)
