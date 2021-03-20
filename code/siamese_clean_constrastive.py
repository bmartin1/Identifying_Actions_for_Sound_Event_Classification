from __future__ import absolute_import
from __future__ import print_function
import numpy as np
from sklearn.preprocessing import normalize
from keras import optimizers
import pandas as pd
import matplotlib
import collections

import random
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout, Lambda
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from keras import utils
from keras.constraints import unit_norm
from keras.layers.normalization import BatchNormalization
from keras import losses
from keras.models import load_model
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
import os
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import Normalizer

# Sets GPU to use
os.environ["CUDA_VISIBLE_DEVICES"]="0"


# Calculates the Distance between two vectors
def euclidean_distance(vects):
	x, y = vects
	return K.exp(-K.sum(K.square(x - y), axis=1, keepdims=True))

# Splits up vectors and returns the output shape
def eucl_dist_output_shape(shapes):
	shape1, shape2 = shapes
	return (shape1[0], 1)

# Contrastive loss function
def contrastive_loss(y_true, y_pred):
	margin = 0.9
	return (1-y_true)*K.square(y_pred)*0.5 + y_true*K.square(K.maximum(margin - y_pred, 0))*0.5


# Stretches out each AV to size 6144 to match shape of AE
def resize_actionvectors(av):
	resized_avs = []
	for v in av:
		resized_vector = []
		for i in range(20):
			resized_vector += [v[i]] * 307
		resized_avs.append(resized_vector + [0,0,0,0])
	resized_avs = np.array(resized_avs)

	return resized_avs


# Creates training pairs for training siamese net
def create_pairsALT(ae, av):

	length = len(ae)
	positives = np.array(list(zip(ae,av)))
	negatives = []

	for i in range(length):
		rand_index = random.randint(0,length-1)
		rand_av = av[rand_index]
		negatives.append([ae[i],rand_av])

	negatives = np.array(negatives)
	X = np.concatenate((positives, negatives))
	Y = [1]*length + [0]*length
	Y = np.array(Y)
	assert(len(X) == len(Y))

	return (X,Y)


# Siamese network to be shared by inputs
def create_base_network(input_shape):

	input = Input(shape=input_shape)
	x = Dense(1024, activation='relu')(input)
	x = Dropout(0.4)(x)
	x = Dense(512, activation='relu')(x)
	x = Dropout(0.4)(x)
	x = Dense(512, activation='relu')(x)
	x = Dropout(0.4)(x)
	x = Dense(1024)(x)

	return Model(input, x)


# Preprocess and imports data
def preprocess_data(av_filename, ae_filename):
	# Action Vectors IMPORTED AND SCALED TO 0-1
	training_data = pd.read_csv(av_filename)
	audio_files = training_data['filename']
	sec = training_data['target']
	training_data = training_data.drop(columns=['fold', 'filename', 'category', 'target'])
	av = training_data.to_numpy()
	actionvectors = np.divide(av, 12.0)

	le_sec = LabelEncoder()
	le_sec.fit(sec)
	sec = le_sec.transform(sec)
	sec = to_categorical(sec)

	embeddings_dict = {}
	raw_embeddings = np.load(ae_filename, allow_pickle=True)
	for i in range(len(raw_embeddings)):
		fname = raw_embeddings[i][0]
		embedding = raw_embeddings[i][1]
		embedding = embedding.mean(axis=0)
		embeddings_dict[fname] = embedding

	x_train = []
	for i in range(len(audio_files)):
		fname = audio_files[i]
		embedding = embeddings_dict[fname]
		x_train.append(embedding)
	audio_embeddings = np.array(x_train)

	return (actionvectors, sec, audio_embeddings)


# Imports and pre-processes data
actionvectors, sec, audio_embeddings = preprocess_data('actionvector_one_per_audiofile.csv', 'ESC-50_openl3_music_mel256_6144.npy')




joint_embeddings = []
# Iterates over each fold of ESC-50 dataset
for fold in range(5):

	# Creates train / test split
	audio_embeddings_train = np.concatenate((audio_embeddings[0:fold*400], audio_embeddings[(fold+1)*400:]))
	actionvectors_train = np.concatenate((actionvectors[0:fold*400], actionvectors[(fold+1)*400:]))
	sec_train = np.concatenate((sec[0:fold*400], sec[(fold+1)*400:]))

	audio_embeddings_test = audio_embeddings[fold*400:(fold+1)*400]
	actionvectors_test = actionvectors[fold*400:(fold+1)*400]
	sec_test = sec[fold*400:(fold+1)*400]

	# Scale AVs up to 6144 dim by stretching each one
	actionvectors_train = resize_actionvectors(actionvectors_train)
	actionvectors_test = resize_actionvectors(actionvectors_test)



	# Scale AE
	scaler = StandardScaler().fit(audio_embeddings_train)
	audio_embeddings_train = scaler.transform(audio_embeddings_train)
	audio_embeddings_test = scaler.transform(audio_embeddings_test)
	# Normalize AE
	normalize = Normalizer(norm='l2').fit(audio_embeddings_train)
	audio_embeddings_train = normalize.transform(audio_embeddings_train)
	audio_embeddings_test = normalize.transform(audio_embeddings_test)
	# Scale AV
	scaler = StandardScaler().fit(actionvectors_train)
	actionvectors_train = scaler.transform(actionvectors_train)
	actionvectors_test = scaler.transform(actionvectors_test)
	# Normalize AV
	normalize = Normalizer(norm='l2').fit(actionvectors_train)
	actionvectors_train = normalize.transform(actionvectors_train)
	actionvectors_test = normalize.transform(actionvectors_test)

	# Creates train / test data for siamese network input
	(x_train_pairs, y_train) = create_pairsALT(audio_embeddings_train, actionvectors_train)
	(x_test_pairs, y_test) = create_pairsALT(audio_embeddings_test, actionvectors_test)


	# Builds and trains siamese net for embedding similarity task
	input_shape = x_train_pairs[0][0].shape
	base_network = create_base_network(input_shape)


	# Put together trainable siamese model
	input_a = Input(shape=input_shape)
	input_b = Input(shape=input_shape)
	processed_a = base_network(input_a)
	processed_b = base_network(input_b)
	distance = Lambda(euclidean_distance,
					  output_shape=eucl_dist_output_shape)([processed_a, processed_b])
	model = Model([input_a, input_b], distance)

	# Compile and Run training on siamese model
	adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
	model.compile(loss=contrastive_loss, optimizer=adam, metrics=['accuracy'])
	model.summary()
	model.fit([x_train_pairs[:,0], x_train_pairs[:,1]], y_train, batch_size=64, epochs=100, validation_data=([x_test_pairs[:, 0], x_test_pairs[:, 1]], y_test))



	# Generate joint embeddings from trained model
	joint_embeddings_x_train_ae = base_network.predict(audio_embeddings_train)
	joint_embeddings_x_test_ae = base_network.predict(audio_embeddings_test)

	# Scale JE
	scaler = StandardScaler().fit(joint_embeddings_x_train_ae)
	joint_embeddings_x_train_ae = scaler.transform(joint_embeddings_x_train_ae)
	joint_embeddings_x_test_ae = scaler.transform(joint_embeddings_x_test_ae)
	# Normalize JE
	normalize = Normalizer(norm='l2').fit(joint_embeddings_x_train_ae)
	joint_embeddings_x_train_ae = normalize.transform(joint_embeddings_x_train_ae)
	joint_embeddings_x_test_ae = normalize.transform(joint_embeddings_x_test_ae)

	joint_embeddings.append(joint_embeddings_x_test_ae)


	# Builds model for SEC prediction
	modelf = Sequential()
	modelf.add(Dense(units=800,input_dim = 1024))
	modelf.add(BatchNormalization())
	modelf.add(Activation('relu'))
	modelf.add(Dropout(0.5))
	modelf.add(Dense(units=500))
	modelf.add(BatchNormalization())
	modelf.add(Activation('relu'))
	modelf.add(Dropout(0.5))
	modelf.add(Dense(units=200))
	modelf.add(BatchNormalization())
	modelf.add(Activation('relu'))
	modelf.add(Dropout(0.5))
	modelf.add(Dense(units=50, activation='softmax'))

	# Complies and trains model
	modelf.compile(loss="categorical_crossentropy", optimizer='sgd', metrics=['accuracy'])
	modelf.summary()
	modelf.fit(joint_embeddings_x_train_ae,	sec_train, batch_size=32, epochs=100, validation_data=(joint_embeddings_x_test_ae, sec_test))



joint_embeddings = np.concatenate(joint_embeddings)
np.save('joint_embeddings.npy', joint_embeddings)
