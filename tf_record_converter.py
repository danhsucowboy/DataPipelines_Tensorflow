import tensorflow as tf
import numpy as np 
import pandas as pd
import imageio
import os
import re
import cv2
import sys
import random
import logging as log

from cv2 import copyMakeBorder, IMREAD_GRAYSCALE

# Helperfunctions to make your feature definition more readable
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def sorted_aphanumeric(data):
		convert = lambda text: int(text) if text.isdigit() else text.lower()
		alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
		return sorted(data, key=alphanum_key)

def load_data(training_path,target_path,index_path,input_shape,input_hist,shuffle,nbins,blur,blur_kernel_shape,sigma):
	img_list = list()
	target_list = list()
	hist_map_list = list()
	hist_list = list()
	training_files = sorted_aphanumeric(os.listdir(training_path))
	target_files = sorted_aphanumeric(os.listdir(target_path))
	bin_range = np.linspace(0, 255,nbins)
	#input_shape = 700

	sigma = sigma
	blur_kernel_size = (blur_kernel_shape,blur_kernel_shape)

	if len(training_files)==len(target_files):
		num = len(training_files)
	else: return
	
	for i in range(num):
	
		img = imageio.imread(training_path+'/'+training_files[i], as_gray=False,pilmode='L').reshape(input_shape,input_shape,1).astype('float32')
		target = imageio.imread(target_path+'/'+target_files[i], as_gray=False,pilmode='L').reshape(input_shape,input_shape,1).astype('float32')/255

		if input_hist:
			hist = cv2.calcHist([img],[0],None,[nbins],[0,256])
			hist_norm = hist.ravel()/np.sum(hist)
			hist_map = np.zeros(img.shape)
			hist_list.append(hist_norm)
			for i in range(nbins):
				if i==nbins-1:
					hist_map[img>=bin_range[i]]=hist_norm[i]
				else:
					hist_map[np.logical_and(img>=bin_range[i],img<bin_range[i+1])]=hist_norm[i]
			hist_map = (hist_map-np.min(hist_map))/(np.max(hist_map)-np.min(hist_map))
			hist_map_list.append(hist_map)

		if blur:
			img_filter_list = list()
			img_filter_list.append(((img-np.min(img))/(np.max(img)-np.min(img))).reshape(input_shape,input_shape))
			for s in sigma:
				tmp = cv2.GaussianBlur(img.reshape(input_shape,input_shape),blur_kernel_size, s)
				tmp = (tmp-np.min(tmp))/(np.max(tmp)-np.min(tmp))
				img_filter_list.append(tmp)
			img_filter_list = np.transpose(np.array(img_filter_list),[1,2,0])
			img_list.append(img_filter_list)
		else:
			img = (img-np.min(img))/(np.max(img)-np.min(img))
			img_list.append(img)
		target_list.append(target)

	img_list = np.array(img_list)
	target_list = np.array(target_list)

	name_list = [x for x in range(num)]

	index = [x for x in range(num)]
	if shuffle:
		random.shuffle(index)
		pd.DataFrame(index).to_csv(index_path)
	else:
		index = pd.read_csv(index_path,sep=',',names =['x','y']).as_matrix()[1:,1].astype('int')

	X = np.array(img_list)[index]
	y = np.array(target_list)[index]	
	name = np.array(training_files)[index]

	if input_hist:
		hist_list = np.array(hist_list)[index]
		hist_map_list = np.array(hist_map_list)[index]
		X = np.concatenate([X,hist_map_list],axis=-1)
		return X.astype('float32'),y,hist_list
	return X,y,None

if __name__ == "__main__":
	log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)

	source_path = "D:/[5]DeepLearning/Images/Original"
	target_path = "D:/[5]DeepLearning/Images/GroundTruth"
	index_path = "D:/[5]DeepLearning/[1]Code/[1]Keras/Datapipline/data/all_index.csv"
	FILEPATH = "D:/[5]DeepLearning/[5]DataPipline/test.tfrecords"
	# load data in numpy
	# image = source
	# label = target
	log.info('Load data')
	image,label,hist_list = load_data(training_path=source_path,target_path=target_path,index_path=index_path,input_shape=500,input_hist=False ,shuffle=True,nbins=256,blur=False,blur_kernel_shape=5,sigma=[1,2,3])
	print('image shape: '+str(image.shape)+', label shape: '+str(label.shape))
	log.info('image shape: {}, label shape:{}'.format(image.shape,label.shape))
	# create filewriter
	writer = tf.io.TFRecordWriter(FILEPATH)
	log.info('Setting TF Writer')

	# Define the features of your tfrecord
	feature = {'image':  _bytes_feature(tf.compat.as_bytes(image.tostring())),
			   'label':  _bytes_feature(tf.compat.as_bytes(label.tostring()))}


	# Serialize to string and write to file
	example = tf.train.Example(features=tf.train.Features(feature=feature))
	writer.write(example.SerializeToString())
	log.info("Write to .tfrecord")
