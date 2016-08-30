import tensorflow as tf 
#from tensorflow.models.rnn import rnn, rnn_cell
import numpy as np 
import dataset as ds
#Params
alpha = 1e-3
iterations = 100
batch_size = 10
display_step = 10

#Network params
input_size = 1
time_steps = 100
hidden_features = 512
output_classes = 256
