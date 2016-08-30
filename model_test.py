'''
This file is intended to become an RNN
that will be trained with our anderson
cooper dataset from CNN
'''
import tensorflow as tf 
#from tensorflow.models.rnn import rnn, rnn_cell
import numpy as np 
import dataset as ds
import datetime
import subprocess
from tensorflow.python.ops import seq2seq

#Params
alpha = 1e-3
iterations = 10000
batch_size = 100
display_step = 10

#Network params
input_size = 1
time_steps = 10
hidden_features = 256
output_classes = 127
num_of_layers = 1

'''
Defining placeholders for the RNN
'''
#Placeholder for our input data
input_var = tf.placeholder("float",[None,time_steps,input_size])
#Placeholder for input into hidden layer
hidden_state = tf.placeholder("float",[None,hidden_features],name="Hidden")
#Placeholder for our 'correct' y values
y_var = tf.placeholder("float",[None,output_classes],name="Output")

'''
Definining weight variables for the RNN
'''
#Weights for hidden layer
W_hidden = tf.Variable(tf.random_normal([input_size,hidden_features]))
#Weights from hidden to output layer
W_out = tf.Variable(tf.random_normal([hidden_features,output_classes]))

#Dictionary of weights for easy access
weights = {
	'hidden': W_hidden,
	'out': W_out
}

'''
Defining bias variables for the RNN
'''
#Bias for the hidden Layer
b_hidden = tf.Variable(tf.random_normal([hidden_features]))
#Bias for output layer
b_output = tf.Variable(tf.random_normal([output_classes]))


#Dictionary of biases for easy access
biases = {
	'hidden': b_hidden,
	'out': b_output
}

'''
Define a method that creates an RNN with our current parameters,
weights, and biases
'''
def buildRNN(input_var,hidden_state,weights,biases):
	input_ = tf.reshape(input_var,[-1,input_size])	
	#lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_features,forget_bias=1.0,state_is_tuple=True)
	lstm_cell = tf.nn.rnn_cell.GRUCell(hidden_features)
	input_2 = tf.split(0,time_steps,input_)	
	cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell]*num_of_layers,state_is_tuple=True)
	#outputs,hidden_state = tf.nn.rnn(cells,input_2,dtype=tf.float32)
	hidden_state = cells.zero_state(batch_size,tf.float32)
	outputs,state = seq2seq.rnn_decoder(input_2,hidden_state,cells)	
	return tf.matmul(outputs[-1],weights['out'])+biases['out']
	
#Use our method to create our hypothesis
hypothesis = buildRNN(input_var,hidden_state,weights,biases)
temp = 0.1
hypothesis_index = tf.argmax(hypothesis/temp,1)
saver = tf.train.Saver()
with tf.Session() as session:
	saver.restore(session,"/home/josh/Documents/Cooperbot/model.ckpt")
	words = "T"*time_steps
	words = list(words)
	text = list("T"*time_steps)
	for i in range(len(words)):
		sys.stdout.write('')
	for j in range(10000):
		for i in range(len(words)):
			text[i] = ord(words[i])
		text = np.asarray(text).astype(np.float32,copy=True)
		index = hypothesis_index.eval({
			input_var: text.reshape(1,time_steps,input_size),
			y_var: np.zeros((1,output_classes))
		})
		next_letter = unichr(index[0])
		words.pop(0)
		words.append(next_letter)
		sys.stdout.write(next_letter)
		if (not(j==0) and (j%100==0)):
			print "\n"
		
		
print "\n"		
