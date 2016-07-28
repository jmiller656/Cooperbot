'''
This file is intended to become an RNN
that will be trained with our anderson
cooper dataset from CNN
'''
import tensorflow as tf 
#from tensorflow.models.rnn import rnn, rnn_cell
import numpy as np 
import dataset as ds
#Params
alpha = 1e-3
iterations = 10000
batch_size = 10000
display_step = 10

#Network params
input_size = 10
time_steps = 10
hidden_features = 128
output_classes = 256

'''
Defining placeholders for the RNN
'''
#Placeholder for our input data
input_var = tf.placeholder("float",[None,time_steps,input_size])
#Placeholder for input into hidden layer
hidden_state = tf.placeholder("float",[None,2*hidden_features],name="Hidden")
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
	input_ = tf.transpose(input_var,[1,1,2])
	input_ = tf.reshape(input_var,[-1,input_size])
	input_ = tf.matmul(input_,weights['hidden']) + biases['hidden']
	lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_features,forget_bias=1.0,state_is_tuple=True)
	input_2 = tf.split(0,time_steps,input_)
	outputs,states = tf.nn.rnn(lstm_cell,input_2,dtype=tf.float32)
	return tf.matmul(outputs[-1],weights['out'])+biases['out']

#Use our method to create our hypothesis
hypothesis = buildRNN(input_var,hidden_state,weights,biases)

'''
Training section
'''
#Define our cost and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(hypothesis,y_var))
optimizer = tf.train.AdamOptimizer(learning_rate=alpha).minimize(cost)

#Define our model evaluator
correct_prediction = tf.equal(tf.argmax(hypothesis,1),tf.argmax(y_var,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

#Initializing all variables
init = tf.initialize_all_variables()

text = ds.getText()
a = 0
batch_x = text[0:batch_size]
charlist = text[1:batch_size+1]
print("Starting training...")
with tf.Session() as session:
	session.run(init)
	for i in range(iterations):
		batch_x = text[a:a+batch_size*time_steps*input_size]
		charlist = text[a*batch_size+1:a*batch_size+batch_size+1]
		batch_y = np.zeros((len(charlist),output_classes))
		for j in range(len(charlist)-1):
			batch_y[j][int(charlist[j])] = 1.0 
		#Reshape batch to input size
		batch_x = batch_x.reshape((batch_size,time_steps,input_size))
		#Run an interation of training
		session.run(optimizer,feed_dict={
			input_var:batch_x,
			y_var:batch_y,
			hidden_state: np.zeros((batch_size,2*hidden_features))
			})
		if i % display_step == 0:
			#Calculate Accuracy
			acc = session.run(accuracy, feed_dict={
				input_var: batch_x,
				y_var: batch_y,
				hidden_state: np.zeros((batch_size,2*hidden_features))
				})
			print "Step: " + str(i) + ", Training Accuracy: " + str(acc)
	print "Training is COMPLETE!"
	x_test = text[i:i+batch_size]
	y_test = text[i*batch_size+1:i*2*batch_size]
	test_accuracy = session.run(accuracy,feed_dict={
		input_var: test_x,
		y_var: test_y,
		hidden_state: np.zeros((batch_size,2*hidden_features))
		})
	print ("Final test accuracy: " + test_accuracy)
