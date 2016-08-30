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
num_of_layers = 3

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
	outputs,state = seq2seq.basic_rnn_seq2seq(input_2,hidden_state,cells)	
	return tf.matmul(outputs[-1],weights['out'])+biases['out']
	
#Use our method to create our hypothesis
hypothesis = buildRNN(input_var,hidden_state,weights,biases)
hypothesis_index = tf.argmax(hypothesis,1)
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
acc_summary = tf.scalar_summary("Accuracy",accuracy)
saver = tf.train.Saver()
print("Getting Dataset...")
text = ds.getText()

print("Starting training...")
with tf.Session() as session:
	address = "/tmp/log/cooperbug/" + str(datetime.datetime.now()).replace(' ','')
	train_writer = tf.train.SummaryWriter(address,session.graph)
	proc = subprocess.Popen(["tensorboard","--logdir="+address])
	session.run(init)
	
	for i in range(iterations):
		batch_x = text[i+0:i+batch_size*time_steps*input_size]
		charlist = text[i+1:i+batch_size+1]
		batch_y = np.zeros((len(charlist),output_classes))
		for j in range(len(charlist)-1):
			batch_y[j][int(charlist[j])] = 1.0 
		#Reshape batch to input size
		batch_x = batch_x.reshape((batch_size,time_steps,input_size))
		#Run an interation of training
		session.run(optimizer,feed_dict={
			input_var:batch_x,
			y_var:batch_y
			})
		if i % display_step == 0:
			#Calculate Accuracy
			summary,acc = session.run([acc_summary,accuracy], feed_dict={
				input_var: batch_x,
				y_var: batch_y
				})
			print "Step: " + str(i) + ", Training Accuracy: " + str(acc)
			train_writer.add_summary(summary,i)
		if i % 100 == 0 and not(i==0):
			seq = ''
			x_inp = batch_x
			for j in range(140):
				index = hypothesis_index.eval({
					input_var: x_inp,
					y_var: batch_y
				})
				next_letter = unichr(index[0])
				x_inp = text[i+0+1+j:i+batch_size*time_steps*input_size+1+j]
				x_inp[-1] = float(ord(next_letter))
				x_inp = x_inp.reshape((batch_size,time_steps,input_size))
				seq += next_letter
			f = open('seq_gen_iter' + str(i) + '.txt', 'w+')
			print "Sequence:\n" +seq
			f.write(seq)
			f.close()
		if i % 1000 == 0 and not(i == 0):
			saver.save(session,"/home/josh/Documents/Cooperbot/model_iter" +str(i) +".ckpt")
			
			
	print "Training is COMPLETE!"
	x_test = text[i:i+batch_size]
	y_test = text[i*batch_size+1:i*2*batch_size]
	test_accuracy = session.run(accuracy,feed_dict={
		input_var: batch_x,
		y_var: batch_y,
		})
	print ("Final test accuracy: %g" %(test_accuracy))
	saver.save(session,"/home/josh/Documents/Cooperbot/model.ckpt")
	proc.kill()
