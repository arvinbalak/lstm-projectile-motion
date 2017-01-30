import tensorflow as tf
import numpy as np
import pandas
import matplotlib.pyplot as plt
import math as m

#HYPER_PARAMETERS
n_input = 2 #Dimension of element in the sequence. (x1, y1), which is 2
n_steps = 2 #Number of items in sequence. Here, it is 2. ((x1, y1), (x2,y2))
n_hidden_lstm = 32 #Number of hidden layers (memory cells) in LSTM. Value chosen by experimentation.
n_hidden_out = 2 #Dimension of input to hidden fully connected layer between LSTM and final output layer (Henceforth called FC_out)
n_classes = 2 # Number of nodes in output  layer
learning_rate = 0.0005 #learning rate of model
batch_size = 50 #Batches of data to  be split into when training
epoch = 5000 #number of training interations
error_threshold = 0.00009

#Placeholders for input/output to model
data = tf.placeholder(tf.float32, [None, 2, 2])
target = tf.placeholder(tf.float32, [None, 2])

### Prepare data shape to match tensorflow `RNN` function requirements
# Current data input shape: (batch_size, n_steps, n_input)
# Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
x = tf.transpose(data, [1, 0, 2]) # Transposing batch_size and n_steps
x = tf.reshape(x, [-1, n_input]) # Reshaping to (n_steps*batch_size, n_input)
x = tf.split(0, n_steps, x) # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)

### LSTM layer ###
lstm_cell = tf.nn.rnn_cell.LSTMCell(n_hidden_lstm, forget_bias=1.0) #single LSTM cell

output, states = tf.nn.dynamic_rnn(lstm_cell, data, dtype=tf.float32) #unroll LSTM and input data

# Only output from the last element in sequence is required for us. Extract it.
output = tf.transpose(output, [1, 0, 2]) 
output = tf.gather(output, int(output.get_shape()[0]) - 1)

### Fully Connected Layer, FC_out ### 
weight = tf.Variable(tf.truncated_normal([n_hidden_lstm, int(target.get_shape()[1])])) #weights of FC_out
bias = tf.Variable(tf.constant(0.1, shape=[target.get_shape()[1]])) #biases of FC_out

#Final predicted output (x3,y3)
pred = tf.matmul(output, weight) + bias

### Minimize error
cost = tf.reduce_sum(tf.squared_difference(pred, target, name=None))/(2*batch_size) #Cost function to minimize while training. Mean squared error
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
minimize = optimizer.minimize(cost)

### Load saved model and predict trajectory of a projectile launched at 45degree at 10m/s
# Assuming starting 2 coordinates are (0,0) and (0.707106781187 ,0.658106781187)
saver = tf.train.Saver()
save_path = 'model'
problem_input = [np.array([[0.0,0.0],[0.707106781187,0.658106781187]])]

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	#Load checkpoint/saved model
	ckpt = tf.train.get_checkpoint_state(save_path)
	if ckpt and ckpt.model_checkpoint_path:
		saver.restore(sess, ckpt.model_checkpoint_path)
	else:
		print "Saved model not found!"
	
	
	timestep_counter = 0
	output = np.array([[99,101]]) #dummy output for while loop
	predicted_trajectory = []	
	
	predicted_trajectory.append(problem_input[0][0].tolist()) #Add first coordinates to trajectory
	predicted_trajectory.append(problem_input[0][1].tolist()) #Add second coordinates to trajectory
	while timestep_counter < 100:
		output = sess.run(pred,{data: problem_input}) #feed 2 sequential coordinates and predict next one
		if output[0][1] > 0:
			predicted_trajectory.append(output[0].tolist()) #add to trajectory list
		else:
			break #if y goes below 0, break
		problem_input = [np.array([problem_input[0][1],output[0]])] #add next sequence as input and repeat loop
		timestep_counter = timestep_counter + 1
	
	#Print and plot predicted trajectory
	#print predicted_trajectory
	plt.plot(predicted_trajectory)
	plt.show()
	
	#Write to csv file
	predicted_trajectory_csv = open('predicted_trajectory.csv', 'w')
	count = 0
	for i in predicted_trajectory:
		print>>predicted_trajectory_csv, "%d,%f,%f" %(count,i[0], i[1])

sess.close()
