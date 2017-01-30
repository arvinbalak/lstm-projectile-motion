import tensorflow as tf
import numpy as np
import pandas

### MODEL INPUT OUTPUT FORMAT  ###
# The data in csv is of format (time_step,x_coord,y_coord) with each sequence of projectile motion starting at origin (0,0,0)
# The model takes 2 sequential coordinates ((x1, y1), (x2,y2)) and predicts the next coordinate (x3,y3) that would follow it in the projectile path. 
# The csv data is is processed into 2 sets - input_training_data and output_training_data. 
# Input for training the model contains a list of 2 sequential coordinates [((x1,y1), (x2,y2)), ((x2,y2), (x3,y3)) ... ((xn-2,yn-2), (xn-1,yn-1))] 
# Output training data contains the coordinate, that is expected to follow its corresponding input test sample. [(x3,y3),(x4,y4) ... (xn,yn)]
###

### PRE-PROCESSING dataset to create input_training_data and output_training_data.  ###
# lookback, means the number of sequential coordinates in a sample of the input_training_data.
# Eg. if lookback = 3, an input sample is ((x1,y1), (x2,y2), (x3,y3)) and its corresponding output sample is (x4,y4)'
# Returns input and output training data as required by the model
def create_dataset(dataset, look_back=2):
	dataX, dataY = [], []
	data_iterator = iter(range(len(dataset)-look_back-1))
	for i in data_iterator:
		# if output element is (0,0,0), its means the sequence data for a new projectile has started. So skip last 2 elements of current sequence
		if int(dataset[i + look_back,0]) == 0:
			next(data_iterator)
			next(data_iterator)
		else:
			a = dataset[i:(i+look_back)]
			dataX.append(np.array(a))
			dataY.append(dataset[i + look_back])
	return dataX, np.array(dataY)

### PARSE CSV
dataframe = pandas.read_csv('data/projectiles.csv', engine='python', usecols=[1,2])
dataset = dataframe.values
data_input,data_output = create_dataset(dataset) #get input and output training data

### Split into training data and model test data.
# Test data is used at the end of training the model, to gauge its accuracy on unknown inputs.
train_input, train_output = data_input[0:1000], data_output[0:1000]
test_input, test_output = data_input[1001:], data_output[1001:]

############# MODEL ################
# The given projectile data can be thought of as a sequence of (x,y) coordinates with a regular interval of 10ms between them. If the sequence can be predicted, the projectile path can be modeled. 
# RNN is the best architecture to handle sequence prediction, because of its ability to keep historical data in memory.
# The input, ((x1, y1), (x2,y2)), is given as input to an LSTM RNN layer.
# The output of the LSTM is transformed by a fully connected layer to give 2 values (x3,y3) that corresponds to the next coordinate in sequence.
# This predicted coordinate is again input to the model as ((x2,y2), (x3,y3)) which gives (x4,y4) - the next in sequence and so on, until y=0 is reached or 100 timesteps whichever is earlier.

#HYPER_PARAMETERS
n_input = 2 #Dimension of element in the sequence. (x1, y1), which is 2
n_steps = 2 #Number of items in sequence. Here, it is 2. ((x1, y1), (x2,y2))
n_hidden_lstm = 32 #Number of hidden layers (memory cells) in LSTM. Value chosen by experimentation.
n_hidden_out = 2 #Dimension of input to hidden fully connected layer between LSTM and final output layer (Henceforth called FC_out)
n_classes = 2 # Number of nodes in output  layer
learning_rate = 0.0005 #learning rate of model
batch_size = 50 #Batches of data to  be split into when training
no_of_batches = int(len(train_input)/batch_size) # Number of batches calculated from dataset size
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

### Start and run tensorflow session
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Training process
epoch_counter = 0
training_cost = 9999
while (training_cost > error_threshold and epoch_counter < epoch):
    ptr = 0
    for j in range(no_of_batches):
        inp, out = train_input[ptr:ptr+batch_size], train_output[ptr:ptr+batch_size] #split into batches
        ptr+=batch_size
        sess.run(minimize,{data: inp, target: out}) # Train model by minimizing error for each batch
	training_cost = sess.run(cost,{data: inp, target: out})
    print "Epoch - ",epoch_counter, "Error: {:3.5f}".format(training_cost)
    epoch_counter = epoch_counter + 1

# Test trained model with dataset set aside of testing
testdata_error = sess.run(cost,{data: test_input, target: test_output})
print('Epoch {:2d}: Error for test data is {:3.5f}'.format(epoch_counter + 1, testdata_error))

###### Save the session ######
save_path = 'model/LSTM_projectile'
saver = tf.train.Saver()
saver.save(sess, save_path)

sess.close()
