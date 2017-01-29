# lstm-projectile-motion
Created a model that predicts the trajectory of a projectile using LSTM

## model.py
Code for training the LSTM model on the data in data/projectiles.csv. All parameters are configured, including data file location. Run the script using command 'python model.py' to beging training a new model and saving it in folder /model. 

## predicted_trajectory.py
Code for loading saved LSTM model inside folder /model and using it to calculate the trajectory of a projectile launched at 45 degrees with an initial velocity of 10 m/s till it hits the ground or time_index=100 whichever is earlier. Assuming the initial two points in the trajectory to be :
0 ,0.0 ,0.0
1 ,0.707106781187 ,0.658106781187

All parameters are configure. Run the script using command 'python predicted_trajectory.py' to print predicteted trajectory, plot graph (and save to predicted_trajectory.png) and save trajectory to predicted_trajectory.csv.

## predicted_trajectory.csv
The predicted trajectory of a projectile launched at 45 degrees with an initial velocity of 10 m/s, in the same format as dataset csv

## predicted_trajectory.png
Plot of predicted trajectory of a projectile launched at 45 degrees with an initial velocity of 10 m/s. Ignore the blue straight line; it corresponds to the time component.

## training_output.txt
Console output of the training run of the checked in saved model.
