#### 1] Cheat 1 :- 
5. Most important concepts to remember
Term	            Simple meaning
-------------------------------------------------------------
FNN	              Feedforward Neural Network
Neuron	          Basic processing unit of a neural network
Epoch	            One complete pass through training data
Batch size	      Samples processed before a weight update
ReLU	            Activation  function used in hidden layers
Softmax	          Converts outputs into class probabilities
Optimizer	        Updates weights to reduce error
Loss	            Measures prediction error
Accuracy	        Measures correct predictions
Overfitting	      Model performs well on training data but poorly on unseen data
Flattening	      Converts 2D image into 1D array
Normalization	    Scales pixel values between 0 and 1



7. Viva Questions and Answers (Important)
Here are 35 questions that your teacher can ask, with simple answers you can memorize.
A. Basic questions
Q1. What is the aim of your practical?

To classify American Sign Language hand gestures into 24 classes using a Feedforward Neural Network.

Q2. What is deep learning?

Deep learning is a subset of machine learning that uses neural networks with multiple layers to learn patterns from data.

Q3. What is a neural network?

A neural network is a computational model inspired by the human brain that learns patterns using interconnected neurons.

Q4. What is FNN?

FNN stands for Feedforward Neural Network. Information flows in one direction, from input to output, without feedback loops.

Q5. Why is it called feedforward?

Because data moves forward through the network, from the input layer to the output layer.

Q6. What is TensorFlow?

TensorFlow is an open-source framework used for machine learning and deep learning.

Q7. What is Keras?

Keras is a high-level deep learning API used to build and train neural networks.

Q8. What is the difference between AI, ML and DL?
- AI: Making machines intelligent.
- ML: Machines learn from data.
- DL: Uses multilayer neural networks to learn from data.
B. Dataset-related questions
Q9. Which dataset is used in your practical?

Sign Language MNIST dataset.

Q10. What is the image size?

28 × 28 pixels.

Q11. How many input features are there?

784 features because 28 × 28 = 784.

Q12. Why are there only 24 classes?

Because letters J and Z are excluded from this dataset since they involve motion.

Q13. What is X_train?

It contains training input images represented by pixel values.

Q14. What is y_train?

It contains the correct labels corresponding to training images.

Q15. Why do we divide data into training and testing?

Training data teaches the model, while testing data checks its performance on unseen data.
C. Neural network questions
Q16. How many hidden layers are used in your model?

Three hidden layers containing 256, 128 and 64 neurons.

Q17. How many output neurons are used?

24 output neurons, one for each class.

Q18. What is a Dense layer?

A Dense layer is a fully connected layer in which every neuron connects to all neurons in the previous layer.

Q19. What is an activation function?

An activation function introduces non-linearity into a neural network so that it can learn complex patterns.

Q20. Why do we use ReLU?

ReLU is computationally simple and helps neural networks learn nonlinear relationships.

Q21. Why do we use Softmax?

Softmax converts the output values into probabilities for multiple-class classification.

Q22. What is the difference between ReLU and Softmax?

ReLU is generally used in hidden layers, while Softmax is used in the output layer for multiclass classification.

Q23. What are weights and biases?

Weights determine the importance of inputs, while biases help shift the output of neurons.

Q24. What is backpropagation?

Backpropagation calculates gradients of the loss with respect to the network parameters to help update weights and biases.
D. Training-related questions
Q25. What is an epoch?

One complete pass through the entire training dataset.

Q26. What is batch size?

The number of samples processed before updating model parameters.

Q27. What is an optimizer?

An optimizer updates weights and biases to minimize the loss function.

Q28. Which optimizer is used in your practical?

Adam optimizer.

Q29. Why do we use Adam?

Adam adapts learning rates for individual parameters and generally provides efficient training.

Q30. What is loss function?

A loss function measures the difference between actual output and predicted output.

Q31. Why do we use sparse categorical crossentropy?

Because our target labels are integer-encoded class numbers rather than one-hot encoded vectors.

Q32. What is normalization?

Normalization scales input values into a smaller range, in our practical from 0 to 1.

Q33. What is overfitting?

Overfitting occurs when a model learns training data very well but performs poorly on unseen data.

Q34. What is validation data?

Validation data is used during training to monitor model performance on data not used for weight updates.

Q35. What is the difference between validation and testing data?

Validation data monitors the model during training, whereas testing data evaluates its final performance.

7. Tricky viva questions your teacher might ask
These are slightly more advanced questions. Pay attention to them.
Question	                                        Answer
---------------------------------------------------------------------------------------------------------------------------------------------------
Can FNN work with images?	                        Yes, after flattening images into feature vectors.
Is FNN a CNN?	                                    No. FNN uses fully connected layers, whereas CNN uses convolutional layers.
Why not use CNN here?	                            FNN is simpler for demonstrating basic neural network classification, although CNN can better preserve spatial                                                     image features.
What happens if we remove ReLU?	                  The network may lose its ability to learn complex nonlinear relationships.
What does random_state=42 do?	                    It makes the random train-validation split reproducible.
Why use stratify?	                                It maintains the class distribution in the training and validation sets.
Can accuracy alone evaluate a model?	            No. Precision, recall, F1-score and confusion matrix can also be useful.
What is a parameter in a neural network?	        A trainable weight or bias.
What happens if epochs are increased too much?	  Training can take longer and may lead to overfitting.
What happens if we use 26 output neurons?	        The model would have extra output classes not present in this dataset.














#### 2] Cheat 2 :- 
No.	   Viva question	                                        Answer
-------------------------------------------------------------------------------------------------------------------------------------------------
1	     What is model.summary()?	                              It displays the neural network architecture.
2	     What is a trainable parameter?	                        A weight or bias that the model learns during training.
3	     What is history in model.fit()?	                      It stores training and validation metrics.
4	     What is history.history?	                              It contains accuracy and loss values for each epoch.
5	     What does [-1] mean?	                                  It selects the last element of a list or array.
6	     What is training accuracy?	                            Accuracy measured on training data.
7	     What is testing accuracy?	                            Accuracy measured on unseen testing data.
8	     What is validation accuracy?	                          Accuracy measured on validation data during training.
9	     What is training loss?	                                Prediction error on training data.
10	   What is validation loss?	                              Prediction error on validation data.
11	   Why do we plot graphs?	                                To understand model performance during training.
12	   What is subplot?	                                      It divides one figure into multiple graph areas.
13	   Why use plt.legend()?	                                To identify different lines in a graph.
14	   Why use plt.grid(True)?	                              To make graph values easier to read.
15	What is overfitting?	                                    When training performance is good but performance on unseen data is poor.












#### 3] Cheat 3 :- 
















#### 4] Cheat 4 :- 

















#### 5] Cheat 5 :- 




















#### 6] Cheat 6 :- 
















#### 7] Cheat 7 :- 


























#### 8] Cheat 8 :- 


























#### 9] Cheat 9 :- 





















#### 10] Cheat 10 :- 















#### 11] Cheat 11 :- 


















#### 12] Cheat 12 :- 













#### 13] Cheat 13 :- 























#### 14] Cheat 14 :- 





















#### 15] Cheat 15 :- 















#### 16] Cheat 16 :- 






















#### 17] Cheat 17 :- 




















#### 18] Cheat 18 :- 
















#### 19] Cheat 19 :- 





















#### 20] Cheat 20 :- 
























#### 21] Cheat 21 :- 




















#### 22] Cheat 22 :- 
























#### 23] Cheat 23 :- 

























#### 24] Cheat 24 :- 






















#### 25] Cheat 25 :- 





















#### 26] Cheat 26 :- 

























#### 27] Cheat 27 :- 
























#### 28] Cheat 28 :- 
























#### 29] Cheat 29 :- 




















#### 30] Cheat 30 :- 























#### 31] Cheat 31 :- 























#### 32] Cheat 32 :- 



















#### 33] Cheat 33:- 




















#### 34] Cheat 34 :- 



























