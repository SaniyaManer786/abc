from pathlib import Path

md = r"""# Deep Learning Practicals – Viva Questions and Answers

This guide is prepared from the 30 Jupyter notebooks in the uploaded `DL.zip` archive. The sections follow the practical/file numbering and the actual models, datasets, preprocessing steps, and training settings found in those notebooks.

> **How to use this guide:** Learn the aim, dataset, model architecture, loss function, optimizer, and evaluation method for each practical. Where two notebooks cover similar work, their differences are called out rather than repeating every general definition.

---

## Practical 1 – `Copy_of_DL_1practicalFNN.ipynb`
### American Sign Language classification using a Feedforward Neural Network

**Q1. What is the aim of this practical?**  
To classify American Sign Language hand-gesture images into alphabet classes using a Feedforward Neural Network (FNN).

**Q2. What is an FNN?**  
A Feedforward Neural Network is an artificial neural network in which information moves from the input layer through hidden layers to the output layer. It has no feedback connections.

**Q3. What is the input data in this practical?**  
Each Sign Language MNIST image is a 28 × 28 grayscale image. It contains 784 pixel values when flattened.

**Q4. Why are images flattened?**  
A Dense layer expects a one-dimensional feature vector. Flattening converts a 28 × 28 image into 784 values.

**Q5. Why are pixel values normalized?**  
Dividing pixel values by 255 scales them from 0–255 to 0–1. This usually makes neural-network training more stable.

**Q6. What is the role of the Dense layer?**  
Every neuron in a Dense layer is connected to all outputs of the previous layer. It learns combinations of input features.

**Q7. Why is Softmax used in the output layer?**  
Softmax converts output scores into class probabilities. The class with the highest probability is selected as the prediction.

**Q8. What is the difference between training and testing data?**  
Training data is used to learn model weights. Testing data is held aside to measure performance on unseen examples.

---

## Practical 2 – `DL_2Practical_FNN.ipynb`
### Sign Language alphabet recognition using an FNN

**Q1. What is the purpose of this practical?**  
To construct, train, and evaluate an FNN for recognizing Sign Language MNIST alphabet gestures, and display accuracy and loss.

**Q2. Why does the code use `drop('label', axis=1)`?**  
The `label` column contains the correct class. It is separated from the pixel columns so that the pixels become input features (`X`) and the label becomes the target (`y`).

**Q3. Why are the original labels mapped to consecutive values?**  
The dataset labels may not be consecutive because some letters are omitted. The mapping converts the existing labels into 0–23, which matches the 24 output classes.

**Q4. What does `stratify=y_train` do in `train_test_split`?**  
It preserves approximately the same class proportions in the training and validation subsets.

**Q5. What is the architecture in this practical?**  
The notebook uses an input of 784 values, Dense hidden layers with 256, 128, and 64 neurons using ReLU, and a 24-neuron Softmax output layer.

**Q6. Why is `sparse_categorical_crossentropy` used?**  
The target labels are integer class IDs, not one-hot vectors. Sparse categorical crossentropy is designed for integer labels in multi-class classification.

**Q7. What do `epochs=15` and `batch_size=128` mean?**  
The model passes through the training data 15 times. It updates its weights after processing batches of 128 examples.

**Q8. What is stored in `history`?**  
The object returned by `model.fit()` stores training and validation metrics, such as accuracy and loss, for each epoch.

**Q9. How are predicted classes obtained?**  
`model.predict()` returns class scores/probabilities. `np.argmax(..., axis=1)` selects the index with the largest score for each example.

---

## Practical 3 – `DL_Pract_3FNN.ipynb`
### FNN for grayscale Sign Language images

**Q1. What is the main task?**  
To classify 28 × 28 grayscale Sign Language images into 24 alphabet classes using an FNN and display five actual-versus-predicted examples.

**Q2. Why is the input shape `(784,)`?**  
A 28 × 28 image contains 784 pixels, and the FNN receives the flattened pixels as a one-dimensional vector.

**Q3. Why is `reshape(-1, 784)` used?**  
It changes each image into a vector of 784 values. `-1` tells NumPy to calculate the number of images automatically.

**Q4. What is the FNN architecture?**  
The model has Dense layers of 256, 128, and 64 neurons with ReLU activation, followed by 24 Softmax output neurons.

**Q5. Why does this notebook use a validation split?**  
The training data is split so that validation performance can be monitored during training without using the final test set for that purpose.

**Q6. What does `chr(classes[label] + 65)` do?**  
It converts a numeric alphabet code into a character using its ASCII/Unicode code point. The `classes` array is used to map the model's consecutive class index back to the original label.

**Q7. Why is `cmap='gray'` used in `imshow()`?**  
It displays the image using a grayscale colour map, which is appropriate for the single-channel hand-gesture images.

**Q8. How does the notebook show whether a prediction is correct?**  
It displays the actual alphabet and the predicted alphabet for each selected test image. Matching values indicate a correct prediction.

---

## Practical 4 – `DL_pract_4FNN.ipynb`
### FNN with two hidden Dense layers

**Q1. What is the main difference from Practical 3?**  
This notebook uses two hidden Dense layers (128 and 64 neurons), whereas Practical 3 uses three hidden Dense layers (256, 128, and 64 neurons).

**Q2. What is the input shape?**  
The input shape is 784 because each 28 × 28 image is flattened.

**Q3. What is the output shape and why?**  
There are 24 output neurons because the Sign Language MNIST task in this notebook uses 24 classes.

**Q4. Why is `model.summary()` used?**  
It displays the model layers, output shapes, and number of trainable and non-trainable parameters.

**Q5. How are training and validation accuracy plotted?**  
The code plots `history.history['accuracy']` and `history.history['val_accuracy']` against epochs.

**Q6. What does a widening gap between training and validation accuracy suggest?**  
It can indicate overfitting: the model is learning the training examples well but generalizing less effectively to validation examples.

**Q7. Why plot loss as well as accuracy?**  
Accuracy measures the proportion of correct classifications, while loss measures the model's prediction error according to the chosen loss function. Both provide useful information.

**Q8. Why does the code predict only `X_test[:5]` for the display?**  
It selects the first five test images to demonstrate the model's predictions without displaying the entire test set.

---

## Practical 5 – `DL_5.ipynb`
### FNN for Sign Language MNIST

**Q1. What is the aim?**  
To train an FNN to classify Sign Language MNIST images into 24 classes and evaluate it on test data.

**Q2. What are `X_train`, `y_train`, `X_test`, and `y_test`?**  
`X` variables contain image pixel features. `y` variables contain the corresponding class labels. The train variables are used for learning; the test variables are used for evaluation.

**Q3. Why are labels remapped?**  
The original Sign Language MNIST labels are converted into consecutive class indices from 0 to 23 so they match the 24 output neurons.

**Q4. What is the FNN architecture?**  
The notebook uses 784 input values, Dense layers of 256, 128, and 64 neurons with ReLU, and a 24-neuron Softmax output layer.

**Q5. Why use ReLU in hidden layers?**  
ReLU returns zero for negative inputs and the input itself for positive inputs. It introduces non-linearity and is commonly used in hidden layers.

**Q6. Why use Softmax in the final layer?**  
The task is multi-class classification. Softmax produces a probability distribution over the 24 classes.

**Q7. How does the notebook evaluate the model?**  
It calls `model.evaluate(X_test, y_test)` and reports test loss and test accuracy.

**Q8. What does `np.argmax(y_pred, axis=1)` do?**  
It converts each row of predicted class probabilities into the index of the most likely class.

---

## Practical 6 – `DL_6.ipynb`
### CNN for American Sign Language recognition

**Q1. What is the aim of this practical?**  
To classify ASL alphabet gestures using a Convolutional Neural Network (CNN) and the Sign Language MNIST dataset.

**Q2. What is a CNN?**  
A CNN is a neural network designed to process grid-like data such as images. Convolutional filters learn local visual patterns such as edges and shapes.

**Q3. Why is the image reshaped to `(28, 28, 1)`?**  
CNN input has height, width, and channel dimensions. The final `1` represents one grayscale channel.

**Q4. What does `Conv2D(32, (3,3), activation='relu')` mean?**  
It applies 32 learnable 3 × 3 filters to the input feature maps and applies ReLU to the resulting activations.

**Q5. What is MaxPooling2D used for?**  
It reduces the spatial dimensions of feature maps by retaining the maximum value in each pooling window. This reduces computation and helps retain strong features.

**Q6. What is the role of Flatten?**  
It converts the final multi-dimensional feature maps into a one-dimensional vector for the Dense layers.

**Q7. Why does the output layer have 24 neurons?**  
The model predicts one of 24 Sign Language alphabet classes.

**Q8. Why is sparse categorical crossentropy suitable?**  
The labels are integer class IDs rather than one-hot encoded vectors.

**Q9. What is the main difference between an FNN and CNN for images?**  
An FNN usually receives flattened pixels and does not explicitly preserve image locality. A CNN uses shared convolutional filters to learn spatial patterns from the image.

---

## Practical 7 – `DL_7.ipynb`
### CNN with validation, predictions, and sample display

**Q1. What is the objective?**  
To train a CNN on Sign Language MNIST, evaluate it, predict alphabet labels, and display five test images with actual and predicted labels.

**Q2. Why is `train_test_split` used?**  
It separates part of the training data for validation so performance can be monitored during training.

**Q3. What does `test_size=0.2` mean?**  
Twenty percent of the supplied training data is reserved for validation and 80% remains for model fitting.

**Q4. Why is `random_state=42` included?**  
It makes the split reproducible so the same random partition can be obtained again.

**Q5. Why is `stratify=y_train` useful?**  
It helps keep the class distribution similar in the training and validation subsets.

**Q6. What does `predictions = model.predict(X_test)` return?**  
It returns a probability or score for each of the 24 classes for every test image.

**Q7. Why use `label_names`?**  
It maps numeric class indices to readable alphabet letters for displaying predictions.

**Q8. What is the purpose of `plt.subplot(1, 5, i+1)`?**  
It creates five plotting positions in one row and selects the position for the current image.

---

## Practical 8 – `DL_8.ipynb`
### CNN training accuracy, testing accuracy, and loss graph

**Q1. What is the purpose of this notebook?**  
To build a CNN for Sign Language alphabet recognition, train it, report training and testing accuracy, and plot the loss graph.

**Q2. What preprocessing is performed?**  
The labels are mapped to 24 consecutive classes, pixel values are divided by 255, and images are reshaped to 28 × 28 × 1.

**Q3. What does `validation_split=0.2` do in `model.fit()`?**  
It reserves 20% of the supplied training data for validation during training.

**Q4. What does `history.history['accuracy'][-1]` represent?**  
It retrieves the training accuracy recorded at the final epoch.

**Q5. How is testing accuracy obtained?**  
By calling `model.evaluate(X_test, y_test)`. This evaluates the model on the separate test dataset.

**Q6. What does the loss graph show?**  
It shows training loss and validation loss across epochs. A decreasing loss generally indicates that the model is improving on that data.

**Q7. Why should testing data not be used repeatedly to tune a model?**  
Repeated tuning against test results can indirectly overfit the test set. A separate validation set should be used for model selection, leaving test data for final evaluation.

---

## Practical 9 – `DL_9.ipynb`
### CNN prediction of Sign Language alphabet labels

**Q1. What is the aim?**  
To build a CNN using Sign Language MNIST, predict the alphabet labels of test images, and display actual and predicted labels for five samples.

**Q2. Why are J and Z absent from the 24-class Sign Language MNIST alphabet set?**  
In the dataset convention, J and Z involve motion, while the dataset contains static hand-gesture images.

**Q3. What does `np.sort(np.unique(y_train))` do?**  
It obtains the unique training labels and sorts them.

**Q4. Why create `label_map`?**  
It maps the original labels to consecutive class indices suitable for the model's output layer.

**Q5. What does `np.argmax(predictions, axis=1)` return?**  
It returns the predicted class index for each test image by selecting the largest output score.

**Q6. How does the code display the result?**  
It uses Matplotlib to show five grayscale test images and writes the actual and predicted alphabet labels in each subplot.

**Q7. What is the limitation of this CNN?**  
It learns from static 28 × 28 images. It does not directly model temporal movement in gestures.

---

## Practical 10 – `DL_10.ipynb`
### CNN for Sign Language alphabet recognition

**Q1. What is the model used?**  
A CNN with two convolutional layers, two max-pooling layers, a Flatten layer, a 128-neuron Dense layer, and a 24-neuron Softmax output layer.

**Q2. Why are 32 and 64 filters used in the convolutional layers?**  
They specify the number of feature maps learned by the first and second convolutional layers.

**Q3. What is the purpose of the 3 × 3 kernel?**  
It is a small learnable filter that scans local regions of the image to detect visual patterns.

**Q4. What is the purpose of `model.summary()`?**  
It displays the architecture and parameter counts of the CNN.

**Q5. What does the validation set measure?**  
It measures performance on examples not used for weight updates during that training run.

**Q6. How is the final testing result reported?**  
The notebook uses `model.evaluate()` and prints test loss and test accuracy.

---

## Practical 11 – `DL_11.ipynb`
### Feedforward Neural Network for Fashion-MNIST

**Q1. What is Fashion-MNIST?**  
It is a dataset of 28 × 28 grayscale clothing images in 10 categories, such as shirts, shoes, bags, and coats.

**Q2. What is the aim of this practical?**  
To build an FNN for clothing-image classification and evaluate it using test data and a confusion matrix.

**Q3. Why use Flatten before Dense layers?**  
It converts each 28 × 28 image into a 784-element vector.

**Q4. What is the architecture?**  
The model uses Flatten, Dense layers with 128 and 64 ReLU neurons, and a 10-neuron Softmax output layer.

**Q5. What is a confusion matrix?**  
It is a table comparing actual classes with predicted classes. It shows correct classifications and which classes are being confused.

**Q6. What are precision and recall?**  
Precision is the proportion of predicted positives that are correct. Recall is the proportion of actual positives that the model correctly identifies.

**Q7. Why is a confusion matrix useful in addition to accuracy?**  
Accuracy gives one overall score, while a confusion matrix reveals which individual categories are classified correctly or confused with others.

---

## Practical 12 – `DL_12.ipynb`
### Feedforward Neural Network for handwritten MNIST digits

**Q1. What is the MNIST dataset?**  
It is a dataset of 28 × 28 grayscale handwritten digit images, representing digits 0 through 9.

**Q2. What is the aim?**  
To build an FNN to classify handwritten digits using the MNIST dataset.

**Q3. How many output neurons are required?**  
Ten output neurons, one for each digit from 0 to 9.

**Q4. Why is Softmax used?**  
It converts the ten output scores into probabilities for the ten digit classes.

**Q5. What does the hidden Dense layer learn?**  
It learns combinations of pixel patterns that help distinguish the digits.

**Q6. How is the model evaluated?**  
The test images and labels are passed to `model.evaluate()` to obtain test loss and accuracy.

---

## Practical 13 – `DL_13.ipynb`
### FNN for Sign Language MNIST

**Q1. What is the aim?**  
To classify Sign Language MNIST hand-gesture images using an FNN and convert predictions into alphabet labels.

**Q2. Why are image pixels flattened?**  
The FNN uses Dense layers, so each image is converted from 28 × 28 pixels into 784 input values.

**Q3. Why map labels to 0–23?**  
The model has 24 output classes. Consecutive indices provide a direct match between class IDs and output neurons.

**Q4. What does the Softmax layer do?**  
It gives a probability distribution across the 24 alphabet classes.

**Q5. How does the notebook convert predicted numbers into letters?**  
It uses the saved original class values and `chr(... + 65)` to obtain the corresponding alphabet character.

**Q6. What is the difference between validation and test data?**  
Validation data monitors training and model choices. Test data is used for a final, independent performance estimate.

---

## Practical 14 – `DL_14.ipynb`
### FNN for CIFAR-10 object classification

**Q1. What is CIFAR-10?**  
It is a dataset of 32 × 32 colour images belonging to 10 object classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

**Q2. What is the aim of this practical?**  
To build an FNN that classifies CIFAR-10 images into ten object categories.

**Q3. What is the input size after flattening?**  
A 32 × 32 RGB image has 32 × 32 × 3 = 3072 values.

**Q4. Why does the model use 10 output neurons?**  
CIFAR-10 contains ten classes.

**Q5. What is the model architecture?**  
The notebook uses a Flatten layer, Dense layers with 256 and 128 ReLU neurons, and a 10-neuron Softmax output layer.

**Q6. Why might an FNN be less suitable than a CNN for image classification?**  
Flattening removes the explicit two-dimensional spatial structure. CNNs preserve local image relationships and share filters across locations.

**Q7. What is the difference between RGB and grayscale?**  
RGB images have three colour channels; grayscale images have one intensity channel.

---

## Practical 14 – `DL_14(using cifar-10 dataset).ipynb`
### FNN using CIFAR-10 data stored in a CSV file

**Q1. What is the aim?**  
To classify CIFAR-10 images using an FNN after loading the pixel data from a CSV file.

**Q2. How is this notebook different from `DL_14.ipynb`?**  
This version reads a prepared `cifar-10.csv` file using Pandas and uses `train_test_split`. The other version loads CIFAR-10 directly from TensorFlow/Keras.

**Q3. Why reshape the data to `(-1, 32, 32, 3)`?**  
The CSV contains flattened pixel values. Reshaping restores the height, width, and RGB channel dimensions of each image.

**Q4. Why use a Flatten layer in the model?**  
The input images are 32 × 32 × 3. Flatten converts them to a vector before passing them to Dense layers.

**Q5. What does `test_size=0.2` mean?**  
Twenty percent of the available examples are assigned to the test set and 80% to the training set.

**Q6. Why divide by 255?**  
It scales 8-bit pixel values to the 0–1 range.

**Q7. What is one caution when comparing this practical with the Keras-loaded version?**  
The exact CSV row and channel ordering must match the assumptions in the reshape operation. Incorrect ordering can scramble image pixels.

---

## Practical 15 – `cheet_15.ipynb`
### Comparing FNN models on handwritten MNIST

**Q1. What is the purpose of this notebook?**  
To compare the effect of adding a hidden Dense layer to an FNN for handwritten digit classification.

**Q2. What is Model 1's architecture?**  
Flatten, Dense(128, ReLU), and Dense(10, Softmax).

**Q3. What is Model 2's architecture?**  
Flatten, Dense(128, ReLU), Dense(64, ReLU), and Dense(10, Softmax).

**Q4. What is the main difference between the two models?**  
Model 2 has an additional hidden layer with 64 neurons.

**Q5. Why compare two architectures?**  
To observe how changing the number of hidden layers affects learning and classification performance.

**Q6. Does adding more layers always improve accuracy?**  
No. More layers increase model capacity but can also increase training time and overfitting. Performance must be measured.

**Q7. Why is the output layer size 10?**  
MNIST has ten digit classes, 0 through 9.

---

## Practical 16 – `cheet_16.ipynb`
### FNN for Fashion-MNIST with confusion matrix

**Q1. What is the aim?**  
To classify Fashion-MNIST clothing images using an FNN and assess its performance using a confusion matrix and classification report.

**Q2. What is the model architecture?**  
Flatten, Dense(128, ReLU), Dense(64, ReLU), and Dense(10, Softmax).

**Q3. Why does the model use 10 output neurons?**  
Fashion-MNIST has ten clothing categories.

**Q4. What does `validation_split=0.2` do?**  
It uses 20% of the supplied training data for validation during training.

**Q5. What does `classification_report` provide?**  
It reports precision, recall, F1-score, and support for each class, along with summary averages.

**Q6. What is F1-score?**  
It is the harmonic mean of precision and recall. It balances the two measures.

**Q7. What does the confusion matrix diagonal represent?**  
Correct predictions for each class. Off-diagonal values show misclassifications.

---

## Practical 17 – `cheet_17.ipynb`
### CNN for handwritten MNIST digits

**Q1. What is the aim?**  
To recognize handwritten digits using a CNN trained on MNIST.

**Q2. What is the CNN architecture?**  
Input 28 × 28 × 1, Conv2D with 32 filters, MaxPooling2D, Conv2D with 64 filters, another MaxPooling2D, Flatten, Dense(128), and Dense(24) Softmax as written in the notebook.

**Q3. How many classes are in MNIST?**  
Ten classes, representing digits 0–9.

**Q4. Is the output layer in this notebook aligned with MNIST?**  
The notebook's CNN definition has 24 output neurons, although MNIST has ten digit classes. For a correct MNIST classifier, the output layer should have 10 neurons and the labels should be in the range 0–9.

**Q5. What is a convolution filter?**  
A small learnable matrix that slides over an image and responds to local patterns.

**Q6. Why use pooling after convolution?**  
Pooling reduces feature-map dimensions and computation while retaining strong local responses.

---

## Practical 18 – `cheet_18.ipynb`
### CNN for Fashion-MNIST clothing images

**Q1. What is the aim?**  
To classify clothing categories from Fashion-MNIST using convolutional and pooling layers, and predict labels for test images.

**Q2. What input shape is used?**  
28 × 28 × 1, because Fashion-MNIST images are 28 × 28 grayscale images.

**Q3. What are the CNN layers?**  
Two Conv2D layers with 32 and 64 filters, each followed by MaxPooling2D, then Flatten, Dense(128), and Dense(24) Softmax in the notebook.

**Q4. Is 24 the correct number of output classes for Fashion-MNIST?**  
No. Fashion-MNIST has ten classes, so the final layer should have 10 neurons for a correctly configured classifier.

**Q5. What does the prediction step do?**  
It passes test images through the network and selects the output class with the highest score.

**Q6. Why compare actual and predicted labels?**  
It gives a simple visual check of whether the classifier recognizes sample clothing images correctly.

---

## Practical 19 – `cheet_19.ipynb`
### CNN for CIFAR-10 object classification from CSV

**Q1. What is the aim?**  
To classify colour object images into ten CIFAR-10 categories using a CNN.

**Q2. What is the image shape?**  
32 × 32 × 3, representing height, width, and RGB colour channels.

**Q3. Why is a CNN suitable for CIFAR-10?**  
A CNN can learn spatial and colour patterns such as edges, textures, and object parts.

**Q4. What is the model architecture?**  
Two Conv2D layers (32 and 64 filters), two MaxPooling2D layers, Flatten, Dense(128), and a 24-neuron Softmax output as written in the notebook.

**Q5. Is the output size consistent with CIFAR-10?**  
No. CIFAR-10 has ten categories, but the notebook specifies 24 output neurons. A correct ten-class model needs 10 output neurons.

**Q6. Why is `train_test_split` used?**  
It separates the prepared dataset into training and testing portions.

**Q7. What does `activation='relu'` do in convolutional layers?**  
It introduces non-linearity by setting negative activations to zero.

---

## Practical 20 – `cheet_20.ipynb`
### CNN for MNIST with accuracy and loss graphs

**Q1. What is the aim?**  
To train a CNN for handwritten digit classification and display training and validation accuracy and loss.

**Q2. What are the input dimensions?**  
28 × 28 × 1.

**Q3. What is the purpose of the two convolutional layers?**  
They learn increasingly complex image features from the digit images.

**Q4. What is the purpose of the two MaxPooling layers?**  
They reduce spatial dimensions of the feature maps.

**Q5. What do the accuracy graphs show?**  
They show training and validation accuracy across epochs.

**Q6. What do the loss graphs show?**  
They show training and validation loss across epochs.

**Q7. What can indicate overfitting in these graphs?**  
Training accuracy continues to rise while validation accuracy stops improving or falls; training loss falls while validation loss rises.

**Q8. What output-size issue appears in the notebook?**  
The notebook specifies 24 output neurons although MNIST has ten digit classes. The output layer should have 10 neurons for the intended task.

---

## Practical 21 – `cheet_21.ipynb`
### Memory-efficient CNN pipeline for CIFAR-10

**Q1. What is the aim?**  
To build and train a CNN for CIFAR-10 using a TensorFlow data pipeline, and evaluate classification performance.

**Q2. Why is the CSV loaded with `dtype=np.uint8`?**  
Image pixels are 8-bit integers from 0 to 255. Loading them as `uint8` can reduce memory usage.

**Q3. Why reshape to `(-1, 3, 32, 32)` and then transpose?**  
The CSV stores channels-first pixel sequences. The transpose changes the data to TensorFlow's channels-last format: number of images, height, width, channels.

**Q4. What is `tf.data.Dataset`?**  
It is TensorFlow's data input pipeline API for creating, transforming, batching, and prefetching datasets.

**Q5. What does `.shuffle(5000)` do?**  
It randomizes the order of examples using a shuffle buffer of 5000 elements.

**Q6. Why use `.batch(32)`?**  
It groups 32 examples into each training batch.

**Q7. What does `tf.cast(x, tf.float32) / 255.0` do?**  
It converts pixel data to floating-point values and normalizes them to the 0–1 range.

**Q8. What does `.prefetch(1)` do?**  
It prepares one batch ahead to help overlap data preparation with model execution.
