# ML_Project16-FaceMaskDetection

### Face Mask Detection with CNN

This project implements a Convolutional Neural Network (CNN) for classifying images with or without face masks. The model is trained on a dataset containing images labeled as "With Mask" or "Without Mask".

### Getting Started

Requirements: Ensure you have PyTorch, torchvision, and other necessary libraries installed. You can install them using pip install torch torchvision.

Download the Dataset: Download the face mask dataset from link to the dataset on Kaggle: https://www.kaggle.com/ashishjangra27/face-mask-12k-images-dataset. Place the unzipped folder containing the training, validation, and test directories in the same location as the script.

Run the Script: Execute the Python script (e.g., main.py). The script performs the following steps:

Loads and pre-processes the image data.

Defines the CNN architecture with convolutional, pooling, and fully-connected layers.

Implements training and validation functions to train the model and evaluate its performance.

Trains the model for a specified number of epochs.

Evaluates the trained model on the validation set and displays the results.

Optionally, predicts masks on images from the test set.

### Code Breakdown:

##### Data Loading and Preprocessing:

The script utilizes ImageFolder from torchvision.datasets to load images from directories.

Data transformations are applied using torchvision.transforms. These transformations include resizing, random horizontal flips, color jittering (for data augmentation), normalization, and conversion to tensors.

Separate DataLoader instances are created for training, validation, and test datasets.

##### CNN Model:

The CNN class inherits from a base FaceMaskDetec class which defines training, validation, and epoch-end logging functionalities.

The CNN architecture consists of several convolutional layers with ReLU activation and Batch Normalization for improved training stability.

Max pooling layers are used for downsampling.

The final layer has two output units corresponding to "With Mask" and "Without Mask" classes.

The sigmoid activation function is applied to the final layer's output to get probabilities between 0 and 1.

##### Training and Evaluation:

The fit function trains the model for a specified number of epochs.

It employs the Adam optimizer with a learning rate scheduler for dynamic learning rate adjustment.

The function calculates the training loss and performs validation after each epoch, logging the validation loss and accuracy.

Gradient clipping is optionally applied to prevent exploding gradients.

##### Testing (Optional):

The test function takes an image batch and predicts the mask label (with or without mask) for each image.


##### Further Exploration:

Experiment with different hyperparameters like the number of epochs, learning rate, optimizer, and CNN architecture to potentially improve performance.

Explore more advanced data augmentation techniques for better generalization.

Visualize the learned filters in the convolutional layers to understand what features the model focuses on for classification.

Implement a user interface to capture images from a webcam and perform real-time mask detection.
