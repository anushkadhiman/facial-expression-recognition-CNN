# facial-expression-recognition-CNN

Used Convolutional neural networks (CNN) for facial expression recognition. The goal is to categorize each face based on the emotion shown in the facial expression in to one of seven categories (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral).

# Data
The dataset for the model is taken from https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data .
The data consists of 48x48 pixel grayscale images of faces. The faces have been automatically registered so that the face is more or less centered and occupies about the same amount of space in each image. 
train.csv contains two columns, "emotion" and "pixels". The "emotion" column contains a numeric code ranging from 0 to 6, inclusive, for the emotion that is present in the image. The "pixels" column contains a string surrounded in quotes for each image. The contents of this string a space-separated pixel values in row major order. test.csv contains only the "pixels" column and your task is to predict the emotion column.

# Library Used
- Keras
- Sklearn
- OpenCV
- pandas
- numpy

We will build and train models on Google Colab, a free Jupyter notebook environment that runs on Google cloud and gives FREE GPU!

# Reference

1. Convolutional Neural Networks for Facial Expression Recognition - Shima Alizadeh,Stanford University and Azar Fazel,Stanford University, http://cs231n.stanford.edu/reports/2016/pdfs/005_Report.pdf
