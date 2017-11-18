This is hw3 of Machine Learning (2017, Fall)

The assignment is Image Sentiment Classification. We will provide about 28000 pictures with 48*48 pixels as the training data.  And each picture has a sentimental label.  (Notice: There's only "one" sentiment for one picture.)  There are totally 7 sentiments, 0 for angry, 1 for hate, 2 for fear, 3 for joy, 4 for sad, 5 for surprise, 6 for neutral( which indicates it can't be classified into any of 0-5).

Testing data is composed of about 7000 48*48 pixels pictures. We hope that every student utilizes training data to generate a CNN model which is used for predicting the sentimental label. Please save your predictions in a csv file.

There are two scripts in this folder:

1. Train the model with training data, and save the model, cnn.h5py in direct folder.

		bash hw3_train.sh (training data)

2. Test the model with test data, the script and python files should put together with model files (cnn.h5py), and it will export result to the direct you specified.

		bash hw3_test.sh (testing data) (prediction file)




