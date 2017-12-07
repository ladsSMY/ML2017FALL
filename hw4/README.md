This is hw4 of Machine Learning (2017, Fall)

For description, please go to: https://www.kaggle.com/c/ml-2017fall-hw4#description

This folder includes three models and ensemble them to predict the answer, the bash script only runs model_1 which have the highest accuracy.

1. Train the model with training data(labeled), training data(no labeled), and save the model in direct folder. (no labeled data are not used)

		bash hw4_train.sh (training_label) (training_nolabel)

2. Test the model with test data, all the files (3 models and 3 tokenize) should put together in the same folder with this script.  It will export result to the direct you specified.

		bash hw3_test.sh (testing data) (prediction file)

3. To run others models:
	
		python3 model_2.py (training_label) (training_nolabel)

		python3 model_3.py (training_label) (testing_data)

p.s. Actually model_2 did not use  (training_nolabel)
p.s. model_3 are based on semi-supervised with (testing_data) but not (training_nolabel)
