This is hw1 of Machine Learning , NTU, Prof.LHY

There are two version of this homework: 

1.	
'hw1.sh' runs 'hw1.py', it is the version which beat the baseline before 2017/10/5.
Before running this script, you need to train the model frist using 'hw1_train.py'.
It will produce the model file 'hw1.model', but it cost a really long time for training, 
so I already push the model file into this repository.

2.
'hw1_best.sh' runs 'hw1_best.py', it is the version which approach the highest score 
in kaggle of mine.

To use those script, you need to insert two address:
	./xxx.sh <testfile address> <output file address>
for example:
	./hw1_best.sh test.csv ans_best.csv

If you meet the permission problem, you can use this command:
	chmod 777 xxx.sh
or use
	bash xxx.sh
to run the script.


