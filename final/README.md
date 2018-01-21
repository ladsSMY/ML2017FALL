0.Introuction 
All codes and attached filed are in src/ 
src/ 
 | 
 |----(1)answer/ 
 | 
 |----(2)data/ 
 |        | 
 |        |----train/ 
 |		  |----test/ 
 |		  |----other data files 
 | 
 |----(3)w2vmodel 
 | 
 |----(4)python files and scripts 



1.Training W2V models 
使用以下指令來訓練W2V模型 

		bash s1.sh <training datas directory>

<training datas directory>為存放五個training data的資料夾，執行後會以我們嘗試出最好的兩組參數訓練出兩個模型，並存放於(3)w2vmodel當中。執行時間大約為2400秒。 




