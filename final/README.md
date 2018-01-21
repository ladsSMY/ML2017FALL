0. Introuction  
All codes and attached filed are in src/  
  
  
1. Training W2V models  
使用以下指令來訓練W2V模型  

		bash s1.sh <training datas directory>

training datas directory 為存放五個training data的資料夾，執行時會進入資料夾把五個txt檔案抓出來讀，但希望相同資料夾內不要放其他txt檔案，不然會造成錯誤。  
執行後會以我們嘗試出最好的兩組參數訓練出兩個模型，並存放於w2vmodel資料夾當中。執行時間大約為2400秒。  
  
  
2. Predict answer  
使用以下指令來產生預測出來的答案  

		bash s2.sh <testing data> <prediction file>

testing data為testing_data.csv，而prediction file為產生出的結果。



 



