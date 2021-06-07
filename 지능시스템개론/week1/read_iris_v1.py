# ID: 2021220699
# NAME: Eunchan Lee 
# File name: read_iris_v1.py
# Platform: Python 3.7 on Spyder4
# Required Package(s): sys numpy pandas

import sys
import numpy as np
import pandas as pd

if len(sys.argv) < 2:
     print('usage: ' + sys.argv[0] + ' text_file_name')
     print("sys.argv is not correct! please re-input sys.argv")

else:

    #set the delimeter ,dependent on input file type    
    if sys.argv[1][-3:].lower() == 'csv': delimeter = ','
    else: delimeter = '[ \t\n\r]'  # default is all white spaces 
    
    
    file_type = sys.argv[1][-3:].lower() #ex) 'csv' or 'txt' or other

    #If the file format is incorrect, program will not be executed.
    if file_type != 'txt' and file_type !='csv':
        print("file type is not txt or csv, please run again")
        exit(1)         
    
    #else, If the file format is correct, it will be executed.
    else:
    
        f = pd.read_csv(sys.argv[1],sep=delimeter,engine='python')    
              
        #delete the last line : ["species"] or ["CLASS"], because it's useless.
        if file_type== 'txt': 
            f = f.drop(["CLASS"],axis='columns')  
        if file_type == 'csv':
            f = f.drop(['species'],axis='columns')
        
        #store the title data : ex) 'sepal_length' or 'SL' ... 
        data_type= f.columns.tolist()
        
        #Change the data type to use Numpy methods
        data= f.to_numpy()
        

        #if set axis=0, calculate between columns
        mean_data = np.mean(data, axis=0)
        std_data =np.std(data, axis=0)

   
        #Round to the second decimal place ex) 5.66666 -> 5.67
        mean_data = mean_data.round(2)
        std_data =std_data.round(2)
        
        
        #Process data to fit the simple output 
        result = [[],[],[]]
        result[0] =[""] + data_type 
        result[1] = ["mean"] + mean_data.tolist()
        result[2] = ["std"] + std_data.tolist()
        
        
        #Beautiful output using string formatting
        str_format = "%-5s%15s%15s%15s%15s"
        print("-----------------------------------------------------------------")
            
        for i in range(len(result)):
            print(str_format%(result[i][0],result[i][1],result[i][2],result[i][3],result[i][4]))
                            
        print("-----------------------------------------------------------------")
            
