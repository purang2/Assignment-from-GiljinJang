# ID: 2021220699
# NAME: Eunchan Lee 
# File name: read_iris_v2.py
# Platform: Python 3.7 on Spyder4
# Required Package(s): sys numpy 

import sys
import numpy as np

#self-made function that makes numpy data a python list. Used in line #82~#85
def tolist(datalist, datatype):
    output_list = []
    
    # if datatype is txt, then split each line by 공백
    # elif datatype is txt, then split each line by ","
    if datatype== "txt":
        for i in range(len(datalist)):
            output_list.append(data1[i].split("\t"))    
    elif datatype == "csv":
        for i in range(len(datalist)):
            output_list.append(data1[i].split(","))    
    else: 
        print("tolist error: datatype is not txt or csv")
    
    return output_list    


# made by @EunchanLee
if len(sys.argv) < 2:
     print('usage: ' + sys.argv[0] + ' text_file_name')
     print("sys.argv is not correct! please re-input sys.argv")
else:
    
    filetype = sys.argv[1][-3:]     #ex) "csv"
    filename = sys.argv[1]          #ex) "iris.csv"
    
         
    #Open the input file and Perform data preprocessing based on file type    
    data = []   
    with open(filename) as f:
        data_read = f.read().splitlines()
        
    if filetype == "txt":
        print("successfully opened the txt file")        
        for i in range(len(data_read)):
            data.append(data_read[i].split("\t"))        
    elif filetype == "csv":
        print("successfully opened the csv file")
        for i in range(len(data_read)):
            data.append(data_read[i].split(","))                    
    
    #if the input file type is not txt or csv, 
    #the program below will output an error, so exit is the best choice.
    else:
        print("argv input is not correct, please re-input sys.argv to csv or txt")
        exit(1) 

    #delete the last line : ["species"] or ["CLASS"]
    for row in data:
        del row[-1]
    
    #The first line is a data name, so save it separately.
    data_type = data[0]
    data = data[1:] 
    
    #convert data type: str to float by using list comprehension!
    data = [[float(y) for y in x] for x in data]
    data = np.array(data)
    

    #if set axis=0, calculate between columns
    mean_data = np.mean(data, axis=0)
    std_data =np.std(data, axis=0)

    
    #Round to the second decimal place
    mean_data = mean_data.round(2)
    std_data =std_data.round(2)
    
    
    #Process data to fit the output 
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
    

