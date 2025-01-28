import numpy as np
import pandas as pd

class DataObject:
    def __init__(datas, x, y, x_err, y_err):
        datas.x = x      
        datas.y = y     
        datas.x_err = x_err
        datas.y_err = y_err  


def read_tables_from_csv(file_path):
    df = pd.read_csv(file_path,
                 sep=';',
                 engine='python', decimal=',', header=None)
    x = df.iloc[:, 0]
    y = df.iloc[:, 1]
    x_err = df.iloc[:, 2]
    y_err = df.iloc[:, 3]

    x_err = np.maximum(abs(x) * 10**(-10), x_err)
    y_err = np.maximum(abs(y) * 10**(-10), y_err)
    
    data = DataObject(x, y, x_err, y_err)
    
    return data