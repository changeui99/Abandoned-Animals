import pandas as pd

if (__name__ == '__main__') :
    data = pd.read_csv('train_data.csv', sep=',', index_col=0)
    print(data)