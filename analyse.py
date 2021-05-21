import pydirectinput
import pandas as pd


if (__name__ == '__main__') :
    pydirectinput.press("down")
    data = pd.read_csv('train/train_data1.csv', sep=',', index_col=0)
    print(data)