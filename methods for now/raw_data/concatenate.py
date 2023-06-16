import pandas as pd
import glob
import os

def run(day):
    df_csv_append = pd.DataFrame()
    #day = '0307'
    sections = ['01', '02', '03', '04', '05']
    csv_files = ["" + day + section + '/final_data.csv' for section in sections]
    # append the CSV files
    df_csv_concat = pd.concat([pd.read_csv(file) for file in csv_files ], ignore_index=True)
    df_csv_concat.iloc[:, 1:].to_csv("" + day + "/final_data.csv")
 
def merge_beh(monkey, day):
    # list all csv files only
    path = day
    os.chdir(path)
    csv_files = glob.glob('*{}.{}'.format('result','csv'))
    df_csv_concat = pd.concat([pd.read_csv(file) for file in csv_files ], ignore_index=True)
    df_csv_concat.iloc[:, 1:].to_csv("" + monkey + day + "_behavior.csv")

if __name__ == '__main__':
    merge_beh('J', '0307')