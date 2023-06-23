import pandas as pd
import glob
import os

def merge_one_day_eye(day, monkey):
    expect_dir = f'C:/Users/River/23summer/dku/edf_monkey/Data/{monkey}/{day}/'
    csv_files = []
    df_csv_append = pd.DataFrame()
    sections = ['01', '02', '03', '04', '05']
    for section in sections:
        csv_files.append("" + expect_dir + day + section + '/final_data.csv')
        print("" + expect_dir + day + section + '/final_data.csv')
        file = pd.read_csv("" + expect_dir + day + section + '/final_data.csv')
    # append the CSV files
    df_csv_concat = pd.concat([pd.read_csv(file) for file in csv_files ], ignore_index=True)
    df_csv_concat.iloc[:, 1:].to_csv(expect_dir + "/final_data.csv")


def merge_eye(days, monkey):
    expect_dir = f'C:/Users/River/23summer/dku/edf_monkey/Data/{monkey}/'
    csv_files = ['' + expect_dir + day + '/final_data.csv' for day in days]
    # append the CSV files
    df_csv_concat = pd.concat([pd.read_csv(file) for file in csv_files ], ignore_index=True)
    df_csv_concat.iloc[:, 1:].to_csv("C:/Users/River/23summer/dku/edf_monkey/Data/Mercury/merged_data.csv")
 
def find_csv_filenames( path_to_dir, suffix=".csv"):
    filenames = os.listdir(path_to_dir)
    return [ filename for filename in filenames if filename.endswith( suffix ) ]

def merge_beh(days):
    # list all csv files only
    for day in days:
        index = ['01', '02', '03', '04', '05']
        csv_files = find_csv_filenames("C:/Users/River/23summer/dku/edf_monkey/Data/Mercury/" + day + "/M_" + day + '_BHV/')
        df_csv_concat = pd.concat([pd.read_csv("C:/Users/River/23summer/dku/edf_monkey/Data/Mercury/" + day + "/M_" + day + '_BHV/'+file) for file in csv_files ], ignore_index=True)
        df_csv_concat.iloc[:, 1:].to_csv("C:/Users/River/23summer/dku/edf_monkey/Data/Mercury/" + day + "/M_" + day + '_BHV/2023' + day+'_bhv.csv')

def merge_beh_all(days, monkey):
    expect_dir = f'C:/Users/River/23summer/dku/edf_monkey/Data/{monkey}/'
    csv_files = ['' + expect_dir + day + "/M_" + day + '_BHV/2023' + day+'_bhv.csv' for day in days]
    # append the CSV files
    df_csv_concat = pd.concat([pd.read_csv(file) for file in csv_files ], ignore_index=True)
    df_csv_concat.iloc[:, 1:].to_csv("C:/Users/River/23summer/dku/edf_monkey/Data/Mercury/merged_behavior.csv")

if __name__ == '__main__':
    days = ['0506', '0507', '0518', '0519', '0520', '0521', '0525', '0526', '0527']
    merge_beh_all(days, 'Mercury')
    #merge_one_day_eye('0304')
    #merge_beh(['0521', '0525', '0526', '0527'])