import pandas as pd
import numpy as np
from my_utils import read_data, filter_with_type, clean_dataframe, reformat_dataframe, encode_dataframe, save_csv
from pandas_profiling import ProfileReport

if __name__ == '__main__':
    FILE_PATH = './data/input/'
    FILE_NAME = 'netflix_titles.csv'

    # reading file
    df = read_data(FILE_PATH, FILE_NAME)

    # filtering only movies
    df = filter_with_type(df, type_value='Movie')

    # Imputing NaNs
    df = clean_dataframe(df)

    # reformating: (director, cast, country, date_added, listed_in)
    df = reformat_dataframe(df)

    # categorical encoding
    df = encode_dataframe(df)

    # saving the clean file
    save_csv(df, out_path="./data/output", file_name='clean_data.csv')
    ProfileReport(df, title="Dataset Report").to_file(
        "./data/output/report.html")
