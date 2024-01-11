import os
import pandas as pd


def csv2df(input_filepath):
    # Extract the name of the CSV file (excluding the extension)
    filename = os.path.splitext(os.path.basename(input_filepath))[0]
    # Load the CSV file into a DataFrame
    dataframe = pd.read_csv(input_filepath)
    return dataframe, filename
