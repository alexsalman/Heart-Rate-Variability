# -*- coding: utf-8 -*-
import os
import click
import logging
import pandas as pd
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from sklearn.model_selection import train_test_split


def loader(input_filepath):
    # Extract the name of the CSV file (excluding the extension)
    filename = os.path.splitext(os.path.basename(input_filepath))[0]
    # Load the CSV file into a DataFrame
    dataframe = pd.read_csv(input_filepath)
    return dataframe, filename


def sampler(dataframe, large_class, small_class, small_class_count):
    count = 0
    large_class_indices, small_class_indices = [], []
    for index, row in dataframe.iterrows():
        if count < small_class_count and row['label'] == large_class:
            large_class_indices.append(index)
            count += 1
        elif row['label'] == small_class:
            small_class_indices.append(index)
        else:
            pass
    return large_class_indices + small_class_indices


def featurer(dataframe):
    # feature selection
    chosen_columns = ['PNS index', 'SNS index', 'Stress index', 'Mean RR  (ms)', 'SDNN (ms)', 'Mean HR (beats/min)',
                      'SD HR (beats/min)', 'Min HR (beats/min)', 'Max HR (beats/min)', 'RMSSD (ms)', 'NNxx (beats)',
                      'pNNxx (%)', 'RR tri index', 'TINN (ms)', 'DC (ms)', 'DCmod (ms)', 'AC (ms)', 'ACmod (ms)',
                      'VLF (Hz)', 'LF (Hz)', 'HF (Hz)', 'VLF (ms^2)', 'LF (ms^2)', 'HF (ms^2)', 'VLF (log)', 'LF (log)',
                      'HF (log)', 'VLF (%)', 'LF (%)', 'HF (%)', 'LF (n.u.)', 'HF (n.u.)', 'Total power (ms^2)',
                      'LF/HF ratio', 'RESP (Hz)', 'SD1 (ms)', 'SD2 (ms)', 'SD2/SD1 ratio', 'Approximate entropy (ApEn)',
                      'Sample entropy (SampEn)', 'alpha 1', 'alpha 2', 'Correlation dimension (D2)',
                      'Mean line length (beats)', 'Max line length (beats)', 'Recurrence rate (REC) (%)',
                      'Determinism (DET) (%)', 'Shannon entropy', 'MSE(1)', 'MSE(2)', 'VLF (Hz) AR spectrum',
                      'LF (Hz) AR spectrum', 'HF (Hz) AR spectrum', 'VLF (ms^2) AR spectrum', 'LF (ms^2) AR spectrum',
                      'HF (ms^2) AR spectrum', 'VLF (log) AR spectrum', 'LF (log) AR spectrum', 'HF (log) AR spectrum',
                      'VLF (%) AR spectrum', 'LF (%) AR spectrum', 'HF (%) AR spectrum', 'LF (n.u.) AR spectrum',
                      'HF (n.u.) AR spectrum', 'label']
    # shuffling selected features
    shuffled = dataframe[chosen_columns].sample(frac=1, random_state=42)
    return shuffled


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be split for use (saved in ../interim).
    """
    # (1) loading dataset
    dataframe, filename = loader(input_filepath)
    # (2) sample selection
    dataframe_indices = sampler(dataframe, 0, 1, len(dataframe.loc[dataframe['label'] == 1, :]))
    dataframe = dataframe.loc[dataframe_indices]
    # (3) & (4) feature selection, shuffling
    shuffled = featurer(dataframe)
    # (5) splitting
    train_df, test_df = train_test_split(shuffled, test_size=0.5, random_state=0)
    # (6) save it in interim
    if output_filepath.endswith(os.path.sep):
        output_filepath = output_filepath[:-1]
    else:
        output_filepath
    train_path = output_filepath + '/' + filename + '_' + 'training.csv'
    test_path = output_filepath + '/' + filename + '_' + 'testing.csv'

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(train_df.head)
    print(test_df.head)
    print(train_df.shape)
    print(test_df.shape)

    logger = logging.getLogger(__name__)
    logger.info('Dataset has been: [1]loaded, [2]sampled (rows selected), [3]featured (columns selected), [4]shuffled, and [5]splitted')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
