# -*- coding: utf-8 -*-
import os
import sys
import click
import logging

sys.path.append('.')
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from sklearn.model_selection import train_test_split
from src.csv2df import csv2df


# selects large class datapoints as many as small class
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


# select features and shuffle the dataset
def feature(dataframe):
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
    dataframe, filename = csv2df(input_filepath)
# (2) sample selection
    label_0_count = dataframe['label'].value_counts().get(0, 1)
    label_1_count = dataframe['label'].value_counts().get(1, 0)
    if label_0_count >= label_1_count:
        dataframe_indices = sampler(dataframe, 0, 1, label_1_count)
    else:
        dataframe_indices = sampler(dataframe, 1, 0, label_0_count)
    dataframe = dataframe.loc[dataframe_indices]
# (3) & (4) feature selection, shuffling
    shuffled = feature(dataframe)
# (5) splitting
    train_df, test_df = train_test_split(shuffled, test_size=0.3, random_state=42)
# (6) save for folder given at prompt
    if output_filepath.endswith(os.path.sep):
        output_filepath = output_filepath[:-1]
    else:
        output_filepath
    train_path = output_filepath + '/' + filename + '_' + 'training.csv'
    test_path = output_filepath + '/' + filename + '_' + 'testing.csv'

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger = logging.getLogger(__name__)
    logger.info(
        '*** Dataset is loaded, sampled (rows selected), featured (columns selected), shuffled, and split (train:test) ***')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
