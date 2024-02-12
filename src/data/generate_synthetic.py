# -*- coding: utf-8 -*-
import os
import sys
import click
import logging
import seaborn as sns
import pandas as pd


sys.path.append('.')
from src.csv2df import csv2df
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from matplotlib import pyplot as plt
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.evaluation.single_table import evaluate_quality


# Gaussian Copula Synthesizer - Data Generator
def gcs(dataframe):
    # create metadata table
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(dataframe)

    # validate metadata
    metadata.validate()
    metadata.validate_data(data=dataframe)

    # generate an instance of the synthesizer model
    synthesizer_GC = GaussianCopulaSynthesizer(
        metadata,  # required
        enforce_min_max_values=True,
        enforce_rounding=False,
        default_distribution='norm'
    )
    # fit model on dataframe
    synthesizer_GC.fit(dataframe)
    # generate synthetic data
    GC_synthetic_data = synthesizer_GC.sample(num_rows=100)


    quality_report = evaluate_quality(
        dataframe,
        GC_synthetic_data,
        metadata
    )
    print(quality_report)


    combined_data = pd.concat([dataframe, GC_synthetic_data], ignore_index=False)

    return combined_data


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data generation scripts to turn data from (../interim) into
        synthetic data ready for use (saved in ../processed).
    """
# (1) loading dataset
    dataframe, filename = csv2df(input_filepath)
# (2) GCSynthesizer
    # heat map real data
    plt.figure(figsize=(20, 8))
    heatmap = sns.heatmap(dataframe.corr())
    heatmap.set_title('Correlation Heatmap for Real Data')

    # generate synthetic data
    gcs_synthetic_data = gcs(dataframe)
    # heat map synthetic data
    plt.figure(figsize=(20, 8))
    heatmap = sns.heatmap(gcs_synthetic_data.corr())
    heatmap.set_title('Correlation Heatmap for Synthetic Data')
    # plt.savefig('reports/figures/heatmap_synthetic_data.png', dpi=400)


# (3) save it in ../processed
    if output_filepath.endswith(os.path.sep):
        output_filepath = output_filepath[:-1]
    else:
        output_filepath
    # path = output_filepath + '/' + filename + '_' + 'synthesized.csv'
    # gcs_synthetic_data.to_csv(path, index=False)

    logger = logging.getLogger(__name__)
    logger.info(
        '*** Dataset is loaded, metadata created, and data synthesized ***')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
