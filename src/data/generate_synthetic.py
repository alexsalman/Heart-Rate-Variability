# -*- coding: utf-8 -*-
import os
import click
import logging
from pathlib import Path

import numpy as np
from dotenv import find_dotenv, load_dotenv
from matplotlib import pyplot as plt

from make_dataset import loader
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer
import seaborn as sns


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
    GC_synthetic_data = synthesizer_GC.sample(num_rows=1000)
    print(type(GC_synthetic_data))
    print(GC_synthetic_data.head())
    print(GC_synthetic_data.shape)
    return GC_synthetic_data


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../interim) into
        cleaned data ready to be split for use (saved in ../processed).
    """
    # (1) loading dataset
    dataframe, filename = loader(input_filepath)
    print(dataframe.shape)
    # (2) GCSynthesizer
    plt.figure(figsize=(20, 8))
    # heat map real data
    heatmap = sns.heatmap(dataframe.corr())
    heatmap.set_title('Correlation Heatmap for Real Data')
    plt.savefig('reports/figures/heatmap_real_data.png', dpi=400)
    # make synthetic data
    gcs_synthetic_data = gcs(dataframe)
    # heat map synthetic data
    plt.figure(figsize=(20, 8))
    heatmap = sns.heatmap(gcs_synthetic_data.corr())
    heatmap.set_title('Correlation Heatmap for Synthetic Data')
    plt.savefig('reports/figures/heatmap_synthetic_data.png', dpi=400)

    # (3) save it in processed
    if output_filepath.endswith(os.path.sep):
        output_filepath = output_filepath[:-1]
    else:
        output_filepath
    path = output_filepath + '/' + filename + '_' + 'synthesized.csv'
    gcs_synthetic_data.to_csv(path, index=False)

    logger = logging.getLogger(__name__)
    logger.info('Dataset has been: [1]loaded, [2]metadata table created, [3]data  synthesized')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
