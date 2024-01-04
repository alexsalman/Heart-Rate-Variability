# -*- coding: utf-8 -*-
import os
import click
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from matplotlib import pyplot as plt

from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer
import seaborn as sns

import tensorflow as tf
import pandas as pdls
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout, Activation
from keras.optimizers import Adam
from keras.models import load_model


def loader(input_filepath):
    # Extract the name of the CSV file (excluding the extension)
    filename = os.path.splitext(os.path.basename(input_filepath))[0]
    # Load the CSV file into a DataFrame
    dataframe = pd.read_csv(input_filepath)
    return dataframe, filename


def test(model, x, y):
    y_probabilities = model.predict(x)
    # threshold adjustment
    threshold = 0.5
    # Convert probabilities to binary predictions based on the threshold
    y_predictions = (y_probabilities >= threshold).astype(int)
    # Evaluate the model with the adjusted threshold
    conf_matrix = confusion_matrix(y, y_predictions)
    class_report = classification_report(y, y_predictions)
    auc = roc_auc_score(y, y_predictions)
    # flatten confusion matrix
    cm_list = conf_matrix.flatten().tolist()

    def calculate_sensitivity_specificity(tp, tn, fp, fn):
        # Sensitivity (True Positive Rate or Recall)
        sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0.0
        # Specificity (True Negative Rate)
        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0.0
        return sensitivity, specificity

    sensitivity, specificity = calculate_sensitivity_specificity(cm_list[0], cm_list[3], cm_list[1], cm_list[2])
    print('Sensitivity:', round(sensitivity, 2))
    print('Specificity:', round(specificity, 2))
    print("Area Under Curve:", round(auc, 2))
    print("Confusion Matrix:\n", conf_matrix)
    print("\nClassification Report:\n", class_report)

    return


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../interim) into
        cleaned data ready to be split for use (saved in ../processed).
    """
    test_path = 'models/test_features.csv'

    # Use genfromtxt to load the CSV file into a NumPy array
    data = np.genfromtxt(test_path, delimiter=',')

    # (1) loading dataset
    dataframe, filename = loader(input_filepath)

    model = load_model('models/naples_model.keras')
    # (2) normalizing

    test(model, data, dataframe['label'])

    # (3) save it in models
    if output_filepath.endswith(os.path.sep):
        output_filepath = output_filepath[:-1]
    else:
        output_filepath

    logger = logging.getLogger(__name__)
    logger.info('Model has been: [1]tested')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
