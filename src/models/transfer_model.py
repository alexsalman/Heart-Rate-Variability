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
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout, Activation
from keras.optimizers import Adam
from tensorflow.python.keras.models import load_model


def loader(input_filepath):
    # Extract the name of the CSV file (excluding the extension)
    filename = os.path.splitext(os.path.basename(input_filepath))[0]
    # Load the CSV file into a DataFrame
    dataframe = pd.read_csv(input_filepath)
    return dataframe, filename


# fully connected neural network
def transfer(model, x_train, y_train):
    tf.keras.utils.set_random_seed(43)

    for layer in model.layers[:-1]:
        print('1')
        layer.trainable = False

    florance = Sequential()
    for layer in model.layers[:-4]:
        florance.add(layer)
    florance.add(Dense(32, activation='relu', name='new_layer1'))
    # florance.add(BatchNormalization(name='new_layer2'))

    # florance.add(Dense(16, activation='relu', name='alex'))

    florance.add(Dense(1, activation='sigmoid', name='new_layer9'))
    florance.summary()
    florance.compile(optimizer=Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])
    florance.fit(x_train, y_train, epochs=40, batch_size=32, validation_split=0.2, shuffle=False)

    return florance


def test(model, x_test, y_test):
    y_probabilities = model.predict(x_test)
    # threshold adjustment
    threshold = 0.5
    # Convert probabilities to binary predictions based on the threshold
    y_predictions = (y_probabilities >= threshold).astype(int)
    # Evaluate the model with the adjusted threshold
    conf_matrix = confusion_matrix(y_test, y_predictions)
    class_report = classification_report(y_test, y_predictions)
    auc = roc_auc_score(y_test, y_predictions)
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

    # (1) loading dataset

    test_path = 'data/interim/dataset_naples_testing.csv'
    dataframe, filename = loader(input_filepath)
    dataframe_test, filename_test = loader(test_path)
    # (2) loading Task1 model
    model = tf.keras.models.load_model('models/florance_model.keras')
    # (3) normalizing
    sc = StandardScaler()
    x_train = sc.fit_transform(dataframe.drop('label', axis=1))
    x_test = sc.transform(dataframe_test.drop('label', axis=1))
    y_train = dataframe['label']
    y_test = dataframe_test['label']

    model = transfer(model, x_train, y_train)
    test(model, x_test, y_test)

    # (3) save it in models
    if output_filepath.endswith(os.path.sep):
        output_filepath = output_filepath[:-1]
    else:
        output_filepath
    path = output_filepath + '/' + 'naples_model.keras'
    model.save(path)
    # # np.savetxt(output_filepath + '/' + 'test_features.csv', x_test, delimiter=',')
    logger = logging.getLogger(__name__)
    logger.info('Model has been: [1]transfered')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
