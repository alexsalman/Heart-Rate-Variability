# -*- coding: utf-8 -*-
import os
import sys
import click
import logging
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

sys.path.append('.')
from src.csv2df import csv2df
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import Adam

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
    return sensitivity, specificity, auc, conf_matrix, class_report


# fully connected neural network
def fc_nn(x_train, y_train):
    # random seed for reproducibility.
    tf.keras.utils.set_random_seed(43)
    # model architecture
    model = Sequential()
    # input layer
    model.add(Dense(64, input_dim=x_train.shape[1], name='DL_input_layer'))
    model.add(Activation('relu'))
    # hidden layer 1
    model.add(Dense(64, name='DL_hidden_layer_1'))
    model.add(Activation('relu'))
    # dropout layer 1 (regularization to prevent overfitting)
    model.add(Dropout(0.5, name='DL_dropout_1'))
    # hidden layer 2
    model.add(Dense(64, name='DL_hidden_layer_2'))
    model.add(Activation('relu'))
    # dropout layer 2 (regularization to prevent overfitting)
    model.add(Dropout(0.5, name='DL_dropout_2'))
    # output layer
    model.add(Dense(1, activation='sigmoid', name='DL_output_layer'))
    # print the model summary to view layer names and shapes
    model.summary()
    # model compilation
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])
    # model training
    model.fit(x_train, y_train, epochs=50, batch_size=32, validation_split=0.2, shuffle=False)

    return model


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs model training scripts on synthetic data from (../processed) into
        a model and scaled test data into (saved in ../models)
    """
# (1) loading synthetic dataset and real test data for scaling
    dataframe, filename = csv2df(input_filepath)
    test_dataframe, test_filename = csv2df('data/interim/dataset_florance_testing.csv')
# (2) normalizing
    sc = StandardScaler()
    x_train = sc.fit_transform(dataframe.drop('label', axis=1))
    x_test = sc.transform(test_dataframe.drop('label', axis=1))
    y_train = dataframe['label']
    y_test = test_dataframe['label']
# (3) training fully connected neural network model
    model = fc_nn(x_train, y_train)
    test(model, x_test, y_test)
# (4) save it in models
    if output_filepath.endswith(os.path.sep):
        output_filepath = output_filepath[:-1]
    else:
        output_filepath
    path = output_filepath + '/' + 'naples_model.keras'
    model.save(path, save_format="tf")
    # np.savetxt(output_filepath + '/' + 'scaled__test_features.csv', x_test, delimiter=',')

    logger = logging.getLogger(__name__)
    logger.info(
        '*** Model is trained ***'
    )


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
