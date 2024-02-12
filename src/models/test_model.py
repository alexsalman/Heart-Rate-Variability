# -*- coding: utf-8 -*-
import os
import sys
import click
import logging
import datetime
import numpy as np
import tensorflow as tf
import shap

sys.path.append('.')
from src.csv2df import csv2df
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix


# general model tester
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


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs model testing scripts on real data from (../interim) into
        metrics figure (saved in ../figures).
    """
# (1) loading trained model, and testing datas
    model = tf.keras.models.load_model('models/naples_model.keras')
    test_path = 'models/scaled_naples_test_features.csv'
    # Use genfromtxt to load the CSV file into a NumPy array
    data = np.genfromtxt(test_path, delimiter=',')
    dataframe, filename = csv2df(input_filepath)
    print(data.shape)
    print(dataframe.shape)
# (2) testing model
    sensitivity, specificity, auc, conf_matrix, class_report = test(model, data, dataframe['label'])
# (3) shap
#     # Use SHAP to explain the model's predictions
#     explainer = shap.DeepExplainer(model)
#     shap_values = explainer.shap_values(dataframe)
#
#     # Plot the SHAP values for a single prediction
#     shap.summary_plot(shap_values, dataframe, feature_names=['PNS index', 'SNS index', 'Stress index', 'Mean RR  (ms)', 'SDNN (ms)', 'Mean HR (beats/min)',
#                       'SD HR (beats/min)', 'Min HR (beats/min)', 'Max HR (beats/min)', 'RMSSD (ms)', 'NNxx (beats)',
#                       'pNNxx (%)', 'RR tri index', 'TINN (ms)', 'DC (ms)', 'DCmod (ms)', 'AC (ms)', 'ACmod (ms)',
#                       'VLF (Hz)', 'LF (Hz)', 'HF (Hz)', 'VLF (ms^2)', 'LF (ms^2)', 'HF (ms^2)', 'VLF (log)', 'LF (log)',
#                       'HF (log)', 'VLF (%)', 'LF (%)', 'HF (%)', 'LF (n.u.)', 'HF (n.u.)', 'Total power (ms^2)',
#                       'LF/HF ratio', 'RESP (Hz)', 'SD1 (ms)', 'SD2 (ms)', 'SD2/SD1 ratio', 'Approximate entropy (ApEn)',
#                       'Sample entropy (SampEn)', 'alpha 1', 'alpha 2', 'Correlation dimension (D2)',
#                       'Mean line length (beats)', 'Max line length (beats)', 'Recurrence rate (REC) (%)',
#                       'Determinism (DET) (%)', 'Shannon entropy', 'MSE(1)', 'MSE(2)', 'VLF (Hz) AR spectrum',
#                       'LF (Hz) AR spectrum', 'HF (Hz) AR spectrum', 'VLF (ms^2) AR spectrum', 'LF (ms^2) AR spectrum',
#                       'HF (ms^2) AR spectrum', 'VLF (log) AR spectrum', 'LF (log) AR spectrum', 'HF (log) AR spectrum',
#                       'VLF (%) AR spectrum', 'LF (%) AR spectrum', 'HF (%) AR spectrum', 'LF (n.u.) AR spectrum',
#                       'HF (n.u.) AR spectrum'])
# (4) save it in /reports/figures
    if output_filepath.endswith(os.path.sep):
        output_filepath = output_filepath[:-1]
    else:
        output_filepath

    # Generate a timestamp for the file name
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # Create a file name with the timestamp
    file_name = f'testing_results_{timestamp}.txt'
    # Open a file in write mode
    with open(output_filepath + '/' + file_name, 'w') as file:
        file.write('Sensitivity: ' + str(round(sensitivity, 2)) + '\n')
        file.write('Specificity: ' + str(round(specificity, 2)) + '\n')
        file.write('Area Under Curve: ' + str(round(auc, 2)) + '\n')
        file.write('Confusion Matrix:\n' + str(conf_matrix) + '\n')
        file.write('\nClassification Report:\n' + str(class_report) + '\n')

    logger = logging.getLogger(__name__)
    logger.info(
        '*** Model is tested ***')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
