# -*- coding: utf-8 -*-
import os
import sys
import shap
import click
import logging
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

from sklearn.model_selection import StratifiedKFold

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
    # num_folds = 10  # Adjust as needed
    # k_fold = StratifiedKFold(n_splits=num_folds, shuffle=False)
    #
    # best_model = None
    # best_metrics = {
    #     'accuracy': 0.0,
    #     'recall_class_0': 0.0,
    #     'recall_class_1': 0.0,
    #     'precision_class_0': 0.0,
    #     'precision_class_1': 0.0,
    #     'f1_class_0': 0.0,
    #     'f1_class_1': 0.0
    # }
    #
    # for fold_num, (train_idx, val_idx) in enumerate(k_fold.split(x_train, y_train)):
    #     x_val, y_val = x_train[val_idx], y_train.iloc[val_idx]
    #     # random seed for reproducibility.
    #     tf.keras.utils.set_random_seed(43)
    #     # model architecture
    #     model = Sequential()
    #     # input layer
    #     model.add(Dense(64, input_dim=x_train.shape[1], name='DL_input_layer'))
    #     model.add(Activation('relu', name='DL_activation'))
    #     # hidden layer 1
    #     model.add(Dense(64, name='DL_hidden_layer_1'))
    #     model.add(Activation('relu'))
    #     # dropout layer 1 (regularization to prevent overfitting)
    #     model.add(Dropout(0.5, name='DL_dropout_1'))
    #     # hidden layer 2
    #     model.add(Dense(64, name='DL_hidden_layer_2'))
    #     model.add(Activation('relu'))
    #     # dropout layer 2 (regularization to prevent overfitting)
    #     model.add(Dropout(0.5, name='DL_dropout_2'))
    #     # output layer
    #     model.add(Dense(1, activation='sigmoid', name='DL_output_layer'))
    #     # print the model summary to view layer names and shapes
    #     model.summary()
    #     # model compilation
    #     model.compile(optimizer=Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])
    #     # model training
    #     model.fit(x_train, y_train, epochs=50, batch_size=32, validation_split=0.2, shuffle=False)
    #
    #     # Evaluate on the validation set
    #     predictions = model.predict(x_val)
    #
    #     # Apply threshold to convert to binary format
    #     threshold = 0.5
    #     binary_predictions = (predictions > threshold).astype(int)
    #     report = classification_report(y_val, binary_predictions, target_names=['Class 0', 'Class 1'], output_dict=True)
    #
    #     # Extract metrics for both classes
    #     accuracy = report['accuracy']
    #     recall_class_0 = report['Class 0']['recall']
    #     recall_class_1 = report['Class 1']['recall']
    #     precision_class_0 = report['Class 0']['precision']
    #     precision_class_1 = report['Class 1']['precision']
    #     f1_class_0 = report['Class 0']['f1-score']
    #     f1_class_1 = report['Class 1']['f1-score']
    #
    #     # Update best metrics if the current fold has higher values
    #     if accuracy > best_metrics['accuracy'] and recall_class_0 > best_metrics['recall_class_0'] \
    #             and recall_class_1 > best_metrics['recall_class_1'] and precision_class_0 > best_metrics['precision_class_0'] \
    #             and precision_class_1 > best_metrics['precision_class_1'] and f1_class_0 > best_metrics['f1_class_0'] \
    #             and f1_class_1 > best_metrics['f1_class_1']:
    #         best_model = model
    #         best_metrics = {
    #             'accuracy': accuracy,
    #             'recall_class_0': recall_class_0,
    #             'recall_class_1': recall_class_1,
    #             'precision_class_0': precision_class_0,
    #             'precision_class_1': precision_class_1,
    #             'f1_class_0': f1_class_0,
    #             'f1_class_1': f1_class_1
    #         }
    #
    # # Train the best model on the entire dataset
    # best_model.fit(x_train, y_train, epochs=50, batch_size=32, validation_split=0.2, shuffle=False)
    # random seed for reproducibility.
    tf.keras.utils.set_random_seed(43)
    # model architecture
    model = Sequential()
    # input layer
    model.add(Dense(64, input_dim=x_train.shape[1], name='DL_input_layer'))
    model.add(Activation('relu', name='DL_activation'))
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
    x_train = sc.fit_transform(dataframe.drop(['label'], axis=1))
    x_test = sc.transform(test_dataframe.drop(['label'], axis=1))
    y_train = dataframe['label']
    y_test = test_dataframe['label']
    print(x_train.shape)
    print(x_test.shape)
# (3) training fully connected neural network model
    model = fc_nn(x_train, y_train)
    test(model, x_test, y_test)

    # Use SHAP to explain the model's predictions
    explainer = shap.Explainer(model, masker=shap.maskers.Independent(data=x_train))
    shap_values = explainer.shap_values(x_test)
    shap.summary_plot(shap_values, x_test, feature_names=['PNS index', 'SNS index', 'Stress index', 'Mean RR  (ms)', 'SDNN (ms)', 'Mean HR (beats/min)',
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
                      'HF (n.u.) AR spectrum'])
    # plt.savefig('reports/figures/SHAP_AF.png', dpi=800)


# (4) save it in models
    if output_filepath.endswith(os.path.sep):
        output_filepath = output_filepath[:-1]
    else:
        output_filepath
    path = output_filepath + '/' + 'florance_model.keras'
    model.save(path, save_format="tf")
    np.savetxt(output_filepath + '/' + 'scaled_florance_test_features.csv', x_test, delimiter=',')

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
