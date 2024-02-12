Heart Rate Variability (HRV) Processing
==============================

![Framework](framework2.png)
*Schematic Overview of HRV Classification Pipeline.*

🔍 Overview
------------
*In this study, we tackle the processing of patients’ heart rate variability (HRV)
datasets for two classification tasks employing resource-efficient methods,
synthetic data generation, deep learning techniques, and transfer learning.* 

## 📑 Table of Contents

- [License](#license)
- [Installation](#installation)
- [Usage](#usage)
- [Project Organization](#project-organization)

## 📜 License
You are free to use, modify, and distribute this project for any purpose. However, if you use this project in your work or derive from it, you must provide appropriate attribution by referencing this repository.

Refer to our, [MIT License](https://github.com/alexsalman/heart_rate_variability?tab=MIT-1-ov-file).

## 🔧 Installation
##### Clone the repository
```
git clone https://github.com/alexsalman/heart_rate_variability.git
```
##### Change directory
```
cd heart_rate_variability
```
##### Create conda environment 
```
conda create -n <env-name>
```
##### Activate conda environment 
```
source activate <env-name>
```
##### Install software requirements
``
pip install -r requirements.txt
``

## 🚀 Usage
##### Data processing
###### Load datasets, undersample records, select features, shuffle, and split into train/test 
```
python src/data/make_dataset.py data/raw/<Atrial Fibrillation Data> data/interim/
```
```
python src/data/make_dataset.py data/raw/<Cardiovascular Event Data> data/interim/
```
##### Synthetic data generation
###### Create metadata for training sets and data generator objects, models fitting, and sample synthetic records
```
python src/data/generate_synthetic.py data/interim/<Atrial Fibrillation Training Data> data/processed/
```
```
python src/data/generate_synthetic.py data/interim/<Cardiovascular Event Training Data> data/processed/
```
##### Deep learning model: Atrial Fibrillation task
###### Train fully connected neural network model on synthetic data
```
python src/models/train_model.py data/processed/<Atrial Fibrillation Synthetic Training Data> models/
```
###### Test model on real data of Atrial Fibrillation
```
python src/models/test_model.py data/interim/<Atrial Fibrillation Real Test Data> reports/figures/
```
##### Deep/ transfer learning model: Cardiovascular Event task
###### Train fully connected neural network model on synthetic + real data
```
python src/models/train_model.py data/processed/<Cardiovascular Event Synthetic/Real Training Data> models/
```
###### Test model on real data of Cardiovascular Event
```
python src/models/test_model.py data/interim/<Cardiovascular Event Real Test Data> reports/figures/
```
## 📁 Project Organization

------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
