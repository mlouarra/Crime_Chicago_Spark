crimes_chigaco
==============================

Prediction of crimes in Chicago

Project Organization
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
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

Configuration of Project Environment
*************************************

The aim of this script is to generate:
* classifier machine learning algorithm (random forest for classification)
* regression machine learning algorithm (gbt: Gradient Boosted Tree Regression mode)
* csv file results for regression and classification

To do this, the following two fields must be True (see file configuration)

* make_regression 
* make_classification



Overview on How to Run this script
================================
1. Install anaconda
2. Create a new environment : conda create --name <name_of_your_env>
3. Activate the new environment to use it (Linux: source activate <name_of_your_env>)
4. Install packages required: pip install -r requirements.txt
5. Execute main.py



Setup procedure
================

A. Configure yaml file
--------------------------------------------------------------------------------------------------------------------------------------
======================================================================================================================================

In this section, We will explain:

- the role of the different fields
- the values that each field can contain


###### With file configuration, it is possible to merge some crimes between them.

List_of_crimes_prediction:

    with_merge: False
    without_merged_pred: ['THEFT', 'BATTERY', 'CRIMINAL DAMAGE', 'NARCOTICS', 'BURGLARY', 'ASSAULT']
  
  to_merge:
  
    THEFT: "THEFT_ROBBERY_BURGLARY"
    ROBBERY: "THEFT_ROBBERY_BURGLARY"
    BURGLARY: "THEFT_ROBBERY_BURGLARY"
    ASSAULT: "ASSAULT_BATTERY"
    BATTERY: "ASSAULT_BATTERY"
    
  with_merge_pred: ["THEFT_ROBBERY_BURGLARY", "ASSAULT_BATTERY", "NARCOTICS"]
  
 
### Explanation:

* if the field with_merge is False, the crime to predict is list of without_merged_pred
* if the field with_merge is True, we merge the crimes (for this configuration)
   * "THEFT_ROBBERY_BURGLARY" in the same crime
   * "ASSAULT_BATTERY" in the same crime
 

#### with file configuration, we can choose classification or regression algorithm

    make_regression: True

    make_classificatin: True
   
#### the possibiliy to choose training model or load model for prediting if model exists

model_ML_classification:

  train_mode:
  
    train: True
    start_date: 2012
    end_date: 2013
    
  predict_mode:
  
    predict: True
    start_date: 2013
    end_date: 2014
    
  param:
  
    numTrees: [20,50]
    maxDepth: [5, 8]
    test_size: [0.9, 0.1]
    
  path:
  
    path_model_rf:  "../models/rfModel"
    path_results: "../reports/result_pred_classification.csv"
    
 
model_ML_regression:

  train_mode:
  
    train: True
    start_date: 2012
    end_date: 2013
    
  predict_mode:
  
    predict: True
    start_date: 2013
    end_date: 2014
    
  param:
  
    maxIter: 100
    maxDepth: 8
    
  path:
  
    path_model_regression: "../models/regression_Model"
    path_results: "../reports/result_pred_regression.csv"


  

 