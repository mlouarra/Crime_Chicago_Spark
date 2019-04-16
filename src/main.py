import os
import sys
from yaml import load as yaml_load
from pyspark.sql import SparkSession
from src.data.make_dataset import LoadDataframe
from src.features.build_features import  extract_features_regression, extract_features_classification
from src.models.train_model import model_classification, model_regression
from src.models.predict_model import predict_model

def _load_config_file(config_file):
    """
    Load configuration file
    :param config_file: is the configuration file
    :return: configuration
    :rtype: dict
    """
    with open(config_file) as yml_config:
        return yaml_load(yml_config)

def _build_configuration(config_file):
    """
    Build the operation configuration dict
    :param config_file: is the path to the yaml config_file
    :type: string
    :return: config: global configuration
    :rtype dict
    """
    # yaml config
    config = _load_config_file(config_file)
    return config


def main(config_file='/home/ml/Documents/Crime_Chigaco_Spark/config/config.yml'):
    """
    Script entry point function
    """
    # Check if configuration file parameter is set and if the file exists
    from pyspark.ml.classification import RandomForestClassifier, RandomForestClassificationModel
    if not config_file:
        print('Configuration file is mandatory, abort')
        sys.exit(1)
    elif not os.path.isfile(config_file):
        print('"{}" configuration file not exists, abort'.format(config_file))
        sys.exit(1)

    # Load configuration from YAML file
    # build configuration

    config = _build_configuration(config_file)
    if config['make_classificatin']:
        # to train model classifcation
        if config['model_ML_classification']['train_mode']['train']:
            start_date_train = config['model_ML_classification']['train_mode']['start_date']
            end_date_train = config['model_ML_classification']['train_mode']['end_date']
            obj_df_loaded = LoadDataframe(config, start_date_train, end_date_train)
            df_crimes_socio = obj_df_loaded.df_crime_socio()
            df_temp = obj_df_loaded.df_temperature()
            df_sky = obj_df_loaded.df_sky()
            obj_extract_features_classification = extract_features_classification(config, df_crimes_socio, df_temp, df_sky)
            df_ml = obj_extract_features_classification.extract_feature()
            obj_model_classification = model_classification(config, df_ml)
            rf_model = obj_model_classification.train_RF()
            rfPath = config['model_ML_classification']['path']['path_model_rf']
            rf_model.save(rfPath)
        # to predict result
        if config['model_ML_classification']['predict_mode']['predict']:
            start_date_train = config['model_ML_classification']['predict_mode']['start_date']
            end_date_train = config['model_ML_classification']['predict_mode']['end_date']
            obj_df_loaded = LoadDataframe(config, start_date_train, end_date_train)
            df_crimes_socio = obj_df_loaded.df_crime_socio()
            df_temp = obj_df_loaded.df_temperature()
            df_sky = obj_df_loaded.df_sky()
            obj_extract_features_classification = extract_features_classification(config, df_crimes_socio, df_temp, df_sky)
            df_ml = obj_extract_features_classification.extract_feature()
            obj_predict_model = predict_model(config)
            obj_predict_model.predict_classification(df_ml)

    if config['make_regression']:
        if config['model_ML_regression']['train_mode']['train']:
            start_date_train = config['model_ML_regression']['train_mode']['start_date']
            end_date_train = config['model_ML_regression']['train_mode']['end_date']
            obj_df_loaded = LoadDataframe(config, start_date_train, end_date_train)
            df_nb_crimes = obj_df_loaded.df_nb_crimes()
            obj_extract_features_regression = extract_features_regression(config, df_nb_crimes)
            df_ml = obj_extract_features_regression.extract_feature()
            obj_model_regression = model_regression(config, df_ml)
            regression_model = obj_model_regression.train_gbt()
            regressionPath = config['model_ML_regression']['path']['path_model_regression']
            regression_model.save(regressionPath)
            # to predict result
        if config['model_ML_regression']['predict_mode']['predict']:
            start_date_train = config['model_ML_regression']['predict_mode']['start_date']
            end_date_train = config['model_ML_regression']['predict_mode']['end_date']
            obj_df_loaded = LoadDataframe(config, start_date_train, end_date_train)
            df_nb_crimes = obj_df_loaded.df_nb_crimes()
            obj_extract_features_regression = extract_features_regression(config, df_nb_crimes)
            df_ml = obj_extract_features_regression.extract_feature()
            obj_predict_model = predict_model(config)
            obj_predict_model.predict_regression(df_ml)


if __name__ == "__main__":
    spark = SparkSession \
        .builder \
        .appName('classification_model') \
        .getOrCreate()

    print('Session created')

    if len(sys.argv) != 1:
        main(sys.argv[1])
    else:
        main()
        spark.stop()








