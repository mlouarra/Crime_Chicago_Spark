import os
import sys
from yaml import load as yaml_load
from pyspark.sql import SparkSession
from src.data.make_dataset import LoadDataframe
from src.features.build_features import extract_features_classification
from src.models.train_model import model_classification


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


def main(config_file='/home/ml/Documents/crimes_chigaco/config/config.yml'):
    """
    Script entry point function
    """
    # Check if configuration file parameter is set and if the file exists
    if not config_file:
        print('Configuration file is mandatory, abort')
        sys.exit(1)
    elif not os.path.isfile(config_file):
        print('"{}" configuration file not exists, abort'.format(config_file))
        sys.exit(1)

    # Load configuration from YAML file
    # build configuration
    config = _build_configuration(config_file)
    obj_df_loaded = LoadDataframe(config, '2013', '2014')
    df_crimes_socio = obj_df_loaded.df_crime_socio()
    df_temp = obj_df_loaded.df_temperature()
    df_sky = obj_df_loaded.df_sky()
    obj_extract_features_classification = extract_features_classification(config, df_crimes_socio, df_temp, df_sky)
    df_ml = obj_extract_features_classification.extract_feature()
    obj_model_classification = model_classification(config, df_ml)
    rf_model = obj_model_classification.train_RF()
    df_test = obj_model_classification.df_test()
    df_prediction = rf_model.transform(df_test)
    print(df_prediction.select('primary_type', 'label', 'prediction', 'predictedLabel').limit(10).toPandas())


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








