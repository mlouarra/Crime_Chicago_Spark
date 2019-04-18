from pyspark.ml import PipelineModel


class predict_model:
    """
    this class loads model of predicting for classification or regression if the model exists
    """

    def __init__(self, config):
        """

        :param config: dict for configuration
        :param df_ml_test: dataframe for predicting crimes; start date and end date are configured
        in file for configuration (see readme.md file)
        """

        self._config = config

    def predict_classification(self, df_ml_test):
        """
        this method loads model and saves results in csv file prediction
        :param df_ml_test: the dataframe for predicting
        :return: None
        """
        # load model
        rf_path = self._config['model_ML_classification']['path']['path_model_rf']
        rf_model = PipelineModel.load(rf_path)
        # predict from df_ml_test
        df_prediction = rf_model.transform(df_ml_test)
        # save results
        df_prediction = df_prediction.select('primary_type', 'label', 'prediction', 'predictedLabel')
        print('write results in csv')
        df_prediction.toPandas().to_csv(self._config['model_ML_classification']['path']['path_results'])
        pass

    def predict_regression(self, df_ml_test):
        """
        this method loads model and saves csv results for prediction
        :param df_ml_test: dataframe for predicting
        :return:
        """
        gbt_path = self._config['model_ML_regression']['path']['path_model_regression']
        regression_model = PipelineModel.load(gbt_path)
        df_prediction = regression_model.transform(df_ml_test)
        # save results
        df_prediction = df_prediction.select('label', 'prediction')
        print('write results in csv')
        df_prediction.toPandas().to_csv(self._config['model_ML_regression']['path']['path_results'])
