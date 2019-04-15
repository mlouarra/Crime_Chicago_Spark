
from pyspark.ml import PipelineModel

class predict_model:
    """
    this class predicts crimes type and creates a csv file containing the results
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

        :return: create csv file with prediction
        """
        # load model

        RFModel = PipelineModel.load('../models/rfModel')
        # predict from df_ml_test
        df_prediction = RFModel.transform(df_ml_test)
        # save results
        df_prediction = df_prediction.select('primary_type', 'label', 'prediction', 'predictedLabel')
        print('write results in csv')
        df_prediction.toPandas().to_csv(self._config['model_ML_classification']['path']['path_results'])
        pass

    def predict_regression(self, df_ml_test):
        """
        This class predicts numbers of crimes by type and region. It creates a csv file containing the results
        :return:
        """
        regression_model = PipelineModel.load("../models/regression_Model")
        df_prediction = regression_model.transform(df_ml_test)
        # save results
        df_prediction = df_prediction.select('label', 'prediction')
        print('write results in csv')
        df_prediction.toPandas().to_csv(self._config['model_ML_regression']['path']['path_results'])

        pass


#