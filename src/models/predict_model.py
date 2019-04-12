
class predict_model:

    def __init__(self, config, df_ml_test):

        self._config = config
        self._df_ml_test = df_ml_test

    def predict_classification(self):
        # load model
        from pyspark.ml import PipelineModel
        RFModel = PipelineModel.load('../models/rfModel')
        # predict from df_ml_test
        df_prediction = RFModel.transform(self._df_ml_test)
        # save results
        df_prediction = df_prediction.select('primary_type', 'label', 'prediction', 'predictedLabel')
        print('write results in csv')
        df_prediction.toPandas().to_csv(self._config['model_ML_classification']['path']['path_results'])
        pass

    def predict_regression(self):

        pass


#