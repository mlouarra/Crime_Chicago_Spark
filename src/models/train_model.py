import xgboost as xgb
from sklearn import svm
import pandas as pd
from sklearn.model_selection import train_test_split
from pyspark.ml.feature import StandardScaler, VectorAssembler, VectorIndexer, StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from pyspark.ml.regression import GBTRegressor
from pyspark.ml import Pipeline


class model_classification:

    def __init__(self, config, df_ml):
        """

        :param config:
        :param df_crime_socio:
        """
        self._config = config
        self._df_ml = df_ml

    def set_param(self):

        return self._config['Xgboost_param_classification']

    def load_for_ml(self):

        """

        :param :
        :return: list of dataframes for machine learning
        """
        pd.options.mode.chained_assignment = None

        if len(self._config['delete_features']) != 0:
            df_ml = self._df_ml[0].drop(columns=self._config['delete_features'])
        else:
            df_ml = self._df_ml[0]
        class_name = self._df_ml[1]
        feature_cols = [x for x in df_ml if x != 'Category']
        X = df_ml[feature_cols]
        y = df_ml['Category']
        X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        return X_train, x_test, y_train, y_test, class_name

    def dtrain(self):
        """

        :return:
        """
        X_train = self.load_for_ml()[0]
        y_train = self.load_for_ml()[2]
        return xgb.DMatrix(X_train, label=y_train)


    def dtest(self):
        """

        :return:
        """
        x_test = self.load_for_ml()[1]
        # y_test = self.load_for_ml()[3]
        return xgb.DMatrix(x_test)

    def train_xgboost_(self, params):
        """

        :param df_crime_socio:
        :return:
        """
        X_train = self.load_for_ml()[0]
        y_train = self.load_for_ml()[2]
        xgb_clas = xgb.XGBClassifier()
        gs = RandomizedSearchCV(xgb_clas, params, n_jobs=1, n_iter=5, cv=5)
        gs.fit(X_train, y_train)
        return gs.best_estimator_

    def train_xgboost(self):
        """

        :param df_crime_socio:
        :return:
        """

        dtrain = self.dtrain()
        # dtest = self.dtest()

        return self._config['connect']['PathTemperature']

    def path_sky(self):

        return self._config['connect']['PathTemperature']


    def train_svm(self):
        """

        :return:
        """
        C = 1.0  # SVM regularization parameter
        X = self.load_for_ml()[0]
        y = self.load_for_ml()[2]
        svc = svm.SVC(kernel='linear', C=C).fit(X, y)
        return svc

    def train_rf(self, param):
        """

        :return:
        """

class model_regression:

    def __init__(self, config, df_ml):
        """

        :param config:
        :param df_ml:
        """
        self._config = config
        self._df_ml = df_ml


    def train_model(self):

        """

        :return:
        """
        features = self._df_ml.columns
        features.remove('label')
        vectorAssembler = VectorAssembler(inputCols=features, outputCol="unscaled_features")
        standardScaler = StandardScaler(inputCol="unscaled_features", outputCol="features")
        gbt = GBTRegressor(featuresCol="features", maxIter=100, maxDepth=8)
        stages = [vectorAssembler, standardScaler, gbt]
        pipeline = Pipeline(stages=stages)
        model = pipeline.fit(self._df_ml)
        return model




