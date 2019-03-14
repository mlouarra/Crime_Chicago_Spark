
import pandas as pd
import datetime as dt
from sklearn.preprocessing import LabelEncoder
from pyspark.sql.types import DoubleType
from astral import Astral
import re

class extract_features_classification:

    city_name = 'Chicago'
    a = Astral()
    a.solar_depression = 'civil'
    city = a[city_name]

    def __init__(self, config, df_crime_socio, df_temperature, df_sky):
        """

        :param config:
        :param df_crime_socio:
        :param df_temperature:
        :param df_sky:
        """
        self._config = config
        self._df_crime_socio = df_crime_socio
        self._df_temperature = df_temperature
        self._df_sky = df_sky


    def list_of_crimes(self):
        """

        :return:
        """
        return self._config["List_of_crimes_prediction"]["with_merge_pred"] \
            if self._config["List_of_crimes_prediction"]["with_merge"] else \
            self._config["List_of_crimes_prediction"]["without_merged_pred"]

    def list_to_drop(self):
        """

        :return:
        """
        return self._config["List_to_drop"]



    def duration_day(self, date):
        """

        :return:
        """

        sun = self.city.sun(date=date, local=True)
        return (sun['sunset'] - sun['sunrise']).total_seconds()


    def extract_feature(self):
        """
        this function extract features for machine learning algorithm
        """
        category = LabelEncoder()
        pd.options.mode.chained_assignment = None
        df_crime_socio = self._df_crime_socio[self._df_crime_socio.primary_type.isin(self.list_of_crimes())]
        df_crime_socio['extract_block'] = df_crime_socio.block.apply(lambda x: re.findall(r"(\w+)$", x)[0])
        df_crime_socio['isStreet'] = df_crime_socio.extract_block.apply(lambda x: self.isStreet(x))
        df_crime_socio['isAV'] = df_crime_socio.extract_block.apply(lambda x: self.isAV(x))
        df_crime_socio['isBLVD'] = df_crime_socio.extract_block.apply(lambda x: self.isBLVD(x))
        df_crime_socio['isRD'] = df_crime_socio.extract_block.apply(lambda x: self.isRD(x))
        df_crime_socio['isPL'] = df_crime_socio.extract_block.apply(lambda x: self.isPL(x))
        df_crime_socio['isBROADWAY'] = df_crime_socio.extract_block.apply(lambda x: self.isBROADWAY(x))
        df_crime_socio['isPKWY'] = df_crime_socio.extract_block.apply(lambda x: self.isPKWY(x))
        df_crime_socio.drop("extract_block", inplace=True, axis=1)
        df_crime_socio['duree_day'] = df_crime_socio['date'].apply(lambda x: self.duration_day(x))
        df_crime_socio['month'] = df_crime_socio['date'].dt.month
        df_crime_socio['day'] = df_crime_socio['date'].dt.day
        df_crime_socio['hours'] = df_crime_socio['date'].dt.hour
        df_crime_socio['minutes'] = df_crime_socio['date'].dt.minute
        df_crime_socio['dayofweek'] = df_crime_socio['date'].apply(lambda x: dt.datetime.strftime(x, '%A'))
        df_crime_socio['XY'] = df_crime_socio.x_coordinate * df_crime_socio.y_coordinate
        df_crime_socio['Category'] = category.fit_transform(df_crime_socio.primary_type)
        df = pd.get_dummies(df_crime_socio, columns=['dayofweek'])
        # df_crime_socio_ml = pd.get_dummies(df, columns=['community_area_name', 'domestic'])

        df_crime_socio_ml = pd.get_dummies(df, columns=['domestic'])
        del df
        del df_crime_socio
        df_ml = pd.merge(df_crime_socio_ml, self._df_temperature, how='left', on=['month', 'day', 'hours'])
        del df_crime_socio_ml
        df_ml = pd.merge(df_ml, self._df_sky, how='left', on=['month', 'day', 'hours'])
        df_ml.dropna(inplace=True)
        return df_ml.drop(self.list_to_drop(), axis=1), list(category.classes_)

class  extract_features_regression():

    def __init__(self, config, df_nb_crimes):
        """

        :param config:
        :param df_nb_crimes:
        """
        self._config = config
        self._df_nb_crimes = df_nb_crimes

    def extract_feature(self):
        """

        :return:
        """

        df_ml = self._df_nb_crimes.drop(*['community_area_number', 'month', 'year',
                                          'primary_type', 'community_area_name'])
        column_names_to_normalize = ['pct_housing_crowded', 'pct_households_below_poverty', 'pct_age16_unemployed',
                                      'pct_age25_no_highschool', 'pct_not_working_age', 'per_capita_income',
                                     'hardship_index']
        df_ml = df_ml.withColumnRenamed('nb_crimes', 'label')
        for f in column_names_to_normalize:
            df_ml = df_ml.withColumn(f, df_ml[f].cast(DoubleType()))
        return df_ml