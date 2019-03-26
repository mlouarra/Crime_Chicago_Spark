
import pandas as pd
import datetime as dt
from sklearn.preprocessing import LabelEncoder
from pyspark.sql.types import LongType, StringType, FloatType, IntegerType, DoubleType
from pyspark.sql.functions import col, pandas_udf, udf
import re

class extract_features_classification:
    """

    """

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


    def duration_day_func(x):
        """
        :return:
        """
        from astral import Astral
        city_name = 'Chicago'
        a = Astral()
        a.solar_depression = 'civil'
        city = a[city_name]
        sun = city.sun(date=x, local=True)
        return float((sun['sunset'] - sun['sunrise']).total_seconds())

    def extract_feature(self):
        """
        this function extract features for machine learning algorithm
        """

        extract_blok = udf(lambda x: re.findall(r"(\w+)$", x)[0], StringType())
        isStreet = udf(lambda x: 1 if x in ['ST', 'St', 'st'] else 0, IntegerType())
        isAV = udf(lambda x: 1 if x in ['Ave', 'AV', 'AVE'] else 0, IntegerType())
        isBLVD = udf(lambda x: 1 if x in ['BLVD'] else 0, IntegerType())
        isRD = udf(lambda x: 1 if x in ['RD'] else 0, IntegerType())
        isPL = udf(lambda x: 1 if x in ['PL', 'pl'] else 0, IntegerType())
        isBROADWAY = udf(lambda x: 1 if x in ['BROADWAY', 'Broadway'] else 0, IntegerType())
        isPKWY = udf(lambda x: 1 if x in ['PKWY', 'Pkwy'] else 0, IntegerType())
        duration_day_udf = udf(lambda x: self.duration_day_func(x), FloatType())



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