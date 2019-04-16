# -*- coding: utf-8 -*-
import pandas as pd
import json
from pyspark.sql import SparkSession
import pyspark.sql.functions as func
from pyspark.sql.types import FloatType

spark = SparkSession.builder.master("local").appName("Data cleaning").getOrCreate()
pd.options.mode.chained_assignment = None


class LoadDataframe:
    """
    This class loads csv files. it is initialized by :

    \n config file
     \n start_year: for example 2013
     \n end_year: for example 2015

    """

    def __init__(self, config, start_year, end_year):
        """

        :param config: dict for configuration
        :param start_year: string or integer
        :param end_year: string or integer
        """

        self._config = config
        self._start_year = pd.to_datetime(str(start_year))
        self._end_year = pd.to_datetime(str(end_year))

    def path_crime(self):
        """

        :return: return the path of csv crimes of chicago
        """
        return self._config['connect']['PathCrimes']

    def path_socio(self):
        """

        :return: return the path of csv socio-economic
        """
        return self._config['connect']['PathSocioEco']

    def path_columns(self):
        """

        :return: the path of json file (we use this json to rename data crime columns and data socio-economic
        """
        return self._config['connect']['Pathcolumns']

    def path_temperature(self):
        """

        :return:  the path of temperature csv file
        """

        return self._config['connect']['PathTemperature']

    def path_sky(self):
        """

        :return:  the path of weather description  csv file
        """
        return self._config['connect']['PathSky']

    def get_dummies(self, df, list_columns):
        """
        Convert categorical variable into dummy/indicator variables

        :param df: spark dataframe
        :param list_columns: list of columns in dataframe to be transformed into zero
        :return: dummy spark dataframe
        """

        def join_all(dfs, keys):
            if len(dfs) > 1:
                return dfs[0].join(join_all(dfs[1:], keys), on=keys, how='inner')
            else:
                return dfs[0]

        combined = []
        pivot_cols = list_columns
        keys = df.columns

        for pivot_col in pivot_cols:
            pivotDF = df.groupBy(keys).pivot(pivot_col).count()
            new_names = pivotDF.columns[:len(keys)] + ["{0}_{1}".format(pivot_col, c)
                                                       for c in pivotDF.columns[len(keys):]]
            df = pivotDF.toDF(*new_names).fillna(0)
            combined.append(df)
        df_result = join_all(combined, keys)

        return df_result

    def df_crime(self):
        """
        return dataframe of crimes, columns are renamed

        :return: spark dataframe
        """
        column_name = json.loads(open(self.path_columns()).read())
        df_crimes = spark.read.format("csv").option("header", "true"). \
            option("mode", "DROPMALFORMED").option("delimiter", ";").load(self.path_crime())
        for old_name, new_name in column_name['DataCrimes'].items():
            df_crimes = df_crimes.withColumnRenamed(old_name, new_name)
        for f in ['latitude', 'longitude']:
            df_crimes = df_crimes.withColumn(f, df_crimes[f].cast(FloatType()))

        df_crimes = df_crimes.withColumn("date", func.to_timestamp("date", "MM/dd/yyyy hh:mm:ss aaa"))
        df_crimes = df_crimes.filter((func.col('date') > self._start_year) & (func.col('date') < self._end_year))
        if self._config["List_of_crimes_prediction"]["with_merge"]:
            df_crimes = df_crimes.na.replace(self._config["List_of_crimes_prediction"]["to_merge"], 'primary_type')
            return df_crimes
        else:
            return df_crimes

    def df_socio(self):
        """
        return socio-economic dataframe
        :return: spark dataframe
        """
        column_name = json.loads(open(self.path_columns()).read())
        df_socio = spark.read.format("csv").option("header", "true"). \
            option("mode", "DROPMALFORMED").option("delimiter", ",").load(self.path_socio())

        for old_name, new_name in column_name['SocioEco'].items():
            df_socio = df_socio.withColumnRenamed(old_name, new_name)

        features_socio = ['pct_housing_crowded',
                          'pct_households_below_poverty', 'pct_age16_unemployed',
                          'pct_age25_no_highschool',
                          'pct_not_working_age',
                          'per_capita_income',
                          'hardship_index']

        for f in features_socio:
            df_socio = df_socio.withColumn(f, df_socio[f].cast(FloatType()))
        return df_socio

    def df_crime_socio(self):
        """
        retun spark dataframe merged between crimes dataframe and socio-economic

        we use community_area_number column for merging

        :return: spark dataframe
        """
        df_crime_socio = self.df_crime().join(self.df_socio(), ['community_area_number'], "inner")
        df_crime_socio = df_crime_socio.na.drop()
        return df_crime_socio

    def df_nb_crimes(self):

        """
        Spark  dataframe with number of crimes by type and region for month

        :return: spark dataframe
        """
        df_S = self.df_socio()
        df_C = self.df_crime()
        df_C = df_C.filter(func.col('primary_type').isin(self._config['List_of_crime_regression']))
        df_C = df_C.withColumn("month", func.month(func.col("date"))). \
            withColumn("year", func.year(func.col("date"))). \
            groupBy('community_area_number', 'month', 'year', 'primary_type'). \
            agg(func.count(df_C.id).alias('nb_crimes')). \
            join(df_S, ['community_area_number'], "inner")
        df_C = self.get_dummies(df_C, list_columns=['primary_type', 'community_area_name', 'month'])
        return df_C

    def df_temperature(self):

        """
        Loads dataframe temperature and extract year, day, hour, month

        :return: spark dataframe
        """

        df = spark.read.format("csv").option("header", "true"). \
            option("mode", "DROPMALFORMED").option("delimiter", ",").load(self.path_temperature())
        df = df.select('Chicago', 'datetime').withColumnRenamed('Chicago', 'Temperature')
        df = df.filter((func.col('datetime') > self._start_year) & (func.col('datetime') < self._end_year))
        df = df.withColumn("month", func.month(func.col("datetime"))). \
            withColumn("year", func.year(func.col("datetime"))). \
            withColumn("day", func.dayofmonth(func.col("datetime"))). \
            withColumn("hour", func.hour(func.col("datetime")))
        df = df.withColumn('Temperature', df['Temperature'].cast(FloatType()))
        df = df.drop('datetime')
        return df

    def df_sky(self):
        """
        loads  dataframe weather description from csv file

        :return:  spark dataframe
        """

        df = spark.read.format("csv").option("header", "true"). \
            option("mode", "DROPMALFORMED").option("delimiter", ",").load(self.path_sky())
        df = df.na.drop()
        df = df.select('Chicago', 'datetime')
        sky_unique_values = list(df.select('Chicago').distinct().toPandas()['Chicago'])
        df = df.filter((func.col('datetime') > self._start_year) & (func.col('datetime') < self._end_year))
        for k in sky_unique_values:
            if k not in list(df.select('Chicago').distinct().toPandas()['Chicago']):
                df = df.withColumn("Chicago_" + k, func.lit(-999))
        df = self.get_dummies(df, list_columns=['Chicago'])
        df = df.withColumn("month", func.month(func.col("datetime"))). \
            withColumn("year", func.year(func.col("datetime"))). \
            withColumn("day", func.dayofmonth(func.col("datetime"))). \
            withColumn("hour", func.hour(func.col("datetime")))

        df = df.drop('datetime')
        return df
