from pyspark.ml.feature import StandardScaler, VectorAssembler, VectorIndexer, StandardScaler, OneHotEncoderEstimator, StringIndexer, IndexToString
from pyspark.ml.regression import GBTRegressor
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier, RandomForestClassificationModel, GBTClassifier,OneVsRest, OneVsRestModel

class model_classification:

    def __init__(self, config, df_ml):
        """

        :param config:
        :param df_crime_socio:
        """
        self._config = config
        self._df_ml = df_ml



    def train_model(self):
        from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
        from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

        """

        :param :
        :return: list of dataframes for machine learning
        """
        rf = RandomForestClassifier(labelCol='label', featuresCol='features')
        categoricalColumns = ['domestic']
        numericCols = ['year', 'month', 'day', 'hour', 'minute', 'latitude',
                       'longitude', 'isStreet', 'isAV',
                       'isBLVD', 'isRD', 'isPL', 'isBROADWAY',
                       'isPKWY', 'duree_day', 'minute', 'dayofmonth',
                       'dayofyear', 'dayofweek', 'Temperature',
                       'pct_housing_crowded',
                       'pct_households_below_poverty',
                       'pct_age16_unemployed',
                       'pct_age25_no_highschool',
                       'pct_not_working_age',
                       'per_capita_income',
                       'hardship_index',
                       'Chicago_broken clouds',
                       'Chicago_drizzle',
                       'Chicago_few clouds',
                       'Chicago_fog',
                       'Chicago_haze',
                       'Chicago_heavy intensity drizzle',
                       'Chicago_heavy intensity rain',
                       'Chicago_heavy snow',
                       'Chicago_light intensity drizzle',
                       'Chicago_light rain',
                       'Chicago_light rain and snow',
                       'Chicago_light snow',
                       'Chicago_mist',
                       'Chicago_moderate rain',
                       'Chicago_overcast clouds',
                       'Chicago_proximity thunderstorm',
                       'Chicago_scattered clouds',
                       'Chicago_sky is clear',
                       'Chicago_snow',
                       'Chicago_thunderstorm',
                       'Chicago_thunderstorm with heavy rain',
                       'Chicago_thunderstorm with light rain',
                       'Chicago_thunderstorm with rain',
                       'Chicago_very heavy rain'
                       ]
        df_train, df_test = self._df_ml.randomSplit([0.7, 0.3])
        stages = []
        stringIndexer_label = StringIndexer(inputCol='primary_type', outputCol='label').fit(df_train)
        labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
                                       labels=stringIndexer_label.labels)
        for categoricalCol in categoricalColumns:
            stringIndexer = StringIndexer(inputCol=categoricalCol, outputCol=categoricalCol + 'Index')
            encoder = OneHotEncoderEstimator(inputCols=[stringIndexer.getOutputCol()],
                                             outputCols=[categoricalCol + "classVec"])
            stages.append(stringIndexer)
            stages.append(encoder)

        stages.append(stringIndexer_label)
        assemblerInputs = [c + "classVec" for c in categoricalColumns] + numericCols
        assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
        stages.append(assembler)
        stages.append(rf)
        stages.append(labelConverter)
        pipeline_and_model = Pipeline(stages=stages)
        paramGrid = (ParamGridBuilder().addGrid(rf.numTrees, [50, 60])
                     .addGrid(rf.maxDepth, [5, 8])
                     .build())

        crossval = CrossValidator(estimator=pipeline_and_model,
                                  estimatorParamMaps=paramGrid,
                                  evaluator=MulticlassClassificationEvaluator(),
                                  numFolds=10)
        cvModel = crossval.fit(df_train)
        bestPipeline = cvModel.bestModel
        return bestPipeline

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




