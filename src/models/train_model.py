from pyspark.ml.feature import VectorAssembler, StandardScaler, OneHotEncoderEstimator, StringIndexer, IndexToString
from pyspark.ml.regression import GBTRegressor
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier


class model_classification:
    """
    this class train random forest for  classification
    """

    def __init__(self, config, df_ml):
        """

        :param config: dict for configuration
        :param df_crime_socio:
        """
        self._config = config
        self._df_ml = df_ml
        self._df_train, self._df_test = self._df_ml.randomSplit([0.9, 0.1])

    def df_train(self):
        return self._df_train

    def df_test(self):
        return self._df_test

    def train_RF(self):
        """
        train randofm forest model for classification
        :return: the best model after cross validation and
        """
        from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
        from pyspark.ml.evaluation import MulticlassClassificationEvaluator
        rf = RandomForestClassifier(labelCol='label', featuresCol='features')
        categoricalColumns = ['domestic']
        numericCols = [item[0] for item in self._df_ml.dtypes if
                       (item[1].startswith('int') or
                        item[1].startswith('float') or
                        item[1].startswith('bigint') or
                        item[1].startswith('double'))]
        stages = []
        stringIndexer_label = StringIndexer(inputCol='primary_type', outputCol='label').fit(self.df_train())
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
        paramGrid = (
            ParamGridBuilder().addGrid(rf.numTrees, self._config['model_ML_classification']['param']['numTrees'])
            .addGrid(rf.maxDepth, self._config['model_ML_classification']['param']['maxDepth'])
            .build())

        crossval = CrossValidator(estimator=pipeline_and_model,
                                  estimatorParamMaps=paramGrid,
                                  evaluator=MulticlassClassificationEvaluator(),
                                  numFolds=10)
        cvModel = crossval.fit(self.df_train())
        bestPipeline = cvModel.bestModel
        return bestPipeline


class model_regression:
    """
    this class train gbt model regression
    """

    def __init__(self, config, df_ml):
        """

        :param config:
        :param df_ml:
        """
        self._config = config
        self._df_ml = df_ml

    def train_gbt(self):
        """

        :return: model of prediction number of crime by type and region (regression)
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
