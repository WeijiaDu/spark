package com.sparkProject

import org.apache.spark.SparkConf
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.sql.SparkSession


object Trainer {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12",
      "spark.driver.maxResultSize" -> "2g"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP_spark")
      .getOrCreate()


    /*******************************************************************************
      *
      *       TP 3
      *
      *       - lire le fichier sauvegarder précédemment
      *       - construire les Stages du pipeline, puis les assembler
      *       - trouver les meilleurs hyperparamètres pour l'entraînement du pipeline avec une grid-search
      *       - Sauvegarder le pipeline entraîné
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/

    import org.apache.spark.ml.feature.{IDF, RegexTokenizer}
    import org.apache.spark.ml.feature.StopWordsRemover
    import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel}
    import org.apache.spark.ml.feature.{StringIndexer}
    import org.apache.spark.ml.feature.OneHotEncoder
    import org.apache.spark.ml.feature.VectorAssembler
    import org.apache.spark.ml.classification.LogisticRegression
    import org.apache.spark.ml.Pipeline
    import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
    import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}


    println("hello world ! from Trainer")

    // Import parquets successivement

    val df_clean = spark.read.parquet("/home/weijia/spark/TP2/prepared_trainingset")

    //ajouter tokens pour les données text, en mettan "tokens" comme la colonne de sortie
    val tokenizer = new RegexTokenizer()
      .setPattern("\\W+")
      .setGaps(true)
      .setInputCol("text")
      .setOutputCol("tokens")

    //supprimer les mots ponctuels comme "the, are, we" etc.
    val remover = new StopWordsRemover()
      .setInputCol("tokens")
      .setOutputCol("filtered")

    // transformer les tokens en vecteur, term frequency
    val cvModel = new CountVectorizer()
      .setInputCol("filtered")
      .setOutputCol("rowfeatures")
      .setVocabSize(3)
      .setMinDF(2)

    //tf-idf calculer leurs apparances au niveau global dans le text, inverse document frequency
    val idf = new IDF()
                  .setInputCol("rowfeatures")
                  .setOutputCol("tfidf")

    // transformer les countries en index, dans le cas de non validation on saute la ligne
    val indexer_country  = new StringIndexer()
      .setInputCol("country2")
      .setOutputCol("country2_indexed")
      .setHandleInvalid("skip")

    // transformer les currency en index
    val indexer_currency  = new StringIndexer()
      .setInputCol("currency2")
      .setOutputCol("currency2_indexed")
      .setHandleInvalid("skip")

    // transformer les index de country en un vecteur de format binaire ex : 3 en 00010
    val encoder_country = new OneHotEncoder()
      .setInputCol("country2_indexed")
      .setOutputCol("country_onehot")

    // tansformer les index de currency en format binaire
    val encoder_currency = new OneHotEncoder()
      .setInputCol("currency2_indexed")
      .setOutputCol("currency_onehot")

    //Assember les variables en un vecteur nommé features
    val assembler = new VectorAssembler()
      .setInputCols(Array("tfidf", "days_campaign", "hours_prepa", "goal", "country_onehot", "currency_onehot"))
      .setOutputCol("features")

    // faire un modèle de regression logistique classique
    val lr = new LogisticRegression()
      .setElasticNetParam(0.0) //sans pénalisation pour elastique net
      .setFitIntercept(true)  //retenu l'intercept
      .setFeaturesCol("features")  //colonne de features
      .setLabelCol("final_status")  // colonne de label
      .setStandardization(true)  //standardiser les donées
      .setRawPredictionCol("raw_predictions")  //garder les probas de prédiction pour chaque lignes
      .setThresholds(Array(0.7, 0.3)) // proba >0.7 pour classer en 1 sinon 0
      .setTol(1.0e-6)  // niveau de tolerance
      .setMaxIter(300)  //maximum itération

    // pepeline pour assembler toutes les fonctions
    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, remover, cvModel, idf, indexer_country, indexer_currency, encoder_country, encoder_currency, assembler, lr))

    // separer le data en 90% de train set et 10% de test set
    val Array(training, test) = df_clean.randomSplit(Array(0.9, 0.1))

    //regParam : régularisation de la régression logistique
    //Specifique minimum nombre de different documents un terme doit apparatre dedans
    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(10e-8, 10e-6, 10e-4, 10e-2))
      .addGrid(cvModel.minDF, Array(55.0,75.0,95.0))
      .build()

    //70% des données pour train et 30% pour la validation
    //fi-score pour comparer les modeles
    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(new MulticlassClassificationEvaluator().setMetricName("f1").setLabelCol("final_status"))
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.7)

    //fit le model
    val model = trainValidationSplit.fit(training)

    //transforme le model
    val df_WithPredictions = model.transform(test)

    //nombre pour chaque final_status
    df_WithPredictions.groupBy("final_status", "prediction").count.show()

    //afficher les F1-score du modèle
    val evaluator = new MulticlassClassificationEvaluator()
      .setMetricName("f1")
      .setLabelCol("final_status")
      .setPredictionCol("prediction")
    val f1 = evaluator.evaluate(df_WithPredictions)
    println(s"F1 score = $f1")

    //sauvgarder le modèle
    model.write.overwrite.save("/home/weijia/spark/TP2/model/")

  }
}
