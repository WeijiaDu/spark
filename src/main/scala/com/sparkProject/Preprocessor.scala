package com.sparkProject

import org.apache.spark.SparkConf
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.dmg.pmml.False


object Preprocessor {

  def main(args: Array[String]): Unit = {

    // Des réglages optionels du job spark. Les réglages par défaut fonctionnent très bien pour ce TP
    // on vous donne un exemple de setting quand même
    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12"
    ))

    // Initialisation de la SparkSession qui est le point d'entrée vers Spark SQL (donne accès aux dataframes, aux RDD,
    // création de tables temporaires, etc et donc aux mécanismes de distribution des calculs.)
    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP_spark")
      .getOrCreate()


    /*******************************************************************************
      *
      *       TP 2
      *
      *       - Charger un fichier csv dans un dataFrame
      *       - Pre-processing: cleaning, filters, feature engineering => filter, select, drop, na.fill, join, udf, distinct, count, describe, collect
      *       - Sauver le dataframe au format parquet
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/
    import spark.implicits._

    println("hello world ! from Preprocessor")

    val data = spark.read.format(source="csv").option("header","true").load(path="/home/weijia/spark/train_clean.csv")
    //data.show(numRows = 5)
    //println(data.count())
    //println(data.columns.size)

    //data.printSchema()

    val data_cast = data
      .withColumn("goal", $"goal".cast("Int"))
      .withColumn("deadline" , $"deadline".cast("Int"))
      .withColumn("state_changed_at", $"state_changed_at".cast("Int"))
      .withColumn("created_at", $"created_at".cast("Int"))
      .withColumn("launched_at", $"launched_at".cast("Int"))
      .withColumn("backers_count", $"backers_count".cast("Int"))
      .withColumn("final_status", $"final_status".cast("Int"))

    //data_cast.printSchema()

    //data_cast.select("goal", "backers_count", "final_status").describe().show

    val df2: DataFrame = data_cast.drop("disable_communication")
    val dfNoFutur: DataFrame = df2.drop("backers_count", "state_changed_at")
    //dfNoFutur.filter($"country" === "False").groupBy("currency").count.orderBy($"count".desc).show(50)
    //dfNoFutur.show(numRows = 5)

    import org.apache.spark.sql.functions.udf

    def udfCountry = udf{(country: String, currency: String) =>
      if (country == "False")
        currency
      else if (country.length != 2)
        null
      else
        country //: ((String, String) => String)  pour éventuellement spécifier le type
    }

    def udfCurrency = udf{(currency: String) =>
      if ( currency != null && currency.length != 3 )
        null
      else
        currency //: ((String, String) => String)  pour éventuellement spécifier le type
    }

    val dfCountry: DataFrame = dfNoFutur
      .withColumn("country2", udfCountry($"country", $"currency"))
      .withColumn("currency2", udfCurrency($"currency"))
      .drop("country", "currency")

    //dfCountry.groupBy("final_status").count.orderBy($"count".desc).show(5)

    def udfStatus = udf{(stat: Int) =>
      if ( stat != 0 && stat != 1 )
        0
      else if (stat == null)
        0
      else
        stat //: ((String, String) => String)  pour éventuellement spécifier le type
    }

    val dfStatus: DataFrame = dfCountry
      .withColumn("final_status", udfStatus($"final_status"))


    //dfCountry.groupBy("final_status").count.orderBy($"count".desc).show(100)
    val dfStatus_drop: DataFrame = dfCountry.filter($"final_status" isin (0, 1))
    //dfStatus_drop.groupBy("final_status").count.orderBy($"count".desc).show(5)

    val df_var1: DataFrame = dfStatus_drop.withColumn("days_campaign", (($"deadline" - $"launched_at")/60/60/24))
    //df_var1.show(10)

    import org.apache.spark.sql.functions._

    val df_var2: DataFrame = df_var1.withColumn("hours_prepa", round(($"launched_at" - $"created_at")/60/60/24,3))
    //df_var2.show(10)

    val df_var: DataFrame = df_var2.drop("launched_at", "created_at", "deadline")
    //df_var.show(10)

    val df_concat: DataFrame = df_var
      .withColumn("text",concat($"name",lit(' '), $"desc",lit(' '),$"keywords"))
      .drop("name","desc","keywords")
    df_concat.show(10)



  }

}
