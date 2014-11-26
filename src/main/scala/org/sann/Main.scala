/**
 * Copyright (c) 2014 Liang Kun. All Rights Reserved.
 * Authors: Liang Kun <liangkun@data-intelli.com>
 */

package org.sann


import scala.io.Source

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.plot.Figure

import org.sann.algos.Perceptron
import org.sann.algos.Perceptron.ListErrorCollector
import org.sann.data.gen.DoubleMoon
import org.sann.data.io
import org.sann.utils.plotBinaryDataPoints
import org.sann.utils.shuffleRows

object Main {
  def main(args: Array[String]): Unit = {
    nnChp1()
  }

  def plotDoubleMoon(size: Int = 1000, d: Float = 0.0f): Unit = {
    val fig = new Figure("DoubleMoon")
    val plt = fig.subplot(0)
    plotBinaryDataPoints(DoubleMoon.gen(size, d)).foreach(s => plt += s)
  }

  def nnChp1(): Unit = {
    val d = -4
    val trainSize = 5000
    val train = DoubleMoon.gen(trainSize, d)
    val trainXs = DenseMatrix.horzcat(DenseMatrix.ones[Float](trainSize, 1), train(::, 0 to 1))
    val trainYs = train(::, 2)
    val weights = DenseVector.zeros[Float](3)
    val errorCollector = new ListErrorCollector
    val (iters, trainErrors) = Perceptron.train(
      weights,
      trainXs,
      trainYs,
      maxIters = 1000,
      learningRate = 1f,
      pocket = true,
      errorCollector = Some(errorCollector)
    )
    val fig = new Figure("Playground")
    val plt = fig.subplot(1, 2, 0)
    utils.plotBinaryDataPoints(train).foreach(s => plt += s)
    plt += utils.plotLine(weights, -14.0f, 24.0f)
    val plt1 = fig.subplot(1, 2, 1)
    plt1 += utils.plotLearningCurve(errorCollector.get)
    println(s"Train Error Rate($trainErrors/$trainSize) = ${trainErrors.toFloat / trainSize}")

    // test
    val testSize = 2000
    val test = DoubleMoon.gen(testSize, d)
    val testXs = DenseMatrix.horzcat(DenseMatrix.ones[Float](testSize, 1), test(::, 0 to 1))
    val testYs = test(::, 2)
    val testErrors = Perceptron.countErrors(weights, testXs, testYs)
    println(s"Test Error Rate($testErrors/$testSize) = ${testErrors.toFloat / testSize}")
  }

  def mlfHw(): Unit = {
    val data = io.loadDelimited(Source.fromFile("D:\\workspace\\nn\\hw1_18_train.txt"))
    val test = io.loadDelimited(Source.fromFile("D:\\workspace\\nn\\hw1_18_test.txt"))
    val testXs = DenseMatrix.horzcat(DenseMatrix.ones[Float](test.rows, 1), test(::, 0 until (test.cols -1)))
    val testYs = test(::, test.cols - 1)

    var totalErrors = 0f
    for (i <- 1 to 2000) {
      val tmpData = shuffleRows(data)
      val n = tmpData.rows
      val xs = DenseMatrix.horzcat(DenseMatrix.ones[Float](n, 1), tmpData(::, 0 until (tmpData.cols - 1)))
      val ys = tmpData(::, tmpData.cols - 1)
      val weights = DenseVector.zeros[Float](xs.cols)
      val (iters, errors) = Perceptron.train(weights, xs, ys, maxIters = 100, pocket = true)
      assert(iters == 101 || errors == 0)
      totalErrors += Perceptron.countErrors(weights, testXs, testYs).toFloat / testXs.rows
    }

    println(totalErrors / 2000.0)
  }
}
