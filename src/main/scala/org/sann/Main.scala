/**
 * Created by liangkun on 2014/10/18.
 */
package org.sann

import scala.io.Source

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.plot.{Figure, plot}

import org.sann.algos.Perceptron
import org.sann.data.gen.DoubleMoon
import org.sann.data.io
import org.sann.utils.shuffleRows

object Main {
  def main(args: Array[String]): Unit = {

  }

  def plotDoubleMoon():Unit = {
    val fig = new Figure("DoubleMoon")
    val plt = fig.subplot(0)
    val size = 1000
    val samples = DoubleMoon.gen(size, 0, shuffle = false)
    val positives = samples(0 until size / 2, 0 to 1)
    val negatives = samples(size / 2 until size, 0 to 1)
    plt += plot(positives(::, 0), positives(::, 1), '+')
    plt += plot(negatives(::, 0), negatives(::, 1), '.')

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
