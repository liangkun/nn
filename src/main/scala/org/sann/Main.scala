/**
 * Created by liangkun on 2014/10/18.
 */
package org.sann

import breeze.linalg.{DenseMatrix, DenseVector}

import scala.io.Source
import scala.util.Random


object Main {
  def main(args: Array[String]): Unit = {
    val data = loadData(Source.fromFile("D:\\workspace\\nn\\hw1_18_train.txt"))
    val test = loadData(Source.fromFile("D:\\workspace\\nn\\hw1_18_test.txt"))
    val testXs = DenseMatrix.horzcat(DenseMatrix.ones[Float](test.rows, 1), test(::, 0 until (test.cols -1)))
    val testYs = test(::, test.cols - 1)

    var totalErrors = 0f
    for (i <- 1 to 2000) {
      val tmpData = data.copy
      shuffle(tmpData, Random.nextLong())
      val n = tmpData.rows
      val xs = DenseMatrix.horzcat(DenseMatrix.ones[Float](n, 1), tmpData(::, 0 until (tmpData.cols - 1)))
      val ys = tmpData(::, tmpData.cols - 1)
      val weights = DenseVector.zeros[Float](xs.cols)
      val (iters, errors) = perceptron(weights, xs, ys, maxIters = 100, pocket = true)
      assert(iters == 101 || errors == 0)
      totalErrors += countErrors(weights, testXs, testYs).toFloat / testXs.rows
    }

    println(totalErrors / 2000.0)
  }
}
