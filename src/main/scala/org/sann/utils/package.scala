/**
 * Copyright (c) 2014 Liang Kun. All Rights Reserved.
 * Authors: Liang Kun <liangkun@data-intelli.com>
 */

package org.sann

import scala.util.Random

import breeze.linalg.{DenseVector, DenseMatrix}
import breeze.plot._

/**
 * Utility functions not supported by breeze yet.
 */
package object utils {
  /**
   * Shuffle rows of a dense matrix.
   *
   * @param data matrix to shuffle.
   * @param seed used for random number generation.
   * @return shuffled matrix.
   */
  def shuffleRows[T](data: DenseMatrix[T], seed: Option[Long] = None): DenseMatrix[T] = {
    val sd = seed.getOrElse(System.currentTimeMillis())
    val result = data.copy
    val rand = new Random(sd)
    val newIndexes = rand.shuffle((0 until data.rows).toVector)
    for ((i, old) <- newIndexes.zipWithIndex) {
      result(i, ::) := data(old, ::)
    }

    result
  }

  /**
   * Plot a single line from weights.
   */
  def plotLine(weights: DenseVector[Float], startX1: Float, endX1: Float): Series = {
    require(weights.length == 3, "weights should has 3 dimensions")
    def getX2(x1: Float) = - weights(0)/weights(2) - x1 * weights(1)/weights(2)

    plot(DenseVector(startX1, endX1), DenseVector(getX2(startX1), getX2(endX1)))
  }

  /**
   * Plot learning curve.
   */
  def plotLearningCurve(errors: Seq[Float]): Series = {
    plot((0 until errors.size).map(_.toFloat), errors)
  }

  /**
   * Plot -1/+1 labeled data points.
   */
  def plotBinaryDataPoints(data: DenseMatrix[Float]): Seq[Series] = {
    require(data.cols == 3, "data should has 3 columns")
    var pos = List[(Float, Float)]()
    var neg = List[(Float, Float)]()

    for (i <- 0 until data.rows) {
      if (data(i, 2) > 0.0) {
        pos = (data(i, 0), data(i, 1)) :: pos
      } else {
        neg = (data(i, 0), data(i, 1)) :: neg
      }
    }

    List(
      plot(pos.map(_._1), pos.map(_._2), '+'),
      plot(neg.map(_._1), neg.map(_._2), '.')
    )
  }
}
