/**
 * Copyright (c) 2014 Liang Kun. All Rights Reserved.
 * Authors: Liang Kun <liangkun@data-intelli.com>
 */

package org.sann.algos

import breeze.linalg.{sum, DenseMatrix, DenseVector}

/**
 * This object encapsulate the perceptron algorithms.
 */
object Perceptron {
  /**
   * Predict with perceptron.
   *
   * @param weights perceptron weights.
   * @param x input vector.
   * @return predicted label.
   */
  def predict(weights: DenseVector[Float], x: DenseVector[Float]): Float = sign(weights.t * x)

  /**
   * stochastic training.
   *
   * @param weights initial weights.
   * @param xs input features of the training data.
   * @param ys labels of the training data, -1 for negative, +1 for positive
   * @param maxIters max iterations.
   * @param learningRate of the trainning.
   * @param pocket use pocket pla if true.
   * @param errorCollector collect error rate after each step if not None.
   * @return totalIterations and remain errors.
   */
  def train(weights: DenseVector[Float],
            xs: DenseMatrix[Float],
            ys: DenseVector[Float],
            maxIters: Int = Int.MaxValue,
            learningRate: Float = 1f,
            pocket: Boolean = false,
            errorCollector: Option[Float => Unit] = None): (Int, Int) = {
    var iters = 0

    var pocketWeights = if (pocket) weights.copy else DenseVector(0f)
    var pocketErrors = if (pocket) countErrors(weights, xs, ys) else 0

    var next = nextError(-1, weights, xs, ys)
    while(next >= 0 && iters <= maxIters) {
      assert(sign(xs(next, ::) * weights) != ys(next))
      val delta = xs(next, ::).t * (ys(next) * learningRate)
      weights := weights + delta

      if (pocket || errorCollector.nonEmpty) {
        val errors = countErrors(weights, xs, ys)
        if (pocket && errors < pocketErrors) {
          pocketErrors = errors
          pocketWeights := weights
        }

        if (errorCollector.nonEmpty) {
          val collector = errorCollector.get
          if (pocket) {
            collector(pocketErrors.toFloat / ys.length.toFloat)
          } else {
            collector(errors.toFloat / ys.length.toFloat)
          }
        }
      }

      next = nextError(next, weights, xs, ys)
      iters += 1
    }

    if (pocket) {
      weights := pocketWeights
    }

    val errors = if (pocket) {
      pocketErrors
    } else if (iters <= maxIters) {
      0
    } else {
      countErrors(weights, xs, ys)
    }

    (iters, errors)
  }

  /**
   * Counter errors for perceptron.
   *
   * @param weights perceptron weights.
   * @param xs input features.
   * @param ys correct labels
   * @return error count.
   */
  def countErrors(weights: DenseVector[Float], xs: DenseMatrix[Float], ys: DenseVector[Float]): Int = {
    val predicts = (xs * weights).mapValues(x => sign(x))
    sum((predicts :!= ys).mapValues(c => if (c) 1 else 0))
  }

  /**
   * Find next error of the perceptron. Wrap around to the first point if reached the last point of
   * the data if nessary.
   *
   * @param start start index(excluded) to begin. If you want to begin with 0, use -1.
   * @param weights current weights.
   * @param xs input data points.
   * @param ys correct data labels.
   * @return next error index, -1 means no error found.
   */
  def nextError(start: Int, weights: DenseVector[Float], xs: DenseMatrix[Float], ys: DenseVector[Float]): Int = {
    for (offset <- 1 to xs.rows) {
      val index = (start + offset) % xs.rows
      if (sign(xs(index, ::) * weights) != ys(index)) return index
    }

    -1  // no errors
  }

  /** sign function that make 0 to be -1. */
  def sign(x: Float): Float = if (x <= 0) -1f else 1f

  /** A simple error collector, put all the errors into a list in order */
  class ListErrorCollector extends (Float => Unit) {
    private[this] var result = List[Float]()

    /** accumulate one error */
    def apply(e: Float): Unit = result = e :: result

    /** get the result list */
    def get: List[Float] = result.reverse
  }
}
