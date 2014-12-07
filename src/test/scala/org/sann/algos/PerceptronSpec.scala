/**
 * Copyright (c) 2014 Liang Kun. All Rights Reserved.
 * Authors: Liang Kun <liangkun@data-intelli.com>
 */

package org.sann.algos

import breeze.linalg.{DenseVector, DenseMatrix}
import org.sann.UnitSpec

class PerceptronSpec extends UnitSpec {
  "sign(x)" should "return -1.0f when x <= 0" in {
    Perceptron.sign(0f) should be (-1f)
    Perceptron.sign(-.1f) should be (-1f)
  }

  it should "return 1.0f when x > 0" in {
    Perceptron.sign(.1f) should be (1f)
  }

  // data for counterErrors() and nextError()
  private val testWeights = DenseVector(1f, 1f, 1f)  // for easy reason
  private val testYs = DenseVector(1f, 1f, 1f/*e*/, -1f, -1f/*e*/, 1f, -1f/*e*/, 1f)
  private val testXs = DenseMatrix(
    (1f, 1f, 1f),
    (1f, 1f, 1f),
    (1f, -1f, 0f),
    (1f, -1f, -.5f),
    (.5f, -1f, 1f),
    (-.5f, 2f, -.5f),
    (1f, 1f, -1.5f),
    (-2f, 1.5f, 1f)
  )

  "countErrors()" should "count mis-classified samples" in {
    val samples = Table(
      ("xs", "ys", "count"),
      (testXs, testYs, 3),
      (testXs(0 to 1, ::), testYs(0 to 1), 0),
      (testXs(0 to 5, ::), testYs(0 to 5), 2)
    )

    forAll(samples) { (xs: DenseMatrix[Float], ys: DenseVector[Float], count: Int) =>
      Perceptron.countErrors(testWeights, xs, ys) should be (count)
    }
  }

  "nextError()" should "return next error sample index" in {
    val samples = Table(
      ("start", "xs", "ys", "index"),
      (-1, testXs, testYs, 2),
      (2, testXs, testYs, 4),
      (5, testXs, testYs, 6),
      (6, testXs, testYs, 2),
      (-1, testXs(0 to 1, ::), testYs(0 to 1), -1)
    )

    forAll(samples) { (start: Int, xs: DenseMatrix[Float], ys: DenseVector[Float], index: Int) =>
      Perceptron.nextError(start, testWeights, xs, ys) should be (index)
    }
  }

  // data for perceptron
  private val samples = Table(
    ("xs", "ys", "minErrors"),
    (DenseMatrix((0f, 0f), (1f, 0f), (0f, 1f)), DenseVector(-1f, -1f, -1f), 0),
    (DenseMatrix((0f, 0f), (1f, 0f), (0f, 1f)), DenseVector(-1f, -1f, 1f), 0),
    (DenseMatrix((0f, 0f), (1f, 0f), (0f, 1f)), DenseVector(-1f, 1f, -1f), 0),
    (DenseMatrix((0f, 0f), (1f, 0f), (0f, 1f)), DenseVector(1f, -1f, -1f), 0),
    (DenseMatrix((0f, 0f), (1f, 0f), (0f, 1f)), DenseVector(-1f, 1f, 1f), 0),
    (DenseMatrix((0f, 0f), (1f, 0f), (0f, 1f)), DenseVector(1f, -1f, 1f), 0),
    (DenseMatrix((0f, 0f), (1f, 0f), (0f, 1f)), DenseVector(1f, 1f, -1f), 0),
    (DenseMatrix((0f, 0f), (1f, 0f), (0f, 1f)), DenseVector(1f, 1f, 1f), 0),
    (DenseMatrix((0f, 0f), (1f, 0f), (0f, 1f), (1f, 1f)), DenseVector(1f, 1f, 1f, -1f), 0),
    (DenseMatrix((0f, 0f), (1f, 0f), (0f, 1f), (1f, 1f)), DenseVector(1f, 1f, -1f, -1f), 0),
    (DenseMatrix((0f, 0f), (1f, 0f), (0f, 1f), (1f, 1f)), DenseVector(1f, -1f, -1f, -1f), 0),
    (DenseMatrix((0f, 0f), (1f, 0f), (0f, 1f), (1f, 1f)), DenseVector(1f, -1f, -1f, 1f), 1),
    (DenseMatrix((0f, 0f), (1f, 0f), (0f, 1f), (1f, 1f)), DenseVector(-1f, 1f, 1f, -1f), 1)
  )

  "perceptron" should "shuffle 3 point but not 4 point" in {
    forAll(samples) { (xs: DenseMatrix[Float], ys: DenseVector[Float], minErrors) =>
      val weights = DenseVector.zeros[Float](xs.cols + 1)
      val weightsPocket = weights.copy
      val realXs = DenseMatrix.horzcat(DenseMatrix.ones[Float](xs.rows, 1), xs)

      val (iters, errors) = Perceptron.train(weights, realXs, ys, maxIters = 10000)
      val (itersPocket, errorsPocket) = Perceptron.train(weightsPocket, realXs, ys, maxIters = 10000, pocket = true)
      if (minErrors == 0) {
        iters should be < 10000
        errors should be (0)
        itersPocket should be < 10000
        errorsPocket should be (0)
      } else {
        iters should be (10001)
        errors should be > minErrors
        itersPocket should be (10001)
        errorsPocket should be > minErrors
      }
      Perceptron.countErrors(weights, realXs, ys) should be (errors)
      Perceptron.countErrors(weightsPocket, realXs, ys) should be (errorsPocket)
    }
  }

  "Learning rate" should "affect learning speed" in {
    forAll(samples) { (xs: DenseMatrix[Float], ys: DenseVector[Float], minErrors) =>
      val zeros = DenseVector.zeros[Float](xs.cols + 1)
      val weights = zeros.copy
      val realXs = DenseMatrix.horzcat(DenseMatrix.ones[Float](xs.rows, 1), xs)
      val (iters, errors) = Perceptron.train(weights, realXs, ys, maxIters = 10, learningRate = 0f)
      breeze.linalg.all(weights :== zeros) should be (true)
    }
  }
}