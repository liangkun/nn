/**
 * Copyright (c) 2014 Liang Kun. All Rights Reserved.
 * Authors: Liang Kun <liangkun@data-intelli.com>
 */

package org.sann.algos

import breeze.linalg.{all, DenseVector, DenseMatrix}
import org.testng.Assert._
import org.testng.annotations.{DataProvider, Test}

class PerceptronTest {
  @Test
  def testSign(): Unit = {
    assertEquals(Perceptron.sign(0f), -1f)
    assertEquals(Perceptron.sign(.1f), 1f)
    assertEquals(Perceptron.sign(-.1f), -1f)
  }

  @Test
  def testErrors(): Unit = {
    val testWeights = DenseVector(1f, 1f, 1f)  // for easy reason
    val testYs = DenseVector(1f, 1f, 1f/*e*/, -1f, -1f/*e*/, 1f, -1f/*e*/, 1f)
    val testXs = DenseMatrix(
      (1f, 1f, 1f),
      (1f, 1f, 1f),
      (1f, -1f, 0f),
      (1f, -1f, -.5f),
      (.5f, -1f, 1f),
      (-.5f, 2f, -.5f),
      (1f, 1f, -1.5f),
      (-2f, 1.5f, 1f)
    )

    assertEquals(Perceptron.countErrors(testWeights, testXs, testYs), 3)
    assertEquals(Perceptron.countErrors(testWeights, testXs(0 to 1, ::), testYs(0 to 1)), 0)
    assertEquals(Perceptron.countErrors(testWeights, testXs(0 to 5, ::), testYs(0 to 5)), 2)
    assertEquals(Perceptron.nextError(-1, testWeights, testXs, testYs), 2)
    assertEquals(Perceptron.nextError(2, testWeights, testXs, testYs), 4)
    assertEquals(Perceptron.nextError(5, testWeights, testXs, testYs), 6)
    assertEquals(Perceptron.nextError(6, testWeights, testXs, testYs), 2)
    assertEquals(Perceptron.nextError(-1, testWeights, testXs(0 to 1, ::), testYs(0 to 1)), -1)
  }

  @Test(dataProvider = "PointProvider")
  def testPerceptron(xs: DenseMatrix[Float], ys: DenseVector[Float], minErrors: Int): Unit = {
    val weights = DenseVector.zeros[Float](xs.cols + 1)
    val weightsPocket = weights.copy
    val realXs = DenseMatrix.horzcat(DenseMatrix.ones[Float](xs.rows, 1), xs)

    val (iters, errors) = Perceptron.train(weights, realXs, ys, maxIters = 10000)
    val (itersPocket, errorsPocket) = Perceptron.train(weightsPocket, realXs, ys, maxIters = 10000, pocket = true)
    if (minErrors == 0) {
      assertTrue(iters < 10000)
      assertEquals(errors, 0)
      assertTrue(itersPocket < 10000)
      assertEquals(errorsPocket, 0)
    } else {
      assertEquals(iters, 10001)
      assertTrue(errors > minErrors)
      assertEquals(itersPocket, 10001)
      assertTrue(errorsPocket > minErrors)
    }
    assertEquals(Perceptron.countErrors(weights, realXs, ys), errors)
    assertEquals(Perceptron.countErrors(weightsPocket, realXs, ys), errorsPocket)
  }

  @Test(dataProvider = "PointProvider")
  def testPerceptronLearningRate(xs: DenseMatrix[Float], ys: DenseVector[Float], minErrors: Int): Unit = {
    val zeros = DenseVector.zeros[Float](xs.cols + 1)
    val weights = zeros.copy
    val realXs = DenseMatrix.horzcat(DenseMatrix.ones[Float](xs.rows, 1), xs)
    val (iters, errors) = Perceptron.train(weights, realXs, ys, maxIters = 10, learningRate = 0f)
    assertTrue(all(weights :== zeros))
  }

  @DataProvider(name = "PointProvider")
  def pointProvider() = Array(
    Array(DenseMatrix((0f, 0f), (1f, 0f), (0f, 1f)), DenseVector(-1f, -1f, -1f), 0),
    Array(DenseMatrix((0f, 0f), (1f, 0f), (0f, 1f)), DenseVector(-1f, -1f, 1f), 0),
    Array(DenseMatrix((0f, 0f), (1f, 0f), (0f, 1f)), DenseVector(-1f, 1f, -1f), 0),
    Array(DenseMatrix((0f, 0f), (1f, 0f), (0f, 1f)), DenseVector(1f, -1f, -1f), 0),
    Array(DenseMatrix((0f, 0f), (1f, 0f), (0f, 1f)), DenseVector(-1f, 1f, 1f), 0),
    Array(DenseMatrix((0f, 0f), (1f, 0f), (0f, 1f)), DenseVector(1f, -1f, 1f), 0),
    Array(DenseMatrix((0f, 0f), (1f, 0f), (0f, 1f)), DenseVector(1f, 1f, -1f), 0),
    Array(DenseMatrix((0f, 0f), (1f, 0f), (0f, 1f)), DenseVector(1f, 1f, 1f), 0),
    Array(DenseMatrix((0f, 0f), (1f, 0f), (0f, 1f), (1f, 1f)), DenseVector(1f, 1f, 1f, -1f), 0),
    Array(DenseMatrix((0f, 0f), (1f, 0f), (0f, 1f), (1f, 1f)), DenseVector(1f, 1f, -1f, -1f), 0),
    Array(DenseMatrix((0f, 0f), (1f, 0f), (0f, 1f), (1f, 1f)), DenseVector(1f, -1f, -1f, -1f), 0),
    Array(DenseMatrix((0f, 0f), (1f, 0f), (0f, 1f), (1f, 1f)), DenseVector(1f, -1f, -1f, 1f), 1),
    Array(DenseMatrix((0f, 0f), (1f, 0f), (0f, 1f), (1f, 1f)), DenseVector(-1f, 1f, 1f, -1f), 1)
  )
}
