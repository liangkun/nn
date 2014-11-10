/**
 * Created by liangkun on 2014/10/18.
 */
package org.sann

import breeze.linalg._

import scala.io.Source

import org.testng.Assert._
import org.testng.annotations.{Test, DataProvider}

import org.sann.utils.shuffleRows

class SANNTest {
  @Test(dataProvider = "SourceDataProvider")
  def testLoadData(src: Source, expected: DenseMatrix[Float]): Unit = {
    val data = loadData(src)
    assertTrue(all(data :== expected))
  }

  @DataProvider(name = "SourceDataProvider")
  def sourceDataProvider() = Array(
    Array("0.1 0.2\t0.3", DenseMatrix((.1f, .2f, .3f))),
    Array("1 2 3\n4  5 \t6 \n", DenseMatrix((1f, 2f, 3f), (4f, 5f, 6f))),
    Array("1 2 3\n 4 5 6\n\n7 8\t9", DenseMatrix((1f, 2f, 3f), (4f, 5f, 6f), (7f, 8f, 9f))),
    Array("\t\t1 2 \n 4 5 \n\n7\t9", DenseMatrix((1f, 2f), (4f, 5f), (7f, 9f)))
  ).map { case Array(str: String, mat) => Array(Source.fromString(str), mat)}

  @Test
  def testSign(): Unit = {
    assertEquals(sign(0f), -1f)
    assertEquals(sign(.1f), 1f)
    assertEquals(sign(-.1f), -1f)
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

    assertEquals(countErrors(testWeights, testXs, testYs), 3)
    assertEquals(countErrors(testWeights, testXs(0 to 1, ::), testYs(0 to 1)), 0)
    assertEquals(countErrors(testWeights, testXs(0 to 5, ::), testYs(0 to 5)), 2)
    assertEquals(nextError(-1, testWeights, testXs, testYs), 2)
    assertEquals(nextError(2, testWeights, testXs, testYs), 4)
    assertEquals(nextError(5, testWeights, testXs, testYs), 6)
    assertEquals(nextError(6, testWeights, testXs, testYs), 2)
    assertEquals(nextError(-1, testWeights, testXs(0 to 1, ::), testYs(0 to 1)), -1)
  }

  @Test(dataProvider = "PointProvider")
  def testPerceptron(xs: DenseMatrix[Float], ys: DenseVector[Float], minErrors: Int): Unit = {
    val weights = DenseVector.zeros[Float](xs.cols + 1)
    val weightsPocket = weights.copy
    val realXs = DenseMatrix.horzcat(DenseMatrix.ones[Float](xs.rows, 1), xs)

    val (iters, errors) = perceptron(weights, realXs, ys, maxIters = 10000)
    val (itersPocket, errorsPocket) = perceptron(weightsPocket, realXs, ys, maxIters = 10000, pocket = true)
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
    assertEquals(countErrors(weights, realXs, ys), errors)
    assertEquals(countErrors(weightsPocket, realXs, ys), errorsPocket)
  }

  @Test(dataProvider = "PointProvider")
  def testPerceptronLearningRate(xs: DenseMatrix[Float], ys: DenseVector[Float], minErrors: Int): Unit = {
    val zeros = DenseVector.zeros[Float](xs.cols + 1)
    val weights = zeros.copy
    val realXs = DenseMatrix.horzcat(DenseMatrix.ones[Float](xs.rows, 1), xs)
    val (iters, errors) = perceptron(weights, realXs, ys, maxIters = 10, learningRate = 0f)
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

  @Test
  def testShuffle(): Unit = {
    val data = DenseMatrix(
      (1f, 1f, 1f),
      (1f, -1f, 0f),
      (1f, -1f, -.5f),
      (.5f, -1f, 1f),
      (-.5f, 2f, -.5f),
      (1f, 1f, -1.5f),
      (-2f, 1.5f, 1f)
    )

    for (i <- 0 to 9) {
      val tmpData = shuffleRows(data)

      assertEquals(tmpData.rows, data.rows)
      assertEquals(tmpData.cols, data.cols)
      for (j <- 0 until data.rows) {
        var equals = 0
        for (k <- 0 until tmpData.rows) {
          if (all(tmpData(k, ::).t :== data(j, ::).t)) equals += 1
        }

        assertEquals(equals, 1)
      }
    }
  }
}
