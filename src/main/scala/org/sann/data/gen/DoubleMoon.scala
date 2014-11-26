/**
 * Copyright (c) 2014 Liang Kun. All Rights Reserved.
 * Authors: Liang Kun <liangkun@data-intelli.com>
 */

package org.sann.data.gen

import breeze.linalg.{DenseVector, DenseMatrix}
import breeze.numerics.{constants, sin, cos}
import breeze.stats.distributions.Uniform
import org.sann.utils.shuffleRows

/**
 * DoubleMoon data point generation.
 */
object DoubleMoon {
  /**
   * Generate double moon data with uniformed sample.
   * @param size total size of the result data.
   * @param d distance of the moons.
   * @return sample matrix, each row contains a data point with xs before y.
   */
  def gen(size: Int, d: Float, shuffle: Boolean = true): DenseMatrix[Float] = {
    val w = 6.0f
    val r = 10.0f - w / 2  // inner circle's radius.
    val upperSamples = sampleFromMoon(size / 2, DenseVector(.0f, .0f), r, w, upper = true)
    val bottomSamples = sampleFromMoon(size / 2, DenseVector(10.0f, -d), r, w, upper = false)
    val result = DenseMatrix.vertcat(upperSamples, bottomSamples)
    if (shuffle) shuffleRows(result) else result
  }

  /**
   * Sample from one half moon.
   *
   * @param size sample size.
   * @param center center of the moon.
   * @param r radius of the inner circle.
   * @param w width of the moon.
   * @param upper is this the upper moon or the bottom moon.
   * @return sample matrix, each row contains a data point with xs before y.
   */
  def sampleFromMoon(size: Int, center: DenseVector[Float], r: Float, w: Float, upper: Boolean): DenseMatrix[Float] = {
    val rSampler = new Uniform(r, r + w)
    val thetaSampler = if (upper) new Uniform(0.0, constants.Pi) else new Uniform(constants.Pi, 2 * constants.Pi)
    val sampleR = DenseVector.zeros[Float](size)
    val sampleTheta = DenseVector.zeros[Float](size)

    for (i <- 0 until size) {
      sampleR(i) = rSampler.draw().toFloat
      sampleTheta(i) = thetaSampler.draw().toFloat
    }

    val result = DenseMatrix.zeros[Float](size, 3)
    result(::, 0) := (sampleR :* cos(sampleTheta)) + center(0)
    result(::, 1) := (sampleR :* sin(sampleTheta)) + center(1)
    result(::, 2) := (if (upper) 1.0f else -1.0f)

    result
  }
}
