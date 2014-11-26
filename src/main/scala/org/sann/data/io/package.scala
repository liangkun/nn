/**
 * Copyright (c) 2014 Liang Kun. All Rights Reserved.
 * Authors: Liang Kun <liangkun@data-intelli.com>
 */

package org.sann.data

import scala.collection.mutable.ArrayBuffer
import scala.io.Source

import breeze.linalg.DenseMatrix

/**
 * Functions to load or store data.
 */
package object io {
  /**
   * Load data in text format.
   *
   * Each line contains a row of the result matrix, values are separated by sep.
   * @param source containing data in text format.
   * @param sep separator regexp.
   * @return matrix of the text source.
   */
  def loadDelimited(source: Source, sep: String = "\\s+"): DenseMatrix[Float] = {
    val builder = ArrayBuffer.empty[Float]
    var dimension = -1

    for (line <- source.getLines(); if line.nonEmpty) {
      val point = line.trim().split(sep).map(_.toFloat)
      if (dimension < 0) {
        dimension = point.size
      } else {
        require(point.size == dimension, s"data points have different dimensions: $dimension, ${point.size}")
      }
      builder ++= point
    }

    val data = builder.toArray
    new DenseMatrix[Float](data.size / dimension, dimension, data, 0, dimension, isTranspose = true)
  }
}
