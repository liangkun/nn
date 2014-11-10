/**
 * Copyright (c) 2014 XiaoMi Inc. All Rights Reserved.
 * Authors: Liang Kun <liangkun@xiaomi.com>
 */
package org.sann

import scala.util.Random

import breeze.linalg.DenseMatrix

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
}
