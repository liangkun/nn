/**
 * Copyright (c) 2014 XiaoMi Inc. All Rights Reserved.
 * Authors: Liang Kun <liangkun@xiaomi.com>
 */

package org.sann.utils

import breeze.linalg.{all, DenseMatrix}
import org.testng.Assert._
import org.testng.annotations.Test

class UtilsTest {
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
