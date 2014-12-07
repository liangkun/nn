/**
 * Copyright (c) 2014 XiaoMi Inc. All Rights Reserved.
 * Authors: Liang Kun <liangkun@xiaomi.com>
 */

package org.sann.utils

import breeze.linalg.DenseMatrix
import org.sann.UnitSpec

class UtilsSpec extends UnitSpec {
  "shuffle()" should "permute the rows' order but do not add or remove rows" in {
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

      tmpData.rows should be (data.rows)
      tmpData.cols should be (data.cols)

      for (j <- 0 until data.rows) {
        var equals = 0
        for (k <- 0 until tmpData.rows) {
          if (breeze.linalg.all(tmpData(k, ::).t :== data(j, ::).t)) equals += 1
        }
        equals should be (1)
      }
    }
  }
}
