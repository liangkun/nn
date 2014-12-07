/**
 * Copyright (c) 2014 Liang Kun. All Rights Reserved.
 * Authors: Liang Kun <liangkun@data-intelli.com>
 */

package org.sann.data.io

import scala.io.Source

import breeze.linalg.DenseMatrix
import org.sann.UnitSpec

class IOSpec extends UnitSpec {
  val sourceLines = Table(
    ("source", "expected"),
    ("0.1 0.2\t0.3", DenseMatrix((.1f, .2f, .3f))),
    ("1 2 3\n4  5 \t6 \n", DenseMatrix((1f, 2f, 3f), (4f, 5f, 6f))),
    ("1 2 3\n 4 5 6\n\n7 8\t9", DenseMatrix((1f, 2f, 3f), (4f, 5f, 6f), (7f, 8f, 9f))),
    ("\t\t1 2 \n 4 5 \n\n7\t9", DenseMatrix((1f, 2f), (4f, 5f), (7f, 9f)))
  )

  "loadDeminited()" should "load data seperated by any space(s)" in {
    forAll(sourceLines) { (line: String, expected: DenseMatrix[Float]) =>
      val data = loadDelimited(Source.fromString(line))
      breeze.linalg.all(data :== expected) should be (true)
    }
  }
}
