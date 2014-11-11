/**
 * Copyright (c) 2014 XiaoMi Inc. All Rights Reserved.
 * Authors: Liang Kun <liangkun@xiaomi.com>
 */

package org.sann.data.io

import scala.io.Source

import breeze.linalg.{all, DenseMatrix}
import org.testng.Assert._
import org.testng.annotations.{DataProvider, Test}

class IOTest {
  @Test(dataProvider = "SourceDataProvider")
  def testLoadDelimited(src: Source, expected: DenseMatrix[Float]): Unit = {
    val data = loadDelimited(src)
    assertTrue(all(data :== expected))
  }

  @DataProvider(name = "SourceDataProvider")
  def sourceDataProvider() = Array(
    Array("0.1 0.2\t0.3", DenseMatrix((.1f, .2f, .3f))),
    Array("1 2 3\n4  5 \t6 \n", DenseMatrix((1f, 2f, 3f), (4f, 5f, 6f))),
    Array("1 2 3\n 4 5 6\n\n7 8\t9", DenseMatrix((1f, 2f, 3f), (4f, 5f, 6f), (7f, 8f, 9f))),
    Array("\t\t1 2 \n 4 5 \n\n7\t9", DenseMatrix((1f, 2f), (4f, 5f), (7f, 9f)))
  ).map { case Array(str: String, mat) => Array(Source.fromString(str), mat)}
}
