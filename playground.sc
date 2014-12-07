// Copyright (c) 2014 Liang Kun. All Rights Reserved.
// Authors: Liang Kun <liangkun@data-intelli.com>

import breeze.linalg._
import breeze.numerics._
import breeze.stats._
import breeze.stats.distributions._
import breeze.plot._
import org.sann._

log(1.0)

val m = DenseMatrix((1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0))
m(List(0, 2), 2)