/**
 * Copyright (c) 2014 Liang Kun. All Rights Reserved.
 * Authors: Liang Kun <liangkun@data-intelli.com>
 */
package org.sann

import org.scalatest.prop.TableDrivenPropertyChecks
import org.scalatest.{OptionValues, Matchers, FlatSpec}

/** Base class of unit tests. */
abstract class UnitSpec extends FlatSpec
  with Matchers
  with OptionValues
  with TableDrivenPropertyChecks
