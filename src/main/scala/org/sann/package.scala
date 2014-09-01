/**
 * Copyright (c) 2014 Liang Kun. All Rights Reserved.
 * Authors: Liang Kun <liangkun@data-intelli.com>
 */

package org

import breeze.linalg.Tensor
import breeze.numerics._

package object sann {

  /**
   * A group of isomorphism neurons.
   *
   * @param cardinality number of neurons in this group.
   * @param activator activating function of the neurons.
   * @param inputs synapses that provide inputs to each of the neuron in this group.
   */
  case class Neurons(
    cardinality: Int,
    activator: Tensor[Int, Double] => Tensor[Int, Double],
    inputs: Iterable[Synapses]
  )

  /**
   * A group of synapses that connects every input neurons to every output neurons.
   *
   * @see Neurons for more information.
   * @param inputs input neurons.
   * @param outputs output neurons.
   * @param weights weight of these synapses.
   * @param updater update function of the weights.
   * @param info extra information used by updater.
   */
  case class Synapses(
    inputs: Iterable[Neurons],
    outputs: Iterable[Neurons],
    weights: Tensor[(Int, Int), Double],
    updater: Synapses => Any,
    info: Any
  )

  // =============================================================================================
  // Common Activator definitions.
  // =============================================================================================
  def linear(zs: Tensor[Int, Double]): Tensor[Int, Double] = zs
}
