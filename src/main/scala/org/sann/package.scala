/**
 * Copyright (c) 2014 Liang Kun. All Rights Reserved.
 * Authors: Liang Kun <liangkun@data-intelli.com>
 */

package org

import scala.language.implicitConversions

import breeze.linalg._

package object sann {
  /** Type of an activating function (vectorized). */
  case class Activator(evaluate: (DenseVector[Double] => DenseVector[Double]),
                       derivate: (DenseVector[Double] => DenseVector[Double]))

  /**
   * A group of isomorphism neurons.
   *
   * @param cardinality number of neurons in this group.
   * @param activator activating function of the neurons.
   * @param inputs synapses that provide inputs to each of the neuron in this group, order is important.
   */
  case class Neurons(cardinality: Int, activator: Activator, inputs: List[Synapses] = List()) {
    def ->(neurons: Neurons) = neurons.copy(inputs = Synapses(this) :: neurons.inputs)
  }

  /**
   * A group of synapses that have the input neurons connected as input.
   *
   * @param input input neurons.
   */
  case class Synapses(input: Neurons)

  /**
   * Input (from external) neurons of the network.
   *
   * @param cardinality of the input neurons.
   */
  def inputs(cardinality: Int) = Neurons(cardinality, linearActivator, List())

  /** Linear(identity) Activator */
  val linearActivator = Activator(
    evaluate = x => x,
    derivate = x => DenseVector.ones(x.length)
  )

  /**
   * Compiled Sann with compact internal representation, ready for training and working.
   */
  class CompiledSann {

  }

  def compile(output: Neurons): CompiledSann = {
    val neuronsCount = countCascade(output)
    ???
  }

  // count all the neurons that can reach the specified neurons.
  private def countCascade(neurons: Neurons, seen: Set[Neurons] = Set()): Int = {
    if (seen.contains(neurons)) {
      0
    } else {
      var count = 1
      val updatedSeen = seen + neurons
      for (input <- neurons.inputs) {
        count += countCascade(input.input, updatedSeen)
      }
      count
    }
  }

  // =============================================================================================
  // Internal
  // =============================================================================================

  def perceptronLearner(
    weights: DenseMatrix[Double],
    input: DenseVector[Double],
    error: DenseVector[Double],
    info: Option[Any]): Option[Any] = {

    weights(::, *) += input * error(0)

    None
  }
}
