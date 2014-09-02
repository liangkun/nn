/**
 * Copyright (c) 2014 Liang Kun. All Rights Reserved.
 * Authors: Liang Kun <liangkun@data-intelli.com>
 */

package org

import breeze.linalg._

package object sann {
  /** Type of an activating function. */
  type Activator = DenseVector[Double] => DenseVector[Double]

  /**
   * Type of weights updating (learning) function.
   * (current weights, current input, current error, learner private info) => learner private info
   */
  type Learner =
    (DenseMatrix[Double], DenseVector[Double], DenseVector[Double], Option[Any]) => Option[Any]

  /**
   * A group of isomorphism neurons.
   *
   * @param cardinality number of neurons in this group.
   * @param activator activating function of the neurons.
   * @param inputs synapses that provide inputs to each of the neuron in this group.
   */
  case class Neurons(cardinality: Int, activator: Activator, inputs: List[Synapses] = List())

  /**
   * A group of synapses that connects every input neurons to every output neurons.
   *
   * @param input input neurons.
   * @param updater update function of the weights.
   */
  case class Synapses(input: Neurons, updater: Learner)

  // =============================================================================================
  // Neural Networks DSL Definition.
  // =============================================================================================
  case class SannNode(self: Neurons) {
    def ->(neurons: Neurons): Neurons = Neurons(
      neurons.cardinality,
      neurons.activator,
      Synapses(self, perceptronLearner)::neurons.inputs
    )
  }

  def inputs(size: Int) = Neurons(size, linearActivator, List())

  implicit def neuronsToSannNode(neurons: Neurons) = SannNode(neurons)

  def perceptronLearner(
    weights: DenseMatrix[Double],
    input: DenseVector[Double],
    error: DenseVector[Double],
    info: Option[Any]): Option[Any] = {

    weights(::, *) += input * error(0)

    None
  }

  // =============================================================================================
  // Compiled neural network
  // =============================================================================================
  class CompiledSnn {
    var weights: List[DenseMatrix[Double]] = List()
    var learners: List[Learner] = List()
    var activators: List[Activator] = List()
    private[this] var _inputs: List[DenseVector[Double]] = List()

    def train(trainSet: Iterable[(DenseVector[Double], DenseVector[Double])]):Unit = {
      var done = false

      while(!done) {
        var hasError = false
        for ((input, expected) <- trainSet) {
          val output = ff(input)
          var error = 0.0
          if (output(0) > 0 && expected(0) < 0) {
            error = -1.0
          } else if (output(0) < 0 && expected(0) > 0) {
            error = 1.0
          }
          if (error != 0.0) {
            hasError = true
            bp(DenseVector(error))
          }
        }
        done = !hasError
      }
    }

    def ff(input: DenseVector[Double]): DenseVector[Double] = {
      _inputs = List(input)
      for ((w, f) <- weights.zip(activators)) {
        val output = f(_inputs.head)
        _inputs = (w.t * output) :: _inputs
      }
      _inputs.head
    }

    def bp(error: DenseVector[Double]): Unit = {
      val ws = weights.reverse.tail
      val ls = learners.reverse
      val ins = _inputs.tail
      for (i <- 0 until ws.size) {
        ls(i)(ws(i), ins(i), error, None)
      }
    }
  }

  def compile(output: Neurons): CompiledSnn = {
    val result = new CompiledSnn

    if (output.inputs.size == 1) {
      val synapses = output.inputs(0)
      val subgraph = compile(synapses.input)
      result.weights = subgraph.weights ++ List(DenseMatrix.rand[Double](synapses.input.cardinality, output.cardinality), DenseMatrix.eye[Double](output.cardinality))
      result.activators = subgraph.activators :+ output.activator
      result.learners = subgraph.learners :+ synapses.updater
    }
    result
  }

  // =============================================================================================
  // Common Activator definitions.
  // =============================================================================================

  def linearActivator(zs: DenseVector[Double]): DenseVector[Double] = zs
}
