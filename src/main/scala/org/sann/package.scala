/**
 * Copyright (c) 2014 Liang Kun. All Rights Reserved.
 * Authors: Liang Kun <liangkun@data-intelli.com>
 */

package org

import breeze.linalg._

import scala.collection.mutable.ArrayBuffer
import scala.io.Source
import scala.util.Random

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

  // =============================================================================================
  // developer APIs.
  // =============================================================================================

  /**
   * Load data in text format.
   *
   * Each line contains a row of the result matrix, values are seperated by white space(s).
   * @param source containing data in text format.
   * @return matrix of the text source.
   */
  def loadData(source: Source): DenseMatrix[Float] = {
    val builder = ArrayBuffer.empty[Float]
    var dimension = -1

    for (line <- source.getLines(); if line.nonEmpty) {
      val point = line.trim().split("\\s+").map(_.toFloat)
      if (dimension < 0) {
        dimension = point.size
      } else {
        require(point.size == dimension, s"data points have different dimentions: $dimension, ${point.size}")
      }
      builder ++= point
    }

    val data = builder.toArray
    new DenseMatrix[Float](data.size / dimension, dimension, data, 0, dimension, isTranspose = true)
  }

  /**
   * Perceptron learning.
   *
   * @param weights initial weights.
   * @param xs input features of the training data.
   * @param ys labels of the training data, -1 for negative, +1 for positive
   * @param maxIters max iterations.
   * @param learningRate of the trainning.
   * @param pocket use pocket pla.
   * @return totalIterations and remain errors.
   */
  def perceptron(
      weights: DenseVector[Float],
      xs: DenseMatrix[Float], ys: DenseVector[Float],
      maxIters: Int = Int.MaxValue,
      learningRate: Float = 1f,
      pocket: Boolean = false): (Int, Int) = {
    var iters = 0

    var pocketWeights = if (pocket) weights.copy else DenseVector(0f)
    var pocketErrors = if (pocket) countErrors(weights, xs, ys) else 0

    var next = nextError(-1, weights, xs, ys)
    while(next >= 0 && iters <= maxIters) {
      assert(sign(xs(next, ::) * weights) != ys(next))
      val delta = xs(next, ::).t * (ys(next) * learningRate)
      weights := weights + delta

      if (pocket) {
        val errors = countErrors(weights, xs, ys)
        if (errors < pocketErrors) {
          pocketErrors = errors
          pocketWeights := weights
        }
      }

      next = nextError(next, weights, xs, ys)
      iters += 1
    }

    if (pocket) {
      weights := pocketWeights
    }

    val errors = if (pocket) {
      pocketErrors
    } else if (iters <= maxIters) {
      0
    } else {
      countErrors(weights, xs, ys)
    }

    (iters, errors)
  }

  /**
   * Counter errors for perceptron.
   *
   * @param weights perceptron weights.
   * @param xs input features.
   * @param ys correct labels
   * @return error count.
   */
  def countErrors(weights: DenseVector[Float], xs: DenseMatrix[Float], ys: DenseVector[Float]): Int = {
    val predicts = (xs * weights).mapValues(x => sign(x))
    sum((predicts :!= ys).mapValues(c => if (c) 1 else 0))
  }

  /** find next error of the perceptron */
  def nextError(current: Int, weights: DenseVector[Float], xs: DenseMatrix[Float], ys: DenseVector[Float]): Int = {
    for (offset <- 1 to xs.rows) {
      val index = (current + offset) % xs.rows
      if (sign(xs(index, ::) * weights) != ys(index)) return index
    }

    -1  // no errors
  }

  /** sign function that make 0 to be -1. */
  def sign(x: Float): Float = if (x <= 0) -1f else 1f
}
