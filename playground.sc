import breeze.linalg._
import breeze.numerics._
import breeze.stats._
import breeze.stats.distributions._
import breeze.plot._
import org.sann.algos.Perceptron
import org.sann.algos.Perceptron.ListErrorCollector
import org.sann.data.gen.DoubleMoon
import org.sann._

val size = 1000
val train = DoubleMoon.gen(size, -2)
val xs = DenseMatrix.horzcat(DenseMatrix.ones[Float](size, 1), train(::, 0 to 1))
val ys = train(::, 2)
val weights = DenseVector(2f, 1f, 1f)
val ecollector = new ListErrorCollector
val (iters, error) = Perceptron.train(
  weights,
  xs,
  ys,
  maxIters = 1000,
  learningRate = 1f,
  pocket = true,
  errorCollector = Some(ecollector)
)
val fig = new Figure("Playground")
val plt = fig.subplot(1, 2, 0)
utils.plotBinaryDataPoints(train).foreach(s => plt += s)
plt += utils.plotLine(weights, -14.0f, 24.0f)
val plt1 = fig.subplot(1, 2, 1)
plt1 += utils.plotLearningCurve(ecollector.get)

