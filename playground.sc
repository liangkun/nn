import breeze.linalg._
import breeze.stats._
import breeze.plot._
import org.sann._

val p = inputs(2)->Neurons(1, linearActivator)
val m = compile(p)

val data = Array (
  (DenseVector(.0, .1), DenseVector(1.0)),
  (DenseVector(.1, .0), DenseVector(-1.0))
)

m.ff(DenseVector(.1, .0))
m.train(data)
m.ff(DenseVector(.0, .1))
m.ff(DenseVector(.1, .0))