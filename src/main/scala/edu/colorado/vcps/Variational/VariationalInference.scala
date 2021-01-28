package Variational

import java.io.PrintWriter
import java.time.LocalDateTime

import StochasticModel._

import scala.util.Random

class VariationalInference(val ip: InferenceProblem, val learningRate: Double = 1.0, val batchSize: Int = 25) {

    val params: List[ModelParam] = ip.params
    val numParams: Int = params.length
    type Histogram = List[Double]
    type VariationalApproximation = List[Histogram]
    /* -- SGD parameters --*/
    val terminalLearningRate = 1E-16
    val beta1 = 0.9
    val beta2 = 0.999
    val doClipping = true
    val clipThreshold = 1E-08 // If doClipping is enabled
    val s = 0.7 // Fraction of max step to be taken at each iteration
    /*-- Method parameters --*/
    val nSamplesPerIteration = batchSize
    val r: Random = new Random(LocalDateTime.now().getNano())
    val smoothingScaleFactor  = 2.0
    val doSmoothing = false
    val numCells: Int = params.map(_.numSubDivs).product


    def sampleFromHistogram(h: Histogram): Int = {
        assert(math.abs(h.sum -1.0) <= 1E-06, s" histogram not normalized: sum is ${h.sum}")
        val fRand = r.nextDouble() // Generate a uniform number in [0,1]
        val histPref = h.scan (0.0) ( _ + _ ) // Compute prefix sums
        val indexedHistPref: Seq[(Int, Double)] = histPref.indices zip histPref // Add indices
        val selectedElement: Option[(Int, Double)] = indexedHistPref.find ( elt => elt._2 >= fRand) // Select an element
        assert(selectedElement.isDefined)
        val j = (selectedElement.get._1) - 1
        assert( j >= 0 && j < h.length, s"Illegal sample from histogram $j -- histogram length: ${h.length}")
        j
    }

    def singleSampleFromCurrentDistribution(q: VariationalApproximation): ParameterCell = {
        val indices: List[Int] = q.map( (h:Histogram) => sampleFromHistogram(h))
        new ParameterCell(indices, params)
    }


    def sampleOneUniformCell = {
        val indices: List[Int] = params.map (p => (r.nextInt(p.numSubDivs)))
        new ParameterCell(indices, params)
    }

    def uniformSamples(nSample: Int = nSamplesPerIteration): Seq[(ParameterCell, Double)] = {
        (1 to nSample).map ( (j: Int) => {
            val cell = sampleOneUniformCell
            val cellLikelihood = ip.computeCellLogLikelihood(cell)
            (cell, cellLikelihood)
        })
    }

    def sampleFromCurrentDistribution(q: VariationalApproximation, nSamples: Int = nSamplesPerIteration): List[(ParameterCell, Double)] = {
        /*  (1 to nSamples)
            .map (elt => {
                val cell = singleSampleFromCurrentDistribution((q))
                val cellLikelihood = ip.computeCellLogLikelihood(cell)
                (cell, cellLikelihood)
            })
            .toSet

         */

        (1 to nSamples).map ( (j: Int) => {
            val cell = singleSampleFromCurrentDistribution(q)
            val cellLikelihood = ip.computeCellLogLikelihood(cell)
            (cell, cellLikelihood)
            }
        ).toList
    }


    def uniformDistribution: VariationalApproximation = {
        params.map (p => List.fill (p.numSubDivs) (1.0 / p.numSubDivs.toDouble))
    }

    def getRandomHistogram(param: ModelParam) = {
        val nSubDivs = param.numSubDivs
        val lst = List.fill (nSubDivs) (r.nextDouble())
        val lstSum = lst.sum
        lst.map (_/lstSum)
    }

    def randomInitialize: VariationalApproximation = {
        params.map ( p => getRandomHistogram(p))
    }

    object MatrixUtils {
        def applyMatrix(mat: List[List[Double]])(fun: Double => Double): List[List[Double]] = {
            mat.map ( lst => lst.map (entry => fun(entry)))
        }

        def scaleMatrix(list: List[List[Double]], d: Double): List[List[Double]] = applyMatrix (list)(_ * d)

        def squareEntrywise(list: List[List[Double]]): List[List[Double]] = applyMatrix(list) (x => x * x)

        def applyMatrices(mat1: List[List[Double]], mat2: List[List[Double]])(fun: (Double, Double)=> Double): List[List[Double]] = {
            assert(mat1.length == mat2.length)
            (mat1 zip mat2).map {
                case (vec1: List[Double], vec2: List[Double]) => {
                    assert(vec1.length == vec2.length)
                    (vec1 zip vec2).map(x => fun(x._1, x._2))
                }
            }
        }

        def linearCombinationOfMatrices(mat1: List[List[Double]], c1: Double, mat2: List[List[Double]], c2: Double):List[List[Double]] = {
            if (c2 == 0)
                scaleMatrix(mat1, c1)
            else if (c1 == 0)
                scaleMatrix(mat2, c2)
            else
                applyMatrices(mat1, mat2) ((x, y) => c1 * x + c2 * y)
        }

        def l2Norm(vec: Histogram) : Double = {
            math.sqrt(vec.map (elt => elt * elt).sum)
        }
        def l2NormMat(q:VariationalApproximation) : Double = {
            math.sqrt(  q.flatMap( vec => vec.map (elt => elt * elt)).sum)
        }

        def normalize(vec: List[Double]) = {
            val rms = math.sqrt(vec.map (elt => elt * elt).sum)
            vec.map (_/rms)
        }

        def findLearningRate(q: List[List[Double]], dir: List[List[Double]], initRate: Double): Double = {
            val newRate = (q zip dir). map {
                case (vec1: List[Double] , vec2: List[Double]) => {
                    (vec1 zip vec2).map ({
                        case (p, d) =>
                            if (p > clipThreshold && p - initRate * d < clipThreshold ) { s * (p-clipThreshold)/d }
                            else { initRate }
                    }).min
                }}.min
            List(initRate, newRate).min
        }
    }

    def getCellProbabilities(cell: ParameterCell, q: VariationalApproximation): List[Double] = {
        (0 until numParams).map(paramIdx => {
            val j = cell.indices(paramIdx)
            q(paramIdx)(j)
        }).toList
    }

    def dumpDistributionOntoCSVFiles(q: VariationalApproximation, fileStem:String): Unit = {
        def printParameter(p: ModelParam): Unit = {
            val file1 = s"$fileStem${p.paramName}.csv"
            val fh1 = new PrintWriter(file1)
            for (j <- 0 until p.numSubDivs){
                fh1.println(s"${p.getCenterForSubdivison(j)}, ${q(p.paramID)(j)} ")
            }
            fh1.close()
        }
        params.foreach(printParameter)
    }

    /*-- Do some posterior simulations --*/

    def simulateSystemOnCellsAndDumpTrajectories(fileStem: String, sampleCells: Seq[(ParameterCell, Double)]) = {
        val sys: DiscreteTimeStochasticModel = ip.sys
        var idx = 0
        for ((cell, l) <- sampleCells) {
            val traj: List[List[Double]] = sys.simulateNSteps( 2* ip.numSteps, cell.centerPoint)
            val fileName = s"$fileStem-$idx"
            val fHandle = new PrintWriter(fileName)
            traj.zipWithIndex.foreach {
                case (state: List[Double], j: Int) => {
                    fHandle.print(s"$j")
                    state.foreach(vi => fHandle.print(s", $vi"))
                    fHandle.println("")
                }
            }
            fHandle.close()
            idx = idx + 1
        }
    }
    def doPosteriorSims(qF: VariationalApproximation, nSims: Int, fileStem: String) = {
        val sampleCells = sampleFromCurrentDistribution(qF, nSims)
        simulateSystemOnCellsAndDumpTrajectories(fileStem, sampleCells)
    }

    def doUniformSims(nSims: Int, fileStem: String)  = {
        val sampleCells = uniformSamples(nSims)
        simulateSystemOnCellsAndDumpTrajectories(fileStem, sampleCells)
    }

/*

    class AdamState (val q: VariationalApproximation,
                     val gradFun: (VariationalApproximation, List[(ParameterCell, Double)]) => VariationalApproximation,
                     val momentum: List[List[Double]],
                     val variance: List[List[Double]],
                     val time : Double,
                     val beta1t: Double,
                     val beta2t: Double,
                     val learningRate: Double = 0.05,
                     val beta1: Double = 0.9,
                     val beta2: Double = 0.99,
                     val eps: Double = 1E-06) {
        val debug = true


        def computeAdamDirection(momentum: List[List[Double]], variance: List[List[Double]]): List[List[Double]]  =
            MatrixUtils.applyMatrices(momentum, variance) {
                case (mij, vij) => mij/(math.sqrt(vij) + eps)
            }



        def updateAdamState: (AdamState, Double) = {
            val cellSamples: List[(ParameterCell, Double)] = uniformSamples()//sampleFromCurrentDistribution(q)
            val grad = gradFun(q, cellSamples)//computeKLGradient(q, cellSamples)
            val updatedMomentum = MatrixUtils.linearCombinationOfMatrices(momentum, beta1, grad, 1.0 - beta1)
            val squaredGrad = MatrixUtils.squareEntrywise(grad)
            val updatedVariance = MatrixUtils.linearCombinationOfMatrices(variance, beta2, squaredGrad, 1.0 - beta2)
            val beta1Factor = 1.0/(1.0 - beta1t)
            val scaledMomentum = MatrixUtils.scaleMatrix(updatedMomentum, beta1Factor)
            val beta2Factor = 1.0/(1.0 - beta2t)
            val scaledVariance = MatrixUtils.scaleMatrix(updatedVariance, beta2Factor)
            val direction = computeAdamDirection(scaledMomentum, scaledVariance)
            val projectedDirection = direction.map ( vec => {
                val avg = vec.sum / vec.length.toDouble
                vec.map (_ - avg)
            })
            // val normalizedDirection = projectedDirection.map (MatrixUtils.normalize(_))
            val lRate = MatrixUtils.findLearningRate(q, projectedDirection, learningRate)
            //if (debug){
            //    println(s"Learning Rate set to : $lRate")
            //}
            val newQ = MatrixUtils.linearCombinationOfMatrices(q, 1.0, projectedDirection, -1.0 * lRate)
            (new AdamState(newQ, gradFun, updatedMomentum, updatedVariance, time+1, beta1t * beta1, beta2t * beta2, learningRate, beta1, beta2, eps),
              lRate)
        }
    }

    def initializeAdamState(q0: List[List[Double]], gradFun: (VariationalApproximation, List[(ParameterCell, Double)])=> VariationalApproximation): AdamState = {
        /* - Start off with a uniform distribution. We can figure out other ways to initialize later -*/

        val momentum = params.map ( p => List.fill (p.numSubDivs) (0.0))
        val variance = params.map ( p => List.fill (p.numSubDivs) (0.0))
        new AdamState(q0, gradFun, momentum, variance, 1, beta1, beta2, learningRate, beta1, beta2)
    }

 */

}
