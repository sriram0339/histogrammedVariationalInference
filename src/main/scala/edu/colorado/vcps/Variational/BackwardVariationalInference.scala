package Variational
import java.io.PrintWriter
import StochasticModel._
import scala.annotation.tailrec


class BackwardVariationalInference (override val ip: InferenceProblem, override val learningRate: Double = 1.0, override val batchSize: Int = 25) extends VariationalInference(ip, learningRate, batchSize){


    /*--- Construct KL Divergence Representation and its gradient ---*/

    def computeEntropy(h: Histogram):Double = {
        h.map { prob => {
            if (prob <= 0.0)
                0.0
            else
                prob * math.log(prob)
        }}.sum
    }



    def computeEntropiesForCurrentDistribution(q: VariationalApproximation): List[Double] = {
        q.map (computeEntropy) // Compute Entropy for each parameter separately
    }

    def productOfAllButJ(lst: List[Double], j: Int): Double = {
        assert (j >= 0 && j < lst.length)
        lst.foldLeft ((0, 1.0)) {
            case ((i, prod), elt) => if (i == j)  (i+1, prod) else (i+1, prod * elt)
        }._2
    }

    def computeLogLikelihoodSumUniformCells(cellSamples: List[(ParameterCell, Double)], param: ModelParam, q: VariationalApproximation): List[Double] = {
        val numSubDivs = param.numSubDivs
        val paramIdx = param.paramID
        val factor = numCells.toDouble/cellSamples.size.toDouble
        val zeros = List.fill (numSubDivs) (0.0)
        cellSamples.foldLeft[List[Double]] (zeros) {
            case (curList, (cell, cellLikelihood) ) => {
                val j = cell.indices(paramIdx)
                assert( j >= 0 && j < param.numSubDivs)
                val hist = q(paramIdx)
                val smoothingGradient = if (doSmoothing){
                    val t1 = if (j > 0) 2.0 * math.abs(hist(j) - hist(j-1)) else 0.0
                    val t2 = if (j < param.numSubDivs -1) 2.0 * math.abs(hist(j+1) - hist(j)) else 0.0
                    t1 + t2
                } else {
                    0.0
                }
                val curLikelihoodSumJ = curList(j)
                val cellProbabilities = getCellProbabilities(cell, q)
                val cellProbabilityProduct = productOfAllButJ(cellProbabilities, paramIdx)
                curList.updated(j, curLikelihoodSumJ + cellLikelihood * cellProbabilityProduct * factor - smoothingScaleFactor * smoothingGradient)
            }
        }
    }

    def computeLogLikelihoodSumWithEntropy(cellSamples: Seq[(ParameterCell, Double)], param: ModelParam, q: VariationalApproximation): List[Double] = {
        val numSubDivs = param.numSubDivs
        val paramIdx = param.paramID
        val zeros = List.fill (numSubDivs) (0.0)
        val h = q(paramIdx)
        val entropyTerms = h.map ( p => {
            if ( p <= 1E-08) {
                println(s"Warning small probability : $p ~~ replaced by e^{-25}")
                -25
            } else
                math.log(p) + 1
        } )
        val raw0 = cellSamples.foldLeft[List[Double]] (zeros) {
            case (curList, (cell, cellLikelihood) ) => {
                val j = cell.indices(paramIdx)
                assert( j >= 0 && j < param.numSubDivs)
                val curLikelihoodSumJ = curList(j)
                //val cellProbabilities = getCellProbabilities(cell, q)
                //val cellProbabilityProduct = productOfAllButJ(cellProbabilities, paramIdx)
                curList.updated(j, curLikelihoodSumJ  + cellLikelihood/h(j))
            }
        }
        val unnormGrad = (entropyTerms zip raw0).map(elt => elt._1 - elt._2)
        val fixed =  unnormGrad//MatrixUtils.normalize(unnormGrad)
        val avg = fixed.sum/ fixed.length.toDouble
        fixed.map( _ - avg)
    }

    def computeKLGradientEstimateForParamUniformWithFullEntropy(param: ModelParam, q: VariationalApproximation,  cellSamples: List[(ParameterCell, Double)]) = {
        val paramIdx = param.paramID
        val h = q(paramIdx)
        val entropyTerms = h.map ( p => {
            if ( p <= 1E-08) {
                println(s"Warning small probability : $p ~~ replaced by e^{-25}")
                 (-25)
            } else
                (math.log(p) + 1 )
        } )
        val logLikelihoodSum = computeLogLikelihoodSumUniformCells(cellSamples, param, q)
        val unnormGrad = (entropyTerms zip logLikelihoodSum).map(elt => elt._1 - elt._2)
        val clippedGrad = if (doClipping) {
             (unnormGrad zip h).map {
                case (ug, p) => if (p <= clipThreshold && ug > 0) {
                    println(s"Warning: Clipping gradient for probability $p")
                    0.0
                } else {
                    ug
                }
            }
        } else {
            unnormGrad
        }
        val normedGrad = clippedGrad //MatrixUtils.normalize(clippedGrad)
        val avg = normedGrad.sum/ normedGrad.length.toDouble
        normedGrad.map( _ - avg)
    }

    def computeKLDivergenceEstimate(q: VariationalApproximation, cellSamples: Seq[(ParameterCell, Double)]): Double = {
        val e = computeEntropiesForCurrentDistribution(q).sum
        val lSum = cellSamples.foldLeft(0.0) {
            case (acc: Double, (cell: ParameterCell, l: Double)) => {
                //val cellProbability = getCellProbabilities(cell, q).product
                l + acc
            }
        }
        e - lSum
    }

    def computeKLGradient(q: VariationalApproximation, cellSamples:Seq[(ParameterCell, Double)] ): List[List[Double]] = {

        params.map(computeLogLikelihoodSumWithEntropy(cellSamples, _, q))
        //params.map( computeKLGradientEstimateForParamUniformWithFullEntropy(_, q, cellSamples))
    }




/*

    def runAdamForNSteps(n: Int, lRateMin:Double = terminalLearningRate ): VariationalApproximation = {
        val q0: List[List[Double]] = randomInitialize//params.map (p => List.fill (p.numSubDivs) (1.0 / p.numSubDivs.toDouble))
        val a = initializeAdamState(q0, computeKLGradient)

        @tailrec
        def runAdamUntilCompletion(k: Int, a: AdamState): AdamState = {
            val runTests = true

            if (k <= 0) {
                println(s"Adam: Completed after $n steps")
                a
            } else {
                val (newA, lRate) = a.updateAdamState
                if (runTests && k % 100 == 0) {

                    val cells = sampleFromCurrentDistribution(a.q, 500)
                    val klDivEst = computeKLDivergenceEstimate(a.q, cells)
                   // println(s"KL Divergence from 500 test samples : $klDivEst")
                    println(s"Adam: Iteration # ${n-k+1}, learning rate: $lRate, KL Estimate (500 samples): $klDivEst")
                }
                if (lRate <= lRateMin) {
                    println(s"Adam: Completed after ${n-k} steps with learning rate $lRate < min rate $lRateMin")
                    newA
                } else {
                    runAdamUntilCompletion(k - 1, newA)
                }
            }
        }

        val newA = runAdamUntilCompletion(n, a)
        newA.q
    }


 */

    def runVanillaSGD(n: Int, runTests: Boolean = false): VariationalApproximation = {
        //val runTests = true
        var q: List[List[Double]] =  uniformDistribution //randomInitialize //params.map (p => List.fill (p.numSubDivs) (1.0 / p.numSubDivs.toDouble))
        val cells0 = sampleFromCurrentDistribution(q,500)
        val klDivEst0 = computeKLDivergenceEstimate(q, cells0)
        println(s"KL Divergence from test samples : $klDivEst0")
        val timeNow = System.nanoTime()
        for (k <- 1 to n){

            if (k % 100 == 0) {
                println(s"-- Iteration # $k -- ")
            }

            val cellSamples = sampleFromCurrentDistribution(q)//uniformSamples(nSamplesPerIteration)
            val g = computeKLGradient(q, cellSamples)
            val lRate = MatrixUtils.findLearningRate(q, g, learningRate)
            if (k % 500 == 0) {
                println(s"\t Learning Rate: $lRate, norm of gradient: ${MatrixUtils.l2NormMat(g)}, # samples: ${cellSamples.size}")
                val timeCurrent = System.nanoTime()
                println(s"Elapsed time: ${((timeCurrent-timeNow).toDouble * 1E-09)}")
                if (runTests ) {
                    val cells = sampleFromCurrentDistribution(q, 500)
                    val klDivEst = computeKLDivergenceEstimate(q, cells)
                    println(s"KL Divergence from test samples : $klDivEst")
                }
            }
            q = MatrixUtils.linearCombinationOfMatrices(q, 1.0, g, - 0.1 * lRate)
            if (lRate <= terminalLearningRate){
                println("Learning rate too low -- terminating")
                return q
            }
        }
        val cells = sampleFromCurrentDistribution(q, 500)//uniformSamples(500)
        val klDivEst = computeKLDivergenceEstimate(q, cells)
        println(s"KL Divergence from test samples : $klDivEst")
        val timeFinish = System.nanoTime()
        println(s"Execution time (total): ${((timeFinish-timeNow).toDouble * 1E-09)}")
        q
    }

    def runIndependentSGDStep(q: VariationalApproximation, p : ModelParam): VariationalApproximation = {
        val cellSamples = sampleFromCurrentDistribution(q)
        //val klDiv0 = computeKLDivergenceEstimate(q, cellSamples)
        //println(s"Estimated --> KL Divergence (pre update)  = $klDiv0")
        val g = computeLogLikelihoodSumWithEntropy(cellSamples, p, q)
        val h = q(p.paramID)
        val lRate = MatrixUtils.findLearningRate(List(h), List(g), learningRate)
        println(s"Stochastic Gradient Step for parameter ${p.paramName} (${p.paramID} with learning rate: $lRate)")
        println(s"norm of the gradient is: ${MatrixUtils.l2Norm(g)}")
        val newHist = (h zip g).map { case (p:Double, grad: Double) => p -  lRate * grad}
        val qHat = q.updated(p.paramID, newHist)
        //val klDiv = computeKLDivergenceEstimate(qHat, cellSamples)
        //println(s"Estimated --> KL Divergence (post update) = $klDiv")
        qHat
    }

    def runIndependentSGD(q: VariationalApproximation): VariationalApproximation = {
        params.foldLeft (q) ((qHat, p) => runIndependentSGDStep(qHat, p))
    }

    def runSGDWithIndependentUpdates(n: Int): VariationalApproximation = {
        var q: List[List[Double]] = randomInitialize//params.map (p => List.fill (p.numSubDivs) (1.0 / p.numSubDivs.toDouble))
        val runTests = true
        for (k <- 1 to n) {
            println(s"-- Iteration # $k -- ")
            if (runTests && k % 10 == 1) {
                val cells = sampleFromCurrentDistribution(q,500)
                val klDivEst = computeKLDivergenceEstimate(q, cells)
                println(s"KL Divergence from test samples : $klDivEst")
            }
            q = runIndependentSGD(q)
        }
        q
    }



    /*--- Run Lagrange optimization ---*/
    /*
    def lagrangeMinimize(p: ModelParam, q: VariationalApproximation,  cells: Set[(ParameterCell, Double)]): Histogram  = {
        /*-- Solve Lagrange minimization at each steps--*/
        val paramID: Int = p.paramID
        val h: Histogram = q(paramID)
        /*-- Compute solution --*/
        val likelihoodSums = computeLogLikelihoodSum(cells, p, q)
        val expLikelihoodSums = likelihoodSums.map (math.exp(_))
        val denom = expLikelihoodSums.sum
        expLikelihoodSums.map(_/denom)
    }

    def lagrangeMinimizationIterations(q: VariationalApproximation): VariationalApproximation = {
        params.foldLeft (q) {
            case (qHat, p) => {
                val cells = sampleFromCurrentDistribution(q)
                val id = p.paramID
                val h1 = lagrangeMinimize(p, q, cells)
                q.updated(id, h1)
            }
        }
    }

    def performNLagrangeIterations(n: Int): VariationalApproximation = {
        var q: List[List[Double]] = uniformDistribution
        for (j <- 1 to n){
            println(s"Lagrange Iteration ${j}")
            q = lagrangeMinimizationIterations(q)

        }
        q
    }*/
}
