package Variational

import StochasticModel._

import scala.annotation.tailrec


class ForwardVariationalInference(override val ip: InferenceProblem, override val learningRate: Double = 1.0, override val batchSize: Int = 25) extends VariationalInference(ip, learningRate, batchSize){
/*
    val totalLikelihood = cells.map (elt => math.exp(elt._2)).sum

    def performInferenceForParameter(p: ModelParam): List[Double] = {
        val paramIdx = p.paramID
        /*
            For each subdivision, collect cells that belong to that subdivision and
            accumulate the likelihood
         */
        val numSubDivs = p.numSubDivs
        val zeros = List.fill (numSubDivs) (0.0)
        val unnormalizedLikelihood = cells.foldLeft[List[Double]] (zeros) {
            case (curList, (pCell, likelihood)) => {
                val subdivIdx = pCell.indices(paramIdx)
                assert(subdivIdx >= 0 && subdivIdx < numSubDivs)
                val subdivLikelihood = curList(subdivIdx)
                curList.updated(subdivIdx, math.exp(likelihood) + subdivLikelihood)
            }
        }
        val total = unnormalizedLikelihood.sum
        unnormalizedLikelihood.map (elt => elt / total)
    }

    def performInference: List[List[Double]] = {
        params.map (performInferenceForParameter(_))
    }
 */
    type LogProbabilities = List[List[Double]]
    type LogHistogram = List[Double]

    def computeForwardKLGradientOnLogParameters(q: LogProbabilities,
                                                cells: Seq[(ParameterCell, Double)],
                                                p: ModelParam): LogHistogram = {
        /*
            KL ~ \sum
         */
        val paramIdx = p.paramID
        val nSubDivs = p.numSubDivs
        val factor = numCells.toDouble / (cells.size.toDouble)
        val zeros = List.fill (nSubDivs) (0.0)
        val derivUnnormalized = cells.foldLeft (zeros){
            case (curList, (cell, cellLogLikelihood)) => {
                val cellIdx = cell.indices(paramIdx)
                assert(cellIdx >= 0 && cellIdx < nSubDivs)
                curList.updated(cellIdx, curList(cellIdx) - factor * math.exp(cellLogLikelihood))
            }
        }

        derivUnnormalized
       // MatrixUtils.normalize(derivUnnormalized)
    }

    def sgdStepProbabilities(q: VariationalApproximation, maxStepSize: Double = learningRate): VariationalApproximation = {
        val logQ = MatrixUtils.applyMatrix (q) (x => { if (x > clipThreshold) Math.log(x) else -25.0 } )
        val cells = uniformSamples()
        val dLogQ = params.map ( p => computeForwardKLGradientOnLogParameters(logQ, cells, p))
        val logQHat = MatrixUtils.linearCombinationOfMatrices(logQ, 1.0, dLogQ, -1.0 * learningRate)
        val qHat = MatrixUtils.applyMatrix(logQHat) (Math.exp)
        val diffQ = MatrixUtils.linearCombinationOfMatrices(qHat, 1.0, q, -1.0 )
        val diffQProjected = diffQ.map (h => {
            val hAvg = h.sum / h.length.toDouble
            h.map (_ - hAvg)
        })
        MatrixUtils.linearCombinationOfMatrices(q, 1.0, diffQProjected, 1.0)
    }

    def computeForwardKLEstimate(q: VariationalApproximation, nCells: Int): Double = {
        val cells = uniformSamples(nCells)
        val fact = numCells.toDouble / nCells.toDouble
        cells.foldLeft (0.0) {
            case (curValue, (cell, cellLogLikelihood)) => {
                val cellLogProbabilities = getCellProbabilities(cell, q)
                  .map (Math.log)
                  .sum

                curValue - fact * (math.exp(cellLogLikelihood) * cellLogProbabilities)
            }
        }
    }

    def applyVanillaSGD(numSteps: Int): VariationalApproximation = {
        var q = uniformDistribution //randomInitialize
        var curLearningRate = learningRate
        for (j <- 1 to numSteps){
            if (j % 1000 == 1) {
                val klEst = computeForwardKLEstimate(q, 1000)
                println(s"Iteration : $j -- KL Estimate: $klEst")
            }
            if (j > 1 && j % 500 == 1 ){
                curLearningRate = learningRate * 500.toDouble/j.toDouble
                println(s"Adjusting Learning Rate to : $curLearningRate")
            }
            q = sgdStepProbabilities(q, curLearningRate)
        }
        q
    }




}
