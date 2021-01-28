package StochasticModel

/* -- Class for model parameters --*/
class ModelParam (val paramID: Int, val paramName: String, val paramLB: Double, val paramUB: Double, val numSubDivs: Int) {
    assert(paramLB <= paramUB, "Invalid parameter bounds lower: $paramLB  upper: $paramUB")
    override def toString: String = {s"$paramName($paramID, $paramLB, $paramUB, $numSubDivs)"}

    val delta = (paramUB - paramLB) / numSubDivs.toDouble

    def contains(d: Double) = paramLB <= d && d <= paramUB

    def getLogPriorForSubdivision(j: Int): Double = {
        -1.0 * math.log(numSubDivs.toDouble)
    }

    def getCenterForSubdivison(j: Int): Double = {
        require(j >= 0 && j < numSubDivs)
        paramLB + delta * 0.5 + j.toDouble * delta
    }

    def getLowerBoundForSubdivision(j: Int): Double = {
        require(j >= 0 && j < numSubDivs)
        paramLB + delta * j.toDouble
    }

    def getUpperBoundForSubdivision(j: Int): Double = {
        require(j >= 0 && j < numSubDivs)
        paramLB + delta * j.toDouble + delta
    }


    def nextSubdivision(k: Int): Int = {
        require (k >= 0 && k < numSubDivs)
        (k + 1) % numSubDivs
    }

    def previousSubdivision(k: Int): Int = {
        require (k >= 0 && k < numSubDivs)
        if (k == 0)
            numSubDivs -1
        else
            k - 1
    }

}


abstract class DiscreteTimeStochasticModel(initCond: List[Double], params: List[ModelParam]) {

    def next(curState: List[Double], params: List[Double]): List[Double]

    def simulateNSteps(n: Int, paramList: List[Double]): List[ List[Double]] = {
       // println(s"Debug: Simulate $paramList")
        val retList = (0 until n).foldLeft[List[List[Double]]] (List(initCond)) {
            case (acc, _) => this.next(acc.head, paramList):: acc
        }
       // println(retList)
        retList.reverse
    }
}


class ObservationData(val observationData: List[(Int, Int, Double)], val observationNoiseSD: Double=1.0) {

    def computeLogLikelihoodGaussianNoise(trajectory: List[List[Double]]): Double = {
        val initialLogLikelihood = 1.0
        def applySingleObservation(actualValue:Double, observedValue: Double ) =
            -1.0 * (math.pow((actualValue - observedValue), 2.0))/ (2.0 * observationNoiseSD * observationNoiseSD ) - 0.5 * math.log(2.0 * 3.1415) - math.log(observationNoiseSD)
        observationData.foldLeft[Double] (initialLogLikelihood) {
            case (curLikelihood: Double, (idx: Int, sysvar: Int, obsVal: Double)) => {
                assert(trajectory.length > idx, s"Trajectory length: ${trajectory.length} is smaller than the index of observation $idx")
                val stateI = trajectory(idx)
                assert(stateI.length > sysvar, s"Observation refers to variable number $sysvar which is not part of the trajectory")
                val trajVal:Double = stateI(sysvar)
                curLikelihood + applySingleObservation(trajVal, obsVal)
            }
        }
    }
}



class ParameterCell(val indices: List[Int], val params: List[ModelParam]){

    val numParams: Int = params.length

    assert (indices.length == numParams, s"Fatal: Parameter cell has different number of indices (${indices.length}) than the number of parameters (${numParams})")

    override def equals(obj: Any): Boolean = {
        obj match {
            case p: ParameterCell => (indices zip p.indices) forall { case (i, j) => i == j}
            case _ => false
        }
    }

    override def hashCode(): Int = indices.hashCode()

    def computeLogPriorForCell : Double = {
        (params zip indices).map {
            case (p, j) => p.getLogPriorForSubdivision(j)
        }.sum
    }

    def isValid: Boolean = {
        (indices zip params).forall {
            case (j, p) => j >= 0 && j < p.numSubDivs
        }
    }

    def centerPoint: List[Double] = {
        (indices zip params).map {
            case (j, p) => p.getCenterForSubdivison(j)
        }
    }

    def hammingNeighbours: List[ParameterCell] = {
        (0 until numParams).foldLeft[List[ParameterCell]] (Nil) {
            case (acc, j) => {
                val pJ = params(j)
                val iJ = indices(j)
                val indices1 = indices.updated(j, pJ.nextSubdivision(iJ))
                val indices2 = indices.updated(j, pJ.previousSubdivision(iJ))
                val p1 = new ParameterCell(indices1, params)
                val p2 = new ParameterCell(indices2, params)
                p1::(p2::acc)
            }
        }
    }

    def lowerBound(idx: Int): Double = {
        require (idx >= 0 && idx < numParams)
        params(idx).getLowerBoundForSubdivision(indices(idx))
    }

    def upperBound(idx: Int): Double = {
        require (idx >= 0 && idx < numParams)
        params(idx).getUpperBoundForSubdivision(indices(idx))
    }

}

class InferenceProblem(val params: List[ModelParam], val sys: DiscreteTimeStochasticModel, val obsList: List[ObservationData]) {

    val numSteps = 1+ obsList.flatMap( _.observationData.map(_._1)).max
    val doCaching = false
    var cellLikelihoodMap: Map[ParameterCell, Double] = Map.empty

    def computeLogLikelihood(p: List[Double]) = {

        val traj = sys.simulateNSteps(numSteps, p)
        val obs = obsList.map( _.computeLogLikelihoodGaussianNoise(traj)).sum
        obs
    }

    def computeCellLogLikelihood(p: ParameterCell) = {
        if (doCaching && cellLikelihoodMap.contains(p)) {
            print("Bingo!")
            cellLikelihoodMap(p)
        } else {
            val obs = computeLogLikelihood(p.centerPoint)
            val cellPrior = p.computeLogPriorForCell
            val retVal = obs + cellPrior
            if (doCaching) {
                cellLikelihoodMap = cellLikelihoodMap + (p -> (retVal))
            }
            retVal
        }
    }

    def getRandomInitialCell: ParameterCell = {
        val r = new scala.util.Random()
        val indices = params.map ( p => {
            r.nextInt(p.numSubDivs)
        })

        new ParameterCell(indices, params)
    }

    def computeMCMCCells(numSteps: Int, burnIn: Int = 1000, period: Int = 5, numRestarts: Int = 5): List[(ParameterCell, Double)] = {
        // Let us start from some random initial parameter cell
        var cells: List[(ParameterCell, Double)] = Nil
        var currentCell: ParameterCell = this.getRandomInitialCell
        var curLogLikelihood: Double = computeCellLogLikelihood(currentCell)
        val r = new scala.util.Random()
        for (k <- (1 to numRestarts)) {
            for (j <- (1 to (burnIn + numSteps)/numRestarts)) {
                if (j > burnIn && j % period == 0) {
                    cells =  ((currentCell, curLogLikelihood))::cells
                }
                val neighbors = currentCell.hammingNeighbours
                val proposedCell = neighbors(r.nextInt(neighbors.length))
                val proposedLikelihood = computeCellLogLikelihood(proposedCell)
                val ratio = math.exp(proposedLikelihood - curLogLikelihood)
                val randVal = r.nextDouble()
                if (randVal < ratio) {
                    currentCell = proposedCell
                    curLogLikelihood = proposedLikelihood
                }

            }
            currentCell = this.getRandomInitialCell
            curLogLikelihood = computeCellLogLikelihood(currentCell)
        }
        cells
    }


}