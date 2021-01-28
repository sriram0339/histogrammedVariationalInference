package ModelExamples

import java.io.{File, PrintWriter}

import StochasticModel.{DiscreteTimeStochasticModel, InferenceProblem, ModelParam, ObservationData}
import Variational.BackwardVariationalInference
import com.github.fons.nr.ode.{Factory, OdeSolverT, Solver}

import scala.util.{Failure, Success, Try}
import scala.util.Try

class OdeSolverException(s: String) extends Exception



object NeuronModelODE extends App{

    val p1: ModelParam = new ModelParam(0, "p1", 0.00, 1.0, 100)
    val p2: ModelParam = new ModelParam(1,"p2", 0.00, 1.0, 100)
    val p3: ModelParam = new ModelParam(2, "p3", 0.0, 1.0, 100)
    val initCond: List[Double] = List(-1.0, 1.0)
    val params = List(p1, p2, p3)
    def vectorField(pList: List[Double]): List[(Double, Seq[Double])=> Double] = {
        val p1 = pList(0)
        val p2 = pList(1)
        val p3 = pList(2)
        List(
            (t: Double, x: Seq[Double]) => p3 * (x(0) - math.pow(x(0), 3.0) / 3.0 + x(1)),
            (t: Double, x: Seq[Double]) => -1.0 / p3 * (x(0) - p1 + p2 * x(1))
        )
    }
    val sys = new  DiscreteTimeStochasticModel(initCond, params) {
        val stepSize = 0.1
        override def next(curState: List[Double], pList: List[Double]): List[Double] = {
            require(pList.length == 3 && curState.length == 2)
            val ode: Try[OdeSolverT] = com.github.fons.nr.ode.Factory(Solver.RK4, 0.05, (0.0, curState), vectorField(pList),   0.0001)
            ode match {
                case Success(ode1) =>{
                    ode1(stepSize) match {
                        case Success(odeRes) => { odeRes(stepSize).get }
                        case _ => throw new OdeSolverException("ODE Solver failed -- Bailing out")
                    }
                }
                case _ => throw new OdeSolverException("ODE Solver failed -- Bailing out")
            }
        }
    }

    val observation1 = new ObservationData(List(
        (20, 0, 1.1),
        (30, 0, 1.6),
        (40, 0, 0.989),
        (50, 0, -0.06),
        (60, 0, -0.99),
        (70, 0, -0.8),
        (80, 0, 0.36)
    ), 0.01)

    val infProb = new InferenceProblem(params, sys, List(observation1))
    val cells = infProb.computeMCMCCells(25000, 0, 1, 25)
    val fileName = "mcmcOutput-nagumo-ode.csv"
    println(s"MCMC finished with ${cells.size} cells collected")
    val fHandle = new PrintWriter(new File(fileName))
    for (elt <- cells) {
        val (cell, likelihood) = elt
        val centerPt = cell.centerPoint
        fHandle.println(s"${centerPt(0)}, ${centerPt(1)}, ${centerPt(2)}, $likelihood")
    }
    fHandle.close()
    println("--- Running Backward variational inference ---")
    val bwdInfer = new BackwardVariationalInference(infProb, learningRate = 1.0, batchSize = 25)
    println(" --- Simulations from the prior: --- ")
    bwdInfer.doUniformSims(500, "trajectories/neuron-ode-prior-")
    val q: List[List[Double]] = bwdInfer.runVanillaSGD(25000) //bwdInfer.runSGDWithIndependentUpdates(2000)//)
    //val q = bwdInfer.runAdamForNSteps(10000)
    bwdInfer.dumpDistributionOntoCSVFiles(q, "nagumo-ode-")
    bwdInfer.doPosteriorSims(q, 1000, "trajectories/fitzhugh-nagumo-")


}
