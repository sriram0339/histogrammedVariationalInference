package edu.colorado.vcps.ModelExamples

import java.io.{File, PrintWriter}

import ModelExamples.OdeSolverException
import StochasticModel.{DiscreteTimeStochasticModel, InferenceProblem, ModelParam, ObservationData}
import Variational.BackwardVariationalInference
import com.github.fons.nr.ode.{Factory, OdeSolverT, Solver}

import scala.util.{Failure, Success, Try}
import scala.util.Try



object LaubLoomisODEModel extends App {

    def createParam(pid: Int, lb: Double, ub: Double) = new ModelParam(pid, s"k${pid+1}", lb, ub, 40)
    val p1 = createParam(0, 0.0, 3.2)
    val p2 = createParam(1, 0.0, 3)
    val p3 = createParam(2, 0.0, 2.5)
    val p4 = createParam(3, 0.0, 5)
    val p5 = createParam(4, 0.0, 3.5)
    val p6 = createParam(5, 0.0, 5)
    val p7 = createParam(6, 0.0, 3.8)
    val p8 = createParam(7, 0.0, 3.8)
    val p9 = createParam(8, 0.0, 2.9)
    val p10 = createParam(9, 0.0, 2.9)
    val p11 = createParam(10, 0.0, 3.3)
    val p12 = createParam(11, 0.0, 6.4)
    val p13 = createParam(12, 0.0, 4.7)

    val params = List(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13)
    val initCond = List(1.2 ,1.0 ,1.5 ,2.4 ,1.0 ,0.1 ,0.45)
    val nParams = 13
    val nStates = 7
    def vectorField(pList: List[Double]): List[(Double, Seq[Double])=> Double] = {
        def k(i: Int) = pList(i-1)
        List(
            (t: Double, x: Seq[Double]) => { k(7) * x(2) - k(1) * x(0)},
            (t: Double, x:Seq[Double]) => {k(4) * x(4) - k(8) * x(1)},
            (t: Double, x:Seq[Double]) => {k(9) * x(6) - k(2) * x(2) * x(1)},
            (t: Double, x:Seq[Double]) => {2 - k(5) * x(3) * x(2)},
            (t: Double, x:Seq[Double]) => {k(10) * x(0) - k(11) * x(3) * x(4)},
            (t: Double, x:Seq[Double]) => { k(3) * x(0) - k(12) * x(5)},
            (t: Double, x:Seq[Double]) => { k(6) * x(5) - k(13) * x(6) * x(1)}
        )
    }

    val sys = new  DiscreteTimeStochasticModel(initCond, params) {
        val stepSize = 0.1
        override def next(curState: List[Double], pList: List[Double]): List[Double] = {
            require(pList.length == nParams && curState.length == nStates)
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
        (5, 0, 1.47203043),
        (10, 0, 1.45554573),
        (15, 0,  1.22603281),
        (20, 0, 1.02438849),
        (25, 0, 0.81610791),
        (30, 0, 0.69712263),
        (35, 0, 0.61564392),
        (40, 0, 0.56118438),
        (45, 0, 0.51421686),
        (50, 0, 0.52804347)
    ), 0.05)

    val infProb = new InferenceProblem(params, sys, List(observation1))
    val cells = infProb.computeMCMCCells(25000, 0, 1, 25)
    /*val fileName = "mcmcOutput-laub-loomis.csv"
    println(s"MCMC finished with ${cells.size} cells collected")
    val fHandle = new PrintWriter(new File(fileName))
    for (elt <- cells) {
        val (cell, likelihood) = elt
        val centerPt = cell.centerPoint
        fHandle.println(s"${centerPt(0)}, ${centerPt(1)}, ${centerPt(2)}, $likelihood")
    }
    fHandle.close()
     */
    println("--- Running Backward variational inference ---")
    val bwdInfer = new BackwardVariationalInference(infProb, learningRate = 1.0, batchSize = 25)
    println(" --- Simulations from the prior: --- ")
    bwdInfer.doUniformSims(1000, "trajectories/laub-loomis-prior-")
    val q: List[List[Double]] = bwdInfer.runVanillaSGD(25000) //bwdInfer.runSGDWithIndependentUpdates(2000)//)
    //val q = bwdInfer.runAdamForNSteps(10000)
    bwdInfer.dumpDistributionOntoCSVFiles(q, "laub-loomis-ode")
    bwdInfer.doPosteriorSims(q, 1000, "trajectories/laub-loomis-ode-")
}
