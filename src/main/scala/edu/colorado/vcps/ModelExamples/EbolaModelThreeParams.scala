package edu.colorado.vcps.ModelExamples

import java.io.{File, PrintWriter}

import StochasticModel.{DiscreteTimeStochasticModel, InferenceProblem, ModelParam, ObservationData}
import Variational.{BackwardVariationalInference, ForwardVariationalInference}

object EbolaModelThreeParams extends App {

    val p1: ModelParam = new ModelParam(0, "p1", 0.0, 1.0, 100)
    val p2: ModelParam = new ModelParam(1,"p2", 0.0, 1.0, 100)
    val p3: ModelParam = new ModelParam(2, "p3", paramLB = 0.0, paramUB = 1.0, numSubDivs = 100)
    val initCond: List[Double] = List(0.7, 0.3, 0.02, 0.02, 0.02)
    val params = List(p1, p2, p3)
    val sys = new  DiscreteTimeStochasticModel(initCond, params) {
        override def next(curState: List[Double], params: List[Double]): List[Double] = {
            require(params.length == 3 && curState.length == 5)
            val p1v = params(0)
            val p2v = params(1)
            val p3v = params(2)
            require(p1.contains(p1v) && p2.contains(p2v) && p3.contains(p3v))
            val s = curState(0)
            val e = curState(1)
            val i = curState(2)
            val r = curState(3)
            val c = curState(4)

            val sNew = s - (s * p1v * i) * 0.5
            val eNew = e + ( (s * p1v * i) - (p2v)*e) * 0.5
            val iNew = i + (p2v * e - p2v * i) * 0.5
            val rNew = r + ( p2v * i) * 0.5
            val cNew = c + p2v * e * 0.5
            List(sNew, eNew, iNew, rNew, cNew)
        }
    }

    val observation1 = new ObservationData( List(
        (5, 2, 0.11),
        (10, 2, 0.13),
        (15, 2, 0.11),
        (25, 2, 0.07),
    ), 0.01)



    val infProb = new InferenceProblem(params, sys, List(observation1))
    val cells = infProb.computeMCMCCells(25000)
    val fileName = "mcmcOutput-ebola-model.csv"
    println(s"MCMC finished with ${cells.size} cells collected")
    val fHandle = new PrintWriter(new File(fileName))
    for (elt <- cells) {
        val (cell, likelihood) = elt
        val centerPt = cell.centerPoint
        fHandle.println(s"${centerPt(0)}, ${centerPt(1)}, ${centerPt(2)}, $likelihood")
    }
    fHandle.close()

    /* val fwdInfer = new ForwardVariationalInference(infProb, learningRate = 0.00000001, batchSize = 10)
    val qFwd = fwdInfer.applyVanillaSGD(50000)
    fwdInfer.dumpDistributionOntoCSVFiles(qFwd, "fwdVariationalInference-") */

    val bwdInfer = new BackwardVariationalInference(infProb, learningRate = 0.01, batchSize = 50)
    println(" --- Simulations from the prior: --- ")
    bwdInfer.doUniformSims(1000, "trajectories/ebola-prior-")
    val q = bwdInfer.runVanillaSGD(20000)
    // // val q: List[List[Double]] = bwdInfer.runSGDWithIndependentUpdates(500)//bwdInfer.runVanillaSGD(250)
    //val q = bwdInfer.runAdamForNSteps(10000)
    bwdInfer.dumpDistributionOntoCSVFiles(q, "ebola-")
    bwdInfer.doPosteriorSims(q, 1000, "trajectories/ebola-")

}
