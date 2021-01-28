package edu.colorado.vcps.ModelExamples

import java.io.{File, PrintWriter}

import StochasticModel.{DiscreteTimeStochasticModel, InferenceProblem, ModelParam, ObservationData}
import Variational.BackwardVariationalInference

object LaubLoomisFiveParam extends App {

    val p1 = new ModelParam(0,"p1", 0.0, 0.3, 40)
    val p2 = new ModelParam(1, "p2", 0.04, 0.2, 40)
    val p3 = new ModelParam(2, "p3", 0.22, 0.35, 40)
    val p5 = new ModelParam(3, "p5",0.0, 0.1, 40)
    val p10 = new ModelParam(paramID = 4, "p10", 0.05, 0.2, 40)
    val params = List(p1, p2, p3, p5, p10)

    val initCond: List[Double] = List(1.1, 0.95, 1.4, 2.3, 0.5, -0.1, 0.3)

    val sys = new  DiscreteTimeStochasticModel(initCond, params) {
            override def next(curState: List[Double], pList: List[Double]): List[Double] = {
                require(pList.length == 5 && curState.length == 7)
                assert(curState.forall(!_.isNaN), s"$pList")
                assert(pList.forall(!_.isNaN))

                val p1 = pList(0)
                val p2 = pList(1)
                val p3 = pList(2)
                val p5 = pList(3)
                val p10 = pList(4)

                val p6 = 0.08
                val p7 = 0.2
                val p8 = 0.13
                val p9 = 0.07
                val p11 = 0.03
                val p12 = 0.31
                val p13 = 0.18
                val p14 = 0.15

                val x1 = curState(0)
                val x2 = curState(1)
                val x3 = curState(2)
                val x4 = curState(3)
                val x5 = curState(4)
                val x6 = curState(5)
                val x7 = curState(6)



                val x1n = x1 + p1 * x3 - p2 * x1
                val x2n = x2 + p3 * x5 - p2 * x2
                val x3n = x3 + p5 * x7 - p6 * x2 * x3
                val x4n = x4 + p7 - p8 * x3 * x4
                val x5n = x5 + p9 * x1 - p10 * x4 * x5
                val x6n = x6 + p11 * x1 - p12 * x6
                val x7n = x7 + p13 * x6 - p14 * x2* x7

                val nextState = List(x1n, x2n, x3n, x4n, x5n, x6n, x7n)
                assert(nextState.forall(!_.isNaN), s"$pList, $curState, $nextState")
                nextState
            }
        }

    val observation1 = new ObservationData( List(
        (40, 0, 0.55),
        (40, 1, 0.34),
        (80, 0, 0.75),
        (80, 1, 0.20),
        (120, 0, 0.96 ),
          (120, 1, 0.5)/*,
          (160, 0, 0.75),
          (160, 1, 0.32 ),
        (200, 0, 0.88 ),
        (200, 1, 0.35)*/
    ), 0.05)

    val infProb = new InferenceProblem(params, sys, List(observation1))
    val cells = infProb.computeMCMCCells(25000)
    val fileName = "mcmcOutput-laub-loomis.csv"
    println(s"MCMC finished with ${cells.size} cells collected")
    val fHandle = new PrintWriter(new File(fileName))
    for (elt <- cells) {
        val (cell, likelihood) = elt
        val centerPt = cell.centerPoint
        fHandle.println(s"${centerPt(0)}, ${centerPt(1)}, ${centerPt(2)}, $likelihood")
    }
    fHandle.close()
    val bwdInfer = new BackwardVariationalInference(infProb, learningRate = 0.01, batchSize = 100)
    val q = bwdInfer.runVanillaSGD(50000)
    // // val q: List[List[Double]] = bwdInfer.runSGDWithIndependentUpdates(500)//bwdInfer.runVanillaSGD(250)
    //val q = bwdInfer.runAdamForNSteps(10000)
    bwdInfer.dumpDistributionOntoCSVFiles(q, "laub-loomis-")


}
