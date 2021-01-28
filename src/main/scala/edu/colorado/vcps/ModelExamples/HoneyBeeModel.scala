package edu.colorado.vcps.ModelExamples

import java.io.{File, PrintWriter}

import StochasticModel.{DiscreteTimeStochasticModel, InferenceProblem, ModelParam, ObservationData}
import Variational.BackwardVariationalInference

object HoneyBeeModel extends App {
    /* GT beta1 = beta2 = 0.001, alpha = 0.3, gamma =0.5, delta = 0.3 */
    val p1 = new ModelParam(0,"alpha", 0.1, 0.9, 40)
    val p2 = new ModelParam(1, "gamma", 0.1, 0.9, 40)
    val p3 = new ModelParam(2, "delta", 0.1, 0.9, 40)
    val p4 = new ModelParam(3,"beta1", 0.0, 0.005, 40)
    val p5 = new ModelParam(4, "beta2", 0.0, 0.005, 40)

    val params = List(p1, p2, p3,p4, p5)
    val initCond: List[Double] = List(475, 352, 110, 30, 40)

    val sys = new  DiscreteTimeStochasticModel(initCond, params) {
        override def next(curState: List[Double], pList: List[Double]): List[Double] = {
            require(pList.length == 5 && curState.length == 5)
            assert(curState.forall(!_.isNaN), s"$pList")
            assert(pList.forall(!_.isNaN))
            val alpha = pList(0) // = p3
            val gamma = pList(1) // = p1
            val delta = pList(2) // = p2
            val beta1 = pList(3)
            val beta2 = pList(4)
            val x = curState(0)
            val y1 = curState(1)
            val y2 = curState(2)
            val z1 = curState(3)
            val z2 = curState(4)

            val xn = x + 0.05 *(- beta1 * x * y1 - beta2 * x * y2)
            val y1n = y1 + 0.05 *( beta1 * x * y1 - gamma * y1 +delta * beta1 * y1 * z1 + alpha * beta1 * y1 * z2 )
            val y2n = y2 + 0.05 * (beta2 * x * y2 - gamma * y2 + delta * beta2 * y2 * z2 + alpha * beta2 * y2 * z1)
            val z1n = z1 + 0.05 * (gamma * y1 - delta * beta1 * y1 * z1 - alpha * beta2 * y2 * z1)
            val z2n = z2 + 0.05 * (gamma * y2 - delta * beta2 * y2 * z2 - alpha * beta1 * y1 * z2)

            val newList = List(xn, y1n, y2n, z1n, z2n)
            assert(newList.forall(!_.isNaN), s"$pList, $curState, $newList")
            newList
        }
    }

    val observation1 = new ObservationData( List(
        (40, 3,  203.7),
        (40, 4, 78.2),
        (80, 3, 306.5),
        (80, 4, 101.8),
        (120, 3, 359.2),
        (120, 4, 115.2),
        (160, 3, 385.3),
        (160, 4, 124.3),
        (200, 3, 397.6),
        (200, 4, 131.7)
    ), 3)

    val infProb = new InferenceProblem(params, sys, List(observation1))
    val mcmcStartTime = System.nanoTime()
    val cells = infProb.computeMCMCCells(25000)
    val mcmcEndTime= System.nanoTime()
    val fileName = "mcmcOutput-honeybee.csv"

    println(s"MCMC finished with ${cells.size} cells collected")
    println(s"MCMC time taken: ${(mcmcEndTime-mcmcStartTime)*1E-09}")
    val fHandle = new PrintWriter(new File(fileName))
    for (elt <- cells) {
        val (cell, likelihood) = elt
        val centerPt = cell.centerPoint
        fHandle.println(s"${centerPt(0)}, ${centerPt(1)}, ${centerPt(2)}, $likelihood")
    }
    fHandle.close()


    val bwdInfer = new BackwardVariationalInference(infProb, learningRate = 0.01, batchSize = 50)
    println(" --- Simulations from the prior: --- ")
    bwdInfer.doUniformSims(500, "trajectories/honeybee-prior-")
    val q = bwdInfer.runVanillaSGD(50000)
    // // val q: List[List[Double]] = bwdInfer.runSGDWithIndependentUpdates(500)//bwdInfer.runVanillaSGD(250)
    //val q = bwdInfer.runAdamForNSteps(10000)
    bwdInfer.dumpDistributionOntoCSVFiles(q, "honeybee-")
    bwdInfer.doPosteriorSims(q, 1000, "trajectories/honeybee-")

}
