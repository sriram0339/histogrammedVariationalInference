package edu.colorado.vcps.ModelExamples

import StochasticModel._
import Variational.BackwardVariationalInference


object NeuronModelTwoParam extends App {
    val alpha: ModelParam = new ModelParam(0, "alpha", 0.00, 2.0, 50)
    val beta: ModelParam = new ModelParam(1,"beta", 0.00, 2.0, 50)
    val initCond: List[Double] = List(-1.0, 1.0)
    val params = List(alpha, beta)
    val sys = new  DiscreteTimeStochasticModel(initCond, params) {
        override def next(curState: List[Double], params: List[Double]): List[Double] = {
            require(params.length == 2 && curState.length == 2)
            val alphaValue = params(0)
            val betaValue = params(1)
            require(alpha.contains(alphaValue) && beta.contains(betaValue))

            val x1 = curState(0)
            val x2 = curState(1)

            val newX1 = x1 + 0.01 * (x1 - math.pow(x1, 3.0)/3.0 + x2)
            val newX2 = x2 - 0.04 * (x1 - alphaValue + betaValue * x2)

            List(newX1, newX2)
        }
    }

    // Ground truth: alpha = 0.15, beta = 0.5
    val observation1 = new ObservationData(List (
        (20, 1, 1.4),
        (40, 1, 1.6),
        (60, 1, 1.55),
        (80, 1, 1.4),
        (100, 1, 1.1),
        (120, 1, 0.9),
        (140, 1, 0.5),
        (160, 1, 0.12),
        (180, 1, -0.1),
        (200, 1, -0.5),
        (220, 1, -0.4),
        (240, 1, -0.8),
        (260, 1, -0.66),
        (280, 1, -0.75),
        (300, 1, -0.74),
        (320, 1, -0.8),
        (340, 1, -0.66),
        (360, 1, -0.5),
        (380,1, -0.4),
        (400, 1, -0.3)
    ), 0.05)

    // Ground truth: alpha = 0.65, beta = 1.20
    val observation2 = new ObservationData(
        List( (20,1, 1.1),
          (40, 1, 1.25),
            ( 60, 1, 1.05),
            (80, 1,  1.2),
            (100,1,0.91),
            (120,1, 0.97),
            (140,1, 0.73),
            (160,1, 0.63),
            (180,1, 0.5),
            (200, 1, 0.42),
            (220, 1, 0.34),
            (240, 1, 0.25),
            (260, 1, 0.16),
            (280, 1, 0.0),
            (300, 1, -0.2),
            (320, 1, -0.25 ),
            (340, 1, -0.22),
            (360, 1, -0.44),
            (380, 1, -0.42),
            (400, 1, -0.56)), 0.05)

    val infProb = new InferenceProblem(params, sys, List(observation1, observation2))
   // println("-- Running Forward Variational Inference --")

 /*   val fwdInfer = new ForwardVariationalInference(infProb, learningRate = 0.1, batchSize = 40)
    val qFwd = fwdInfer.applyVanillaSGD(5000)
    fwdInfer.dumpDistributionOntoCSVFiles(qFwd, "fwdVariationalInference-")

  */

    println("--- Running Backward variational inference ---")
    val bwdInfer = new BackwardVariationalInference(infProb, learningRate = 0.1, batchSize = 25)
   val q: List[List[Double]] = bwdInfer.runVanillaSGD(20000) //bwdInfer.runSGDWithIndependentUpdates(2000)//)
   //val q = bwdInfer.runAdamForNSteps(10000)
    bwdInfer.dumpDistributionOntoCSVFiles(q, "bwdVariationalInference-")
}