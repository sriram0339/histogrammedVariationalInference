package edu.colorado.vcps.ModelExamples

import java.io.{File, PrintWriter}


import StochasticModel.{DiscreteTimeStochasticModel, InferenceProblem, ModelParam, ObservationData}
import Variational.BackwardVariationalInference
import com.github.fons.nr.ode.{Factory, OdeSolverT, Solver}

import scala.util.{Failure, Success, Try}
import scala.util.Try

class OdeSolverException(s: String) extends Exception



object DallaManODE extends App {
    def createParam(pid: Int, lb: Double, ub: Double) = new ModelParam(pid, s"k${pid+1}", lb, ub, 20)

    val pRanges = List((0.05, 0.06),
        (0.45, 0.65),
         (0.01, 0.1),
         (0.01, 0.1),
         (0.01, 0.04),
        (0.2, 0.6),
         (0.2, 0.5),
        (0.001, 0.008),
         (0.001, 0.008),
         ( 0.01, 0.05),
       (0.005, 0.05),
        (0.01, 0.1),
         (0.05, 0.2),
        (0.001, 0.04),
        (0.05, 0.2),
        (0.001, 0.006),
         (0.005, 0.05),
         (0.01, 0.05),
        (0.001, 0.005),
         (0.001, 0.05))
    val params = pRanges.zipWithIndex.map {
        case ((lb, ub), j) => createParam(j, lb, ub)
    }
    val initCond = List(140, 72.43, 141.15,162.45, 268.128,3.2,5.5, 100.25,100.25, 0.0, 0.0)

    val nParams = 20
    val nStates = 11


    def split(xList: Seq[Double]): Map[String, Double] = {
        Map.empty ++ Seq(
            "Gs" -> xList(0),
            "Isc1" -> xList(1),
            "Isc2" -> xList(2),
            "Gt"-> xList(3),
            "Gp"-> xList(4),
            "Il" -> xList(5),
            "Ip" -> xList(6),
            "I1" -> xList(7),
            "Id" -> xList(8),
            "X" -> xList(9),
            "temp" -> xList(10)
        )
    }


    def fun1(pList:List[Double], x: Seq[Double]): Double = {
        def k(pid: Int) = pList(pid -1)
        val m = split(x)
        0.1*(k(2) * m("Gp") - m("Gs"))
    }

    def fun2(pList:List[Double], x: Seq[Double]): Double = {
        def k(pid: Int) = pList(pid -1)
        val m = split(x)
        -k(3) * m("Isc1")  - k(11) * m("Isc2") +0.97751710655*0.5
    }

    def fun3(pList:List[Double], x: Seq[Double]): Double = {
        def k(pid: Int) = pList(pid -1)
        val m = split(x)
          k(4) * m("Isc1") - k(11) * m("Isc2")
    }

    def fun4(pList:List[Double], x: Seq[Double]): Double = {
        def k(pid: Int) = pList(pid -1)
        val m = split(x)
        -1.0 * k(20)* (3.2267+k(12)*m("X")) * m("Gt") * ( 1 - k(19) * m("Gt") +
          (2.5097e-6) *m("Gt")*m("Gt")) + k(1) *m("Gp") - k(13) * m("Gt")
    }

    def fun5(pList:List[Double], x: Seq[Double]): Double = {
        def k(pid: Int) = pList(pid - 1)

        val m = split(x)
        3.7314 - k(14) * m("Gp") - k(5) * m("Id")  - k(1) *m("Gp") +  k(15)* m("Gt")
           + (1.140850553428184e-4)*m("temp") * m("temp")*50 + (6.134609247812877e-5)*m("temp")*50

    }

    def fun6(pList:List[Double], x: Seq[Double]): Double = {
        def k(pid: Int) = pList(pid - 1)

        val m = split(x)
         -k(6) * m("Il") + 0.2250 * m("Ip")
    }

    def fun7(pList:List[Double], x: Seq[Double]): Double = {
        def k(pid: Int) = pList(pid - 1)

        val m = split(x)
        -k(7) * m("Ip") + 0.1545 * m("Il") + k(16) * m("Isc1") + k(17) * m("Isc2")
    }

    def fun8(pList:List[Double], x: Seq[Double]): Double = {
        def k(pid: Int) = pList(pid - 1)

        val m = split(x)
        -k(8) * (m("I1") - 18.2129 * m("Ip"))
    }

    def fun9(pList:List[Double], x: Seq[Double]): Double = {
        def k(pid: Int) = pList(pid - 1)
        val m = split(x)
        -k(9) * (m("Id") - m("I1"))
    }

    def fun10(pList:List[Double], x: Seq[Double]): Double = {
        def k(pid: Int) = pList(pid - 1)
        val m = split(x)
        -k(10) * m("X") + k(18) * (18.2129 * m("Ip") - 100.25)
    }






    def vectorField(pList: List[Double]): List[(Double, Seq[Double])=> Double] = {
        List(
            (t: Double, x: Seq[Double]) => {
                fun1(pList, x)
            },
            (t: Double, x: Seq[Double]) => {
                fun2(pList, x)
            },
            (t: Double, x: Seq[Double]) => {
                fun3(pList, x)
            },
            (t: Double, x: Seq[Double]) => {
                fun4(pList, x)
            },
            (t: Double, x: Seq[Double]) => {
                fun5(pList, x)
            },
            (t: Double, x: Seq[Double]) => {
                fun6(pList, x)
            },
            (t: Double, x: Seq[Double]) => {
                fun7(pList, x)
            },
            (t: Double, x: Seq[Double]) => {
                fun8(pList, x)
            },
            (t: Double, x: Seq[Double]) => {
                fun9(pList, x)
            },
            (t: Double, x: Seq[Double]) => {
                fun10(pList, x)
            },
            (t: Double, x: Seq[Double]) => {
                1
            })
    }



    val sys = new  DiscreteTimeStochasticModel(initCond, params) {
        val stepSize = 1.0
        override def next(curState: List[Double], pList: List[Double]): List[Double] = {
            require(pList.length == nParams && curState.length == nStates)
            val ode: Try[OdeSolverT] = com.github.fons.nr.ode.Factory(Solver.EU, 0.25, (0.0, curState), vectorField(pList), 1.0)
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



    def makeObservationList: List[(Int, Int, Double)] = {
        val rawData: List[Double]= List(
            140.13356978,
        139.81134955,
        140.19058120,
        139.86570633,
        139.92910925,
        139.93849460,
        140.00840544,
        139.95933017,
        139.82630198,
        140.05131061,
        139.99384184,
        140.11340342,
        140.15254695,
        140.14024996,
        140.41420802,
        140.80468726,
        140.91487629,
        141.05955294,
        141.49414334,
        141.99564720,
        142.28324319,
        142.58556596,
        143.33791353,
        143.90767911,
        144.52184003,
        145.20151718,
        146.04936940,
        147.12001371,
        148.01954001,
        148.98611006,
        150.06043095,
        151.36472194,
        152.67104360,
        154.06343275,
        155.56660357,
        157.07168664,
        158.82048848,
        160.65421220,
        162.49222240,
        164.61657251
        )

        rawData.zipWithIndex.map { case (v: Double,j: Int) => (j, 0, v) }
    }


    val observation1 = new ObservationData(makeObservationList, 5.0)
    val infProb = new InferenceProblem(params, sys, List(observation1))
    /*val cells = infProb.computeMCMCCells(25000, 0, 1, 25)
    val fileName = "mcmcOutput-dalla-man.csv"
    println(s"MCMC finished with ${cells.size} cells collected")
    val fHandle = new PrintWriter(new File(fileName))
    for (elt <- cells) {
        val (cell, likelihood) = elt
        val centerPt = cell.centerPoint
        fHandle.println(s"${centerPt(0)}, ${centerPt(1)}, ${centerPt(2)}, $likelihood")
    }
    fHandle.close()*/

    println("--- Running Backward variational inference ---")
    val bwdInfer: BackwardVariationalInference = new BackwardVariationalInference(infProb, learningRate = 1.0, batchSize = 20)
    println(" --- Simulations from the prior: --- ")
    bwdInfer.doUniformSims(500, "trajectories/dalla-man-prior-")
    println(" --- Beginning Stochastic Gradient Descent --- ")
    val q: List[List[Double]] = bwdInfer.runVanillaSGD(50000, false) //bwdInfer.runSGDWithIndependentUpdates(2000)//)
    //val q = bwdInfer.runAdamForNSteps(10000)
    bwdInfer.dumpDistributionOntoCSVFiles(q, "dalla-man-ode")
    bwdInfer.doPosteriorSims(q, 2000, "trajectories/dalla-man-ode-")

}
