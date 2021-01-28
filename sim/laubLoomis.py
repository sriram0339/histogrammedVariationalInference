from random import uniform, gauss

def truncGaussian( mean, sigma, a, b):
    r = gauss(mean, sigma)
    while (r < a or r > b):
        r = gauss(mean, sigma)
    return r

def simulateOnce(nSteps):
    x1 = 1.1
    x2 = 0.95
    x3 = 1.4
    x4 = 2.3
    x5 = 0.5
    x6 = -0.1
    x7 = 0.3
    p1 = 0.14
    p2 = 0.09
    p3 = 0.25

    p5 = 0.06
    p6 = 0.08
    p7 = 0.2
    p8 = 0.13
    p9 = 0.07
    p10 = 0.1
    p11 = 0.03
    p12 = 0.31
    p13 = 0.18
    p14 = 0.15
    for j in range(nSteps):
        x1n = x1 + p1 * x3 - p2 * x1 
        x2n = x2 + p3 * x5 - p2 * x2 
        x3n = x3 + p5 * x7 - p6 * x2 * x3 
        x4n = x4 + p7 - p8 * x3 * x4 
        x5n = x5 + p9 * x1 - p10 * x4 * x5 
        x6n = x6 + p11 * x1 - p12 * x6 
        x7n = x7 + p13 * x6 - p14 * x2* x7
        (x1, x2, x3, x4, x5, x6, x7) = (x1n, x2n, x3n, x4n, x5n, x6n, x7n)
    return (x1, x2)

if __name__ == '__main__':
    print(simulateOnce(200))
