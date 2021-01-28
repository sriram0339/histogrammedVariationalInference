from random import uniform, gauss

def truncGaussian( mean, sigma, a, b):
    r = gauss(mean, sigma)
    while (r < a or r > b):
        r = gauss(mean, sigma)
    return r

def simulateOnce(nSteps):
    s = truncGaussian(0.7,0.02,0.6,0.8)
    e = uniform(0.2, 0.4)
    i = uniform(0.0, 0.04)
    r = uniform(0.0, 0.04)
    c = uniform(0.0, 0.04)
    for j in range(nSteps):
        sn = s - (s * 0.35 * i) * 0.5
        en = e + ( (s * 0.35 * i) - (0.28)*e) * 0.5
        ine = i + (0.28 * e - 0.29 * i) * 0.5
        rn = r + ( 0.29 * i) * 0.5
        cn = c + 0.28 * e * 0.5
        s = sn
        i = ine
        r = rn
        e = en
        c = cn
    return (i, e)

def simulateAll(nSteps):
    nSamples = 1
    p1v = 0.11
    p2v = 0.22
    p3v = 0.1
    s = 0.7
    e = 0.3
    i = 0.02
    r = 0.02
    c = 0.02
    for j in range(nSteps):
        sNew = s - (s * p1v * i) * 0.5
        eNew = e + ( (s * p1v * i) - (p2v)*e) * 0.5
        iNew = i + (p2v * e - p2v * i) * 0.5
        rNew = r + ( p2v * i) * 0.5
        cNew = c + p2v * e * 0.5
        (s, e, i, r, c) = (sNew, eNew, iNew, rNew, cNew)
    return i

if __name__ == '__main__':
    print(simulateAll(5))
    print(simulateAll(10))
    print(simulateAll(15))
    print(simulateAll(25))
    print(simulateAll(30))
