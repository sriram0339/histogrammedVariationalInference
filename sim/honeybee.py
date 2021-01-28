from random import uniform, gauss

def truncGaussian( mean, sigma, a, b):
    r = gauss(mean, sigma)
    while (r < a or r > b):
        r = gauss(mean, sigma)
    return r

def simulateOnce(nSteps):
    x = 475
    y1 = 352
    y2 = 110
    z1 = 30
    z2 = 40
    p1 = 0.3
    p2 = 0.5
    p3 = 0.6

    for j in range(nSteps):
        xn = x + 0.1 *(-0.001 * x * y1 - 0.001 * x * y2)
        y1n = y1 + 0.1 *( 0.001 * x * y1 - p1 * y1 + p2 * 0.001 * y1 * z1 + p3 * 0.001 * y1 * z2 )
        y2n = y2 + 0.1 * (0.001 * x * y2 - p1 * y2 + p2 * 0.001 * y2 * z2 + p3 * 0.001 * y2 * z1)
        z1n = z1 + 0.1 * (p1 * y1 - p2 * 0.001 * y1 * z1 - p3 * 0.001 * y2 * z1)
        z2n = z2 + 0.1 * (p1 * y2 - p2 * 0.001 * y2 * z2 - p3 * 0.001 * y1 * z2)
        (x,y1, y2, z1, z2) = (xn, y1n, y2n, z1n, z2n)
    return (z1, z2)

if __name__ == '__main__':
    print('20',simulateOnce(20))
    print('40', simulateOnce(40))
    print('60', simulateOnce(60))
    print('80', simulateOnce(80))
    print('100', simulateOnce(100))
