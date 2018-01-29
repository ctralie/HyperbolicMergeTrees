import numpy as np
import matplotlib.pyplot as plt

def equalsEps(x1, y1, x2, y2, eps = 1e-11):
    if np.abs(x1 - x2) < eps or np.abs(y1 - y2) < eps:
        return True
    return False

def acoscrop(arg):
    if arg < -1:
        return np.pi
    if arg > 1:
        return 0
    return np.arccos(arg)

def intersectLineArc(line, circle, ysline, xsarc, doPlot = False):
    ysline = np.sort(ysline)
    [a, b] = circle
    r = (b-a)/2.0
    xsarc = np.sort(xsarc)
    if doPlot:
        theta1 = acoscrop((xsarc[0]-(a+r))/r)
        theta2 = acoscrop((xsarc[1]-(a+r))/r)
        t = np.linspace(theta1, theta2, 100)
        plt.plot(a+r+r*np.cos(t), r*np.sin(t), linewidth=3)
        plt.plot([line[0], line[0]], ysline)
        plt.axis('equal')
    if not (line[0] >= xsarc[0] and line[0] <= xsarc[1]):
        return None
    y = np.sqrt(r**2-(line[0]-(a+r))**2)
    isValid = False
    if y >= ysline[0] and y <= ysline[1]:
        isValid = True
    if doPlot and isValid:
        plt.scatter(line[0], y, 40, 'k')
    if isValid:
        return (line[0], y)

def intersectArcs(end1, end2, x1, x2, doPlot = False):
    """
    :param end1: A 2 element list of endpoints for the semicircle containing\
                the first arc
    :param end2: A 2 element list of endpoints for the semicircle containing\
                the second arc
    :param x1: A 2x2 matrix, where each row is x, y of an endpoint of arc 1\
                in CCW order
    :param x2: A 2x2 matrix, where each row is x, y of an endpoint of arc 2
    :returns: (x, y) tuple of solution if it exists, or None otherwise 
    """
    #Check to see if either "arc" is a vertical line
    if np.isinf(end1[1]) and np.isfinite(end2[1]):
        return intersectLineArc(end1, end2, x1[:, 1], x2[:, 0], doPlot)
    elif np.isfinite(end1[1]) and np.isinf(end2[1]):
        return intersectLineArc(end2, end1, x2[:, 1], x1[:, 0], doPlot)
    elif np.isinf(end1[1]) and np.isinf(end2[1]):
        #Intersection of two vertical lines
        return None

    #First check to make sure the semicircles even intersect
    #(exactly one of the endpoints of one needs to be inside
    #the endpoints of the other)
    A = np.array(end1)[:, None] < np.array(end2)[None, :]
    A = np.logical_xor(A[:, 0], A[:, 1])
    if not np.logical_xor(A[0], A[1]):
        return None
    #Now check the arcs themselves
    [a, b, c, d] = [end1[0], end1[1], end2[0], end2[1]]
    r1 = (b-a)/2.0
    r2 = (d-c)/2.0
    y = None
    x = (r1**2-r2**2-(r1+a)**2+(r2+c)**2)/(2*(r2+c-r1-a))
    x1bound = [np.min(x1[:, 0]), np.max(x1[:, 0])]
    x2bound = [np.min(x2[:, 0]), np.max(x2[:, 0])]
    if x >= x1bound[0] and x <= x1bound[1] and x >= x2bound[0] and x <= x2bound[1]:
        y = np.sqrt(r2**2-(x-(c+r2))**2)
    if doPlot:
        theta1 = acoscrop((x1[0, 0]-(a+r1))/r1)
        theta2 = acoscrop((x1[1, 0]-(a+r1))/r1)
        t = np.linspace(theta1, theta2, 100)
        plt.plot(a+r1+r1*np.cos(t), r1*np.sin(t), linewidth=3)
        theta1 = acoscrop((x2[0, 0]-(c+r2))/r2)
        theta2 = acoscrop((x2[1, 0]-(c+r2))/r2)
        t = np.linspace(theta1, theta2, 100)
        plt.plot(c+r2+r2*np.cos(t), r2*np.sin(t), linewidth=3)
        t = np.linspace(0, np.pi, 100)
        plt.plot(a+r1+r1*np.cos(t), r1*np.sin(t), 'k', linestyle='--')
        plt.plot(c+r2+r2*np.cos(t), r2*np.sin(t), 'k', linestyle='--')
        if y:
            plt.scatter(x, y, 40, 'k')
        plt.axis('equal')
    if y:
        return (x, y)
    return None

def testArcIntersection():
    for i in range(400):
        end1 = np.sort(np.random.randn(2))
        end2 = np.sort(np.random.randn(2))
        plt.clf()
        x1 = np.zeros((2, 2))
        x2 = np.zeros((2, 2))
        x1[:, 0] = np.sort(np.random.rand(2)*(end1[1]-end1[0])+end1[0])
        x2[:, 0] = np.sort(np.random.rand(2)*(end2[1]-end2[0])+end2[0])
        intersectArcs(end1, end2, x1, x2, doPlot =True)
        plt.savefig("%i.svg"%i)

def testLineArcIntersection():
    for i in range(400):
        circle = np.sort(np.random.randn(2))
        xsarc = np.sort(np.random.rand(2)*(circle[1]-circle[0])+circle[0])
        line = [np.random.randn(), np.inf]
        ysline = np.sort(np.abs(np.random.randn(2)))
        plt.clf()
        intersectLineArc(line, circle, ysline, xsarc, doPlot = True)
        plt.savefig("%i.svg"%i, bbox_inches = 'tight')

if __name__ == '__main__':
    testArcIntersection()