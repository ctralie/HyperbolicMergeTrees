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
    :param x1: A 2x2 matrix, where each row is x, y of an endpoint of arc 1
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

def pointInRegion(region, z, cidx, p, eps = 1e-7):
    """
    Return true if a point *p* is in a region around a point *c*
    :param region: A list of tuples 
        (circle x endpoints, 
        arc 2x2 matrix endpoints, 
        set([index 1, index 2 of points between which bisector is formed]))
    :param z: Locations of all points
    :param cidx: Index of the point at the center of this region
    :param p: 2D array of point to test
    :param eps: Some tolerance for numerical precision
    """
    for ([e1, e2], x, idxs) in region:
        if (-1 in idxs or -2 in idxs):
            if np.isinf(e2):
                if e1 == 0 and p[0] < z[0] + eps: #Left vertical boundary
                    return False
                if e1 > 0 and p[0] > e1 - eps: #Right vertical boundary
                    return False
            else:
                #Point should be above the arcs on the boundary
                cx = 0.5*(e1+e2)
                rSqr = (e2-cx)**2
                #print("idxs = ", idxs, ", cx = ", cx, ", e2 = ", e2, ", rSqr = ", rSqr)
                if (p[0]-cx)**2 + p[1]**2 < rSqr + eps:
                    return False
        elif np.isinf(e2):
            #Vertical line
            s11 = np.sign(e1-p[0]-eps)
            s12 = np.sign(e1-p[0]+eps)
            s2 = np.sign(e1-z[cidx])
            if not(s11 == s2) and not (s12 == s2):
                return False #Points not on same side
        else:
            #Arc: Point needs to lie on the correct side
            c = np.inf
            if cidx < len(z):
                c = z[cidx]
            cx = 0.5*(e1+e2)
            rSqr = (e2-cx)**2
            d1 = (p[0]-cx)**2 + p[1]**2 - rSqr
            d2 = (c-cx)**2 - rSqr
            s11 = np.sign(d1-eps)
            s12 = np.sign(d1+eps)
            s2 = np.sign(d2)
            if not(s11 == s2) and not (s12 == s2):
                return False #Points not on same side
    return True

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

def testPointInRegion():
    region = [[[0, 1], None, set([-1, 0])]]
    region.append([[1, 3], None, set([-2, 0])])
    region.append([[0, np.inf], None, set([0, 1])])
    region.append([[2.4, 5.5], None, set([0, 2])])
    region.append([[-2, 5.5], None, set([0, 3])])

    PX, PY = np.meshgrid(np.linspace(-3, 6, 100), np.linspace(0, 5, 100))
    PX = PX.flatten()
    PY = PY.flatten()
    PsInside = []
    PsOutside = []
    for i in range(PX.size):
        p = np.array([PX[i], PY[i]])
        if pointInRegion(region, 1, p):
            PsInside.append(p)
        else:
            PsOutside.append(p)
    PsInside = np.array(PsInside)
    PsOutside = np.array(PsOutside)

    ymax = 10
    t = np.linspace(0, np.pi, 100)
    xcirc = np.zeros((len(t), 2))
    xcirc[:, 0] = np.cos(t)
    xcirc[:, 1] = np.sin(t)
    for r in region:
        [x1, x2] = r[0]
        if np.isinf(x2):
            plt.plot([0, 0], [0, ymax], 'k')
        else:
            cx = 0.5*(x1+x2)
            r = x2 - cx
            plt.plot(r*xcirc[:, 0]+r+x1, r*xcirc[:, 1])
    plt.scatter(PsInside[:, 0], PsInside[:, 1], 20, 'y')
    plt.scatter(PsOutside[:, 0], PsOutside[:, 1], 20, 'g')
    plt.axis('equal')
    plt.show()

def hyperbolicArclen(end, x):
    """
    Paramters
    ---------
    end: list
        A 2 element list of endpoints for the semicircle containing the arc
    x: ndarray(2, 2)
        The endpoints of the segment, each along a row
    Returns
    -------
    len: float 
        Length of arc in the upper halfplane hyperbolic metric
    """
    if np.isinf(end[1]):
        a = x[0, 1]
        b = x[1, 1]
    else:
        gamma = lambda z: (z-end[0])/(z-end[1])
        x1c = np.complex(x[0, 0], x[0, 1])
        x2c = np.complex(x[1, 0], x[1, 1])
        a = np.imag(gamma(x1c))
        b = np.imag(gamma(x2c))
    return np.abs(np.log(a/b))

def hyperbolicArclen2(x):
    x1, x2, y1, y2 = x[0, 0], x[1, 0], x[0, 1], x[1, 1]
    arg = 1 + ((x2-x1)**2 + (y2-y1)**2)/(2*y1*y2)
    return np.arccosh(arg)

if __name__ == '__main__':
    #testArcIntersection()
    #testPointInRegion()
    e1 = 2.0
    r = 10.0
    e2 = e1 + r
    end = [e1, e2]
    t = np.pi/3
    x1 = [e1 + r/2 + (r/2)*np.cos(t), (r/2)*np.sin(t)]
    t = np.pi/5
    x2 = [e1 + r/2 + (r/2)*np.cos(t), (r/2)*np.sin(t)]
    x = np.array([x1, x2])
    res = hyperbolicArclen(end, x)
    print(res)
    print(hyperbolicArclen2(x))
