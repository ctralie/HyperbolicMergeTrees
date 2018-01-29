import numpy as np 
import matplotlib.pyplot as plt
from GeomTools import *

def getPointsNumDenom(a, b, c, d):
    res = np.array([np.inf, np.inf])
    if np.abs(c + d) > 0:
        res[0] = (a + b) / (c + d)
    if np.abs(c - d) > 0:
        res[1] = (a - b) / (c - d)
    res = [np.min(res), np.max(res)]
    return res

def printPLPR(z, PL, PR):
    N = PL.shape[0]-1
    for i in range(N):
        s1 = "%i"%z[i]
        for j in range(i+1, N+1):
            s2 = "inf"
            if j < N:
                s2 = "%i"%z[j]
            print("%s - %s: (%g, %g)"%(s1, s2, PL[i, j], PR[i, j]))

class HypMergeTree(object):
    """
    A class for storing, rendering, and computing information about
    hyperbolic merge trees
    """
    def __init__(self):
        #Vertices (excluding point at infinity and point at zero)
        #By convention, vertex at infinity is indexed by length(z)
        self.z = np.array([])
        self.radii = np.array([])
    
    def render(self, plotVertices = True, plotBoundary = True, plotBisectors = True, drawOnly = []):
        z = self.z
        N = len(z)
        xlims = [-0.2*(z[-1]), 1.2*z[-1]]
        ylims = xlims

        #Come up with color cycling for vertices
        color_cycle = plt.rcParams['axes.prop_cycle']

        #Plot vertices
        if plotVertices:
            for i, c in zip(range(N), color_cycle()):
                plt.scatter(z[i], [0], 40, color = c['color'])
        
        #Setup circle points
        t = np.linspace(0, 2*np.pi, 100)
        XCirc = np.zeros((len(t), 2))
        XCirc[:, 0] = np.cos(t)
        XCirc[:, 1] = np.sin(t)
        t = np.linspace(0, np.pi, 50)
        XSemi = np.zeros((len(t), 2))
        XSemi[:, 0] = np.cos(t)
        XSemi[:, 1] = np.sin(t)

        rs = self.radii
        #First draw horocycle at infinity
        plt.plot([0, z[-1]], [rs[-1], rs[-1]], 'gray')
        ylims[1] = max(ylims[1], 1.1*rs[-1])
        #Now draw all other horocycles
        for i in range(N):
            plt.plot(rs[i]*XCirc[:, 0] + z[i], rs[i]*XCirc[:, 1]+rs[i], 'gray')
        
        #Plot bisectors
        if plotBisectors:
            (PL, PR) = self.getBisectorPoints()
            for i, c in zip(range(N), color_cycle()):
                for j in [N] + np.arange(N).tolist():
                    if i == j:
                        continue
                    if len(drawOnly) > 0:
                        if not(i in drawOnly or j in drawOnly):
                            continue
                    xl = PL[i, j]
                    xr = PR[i, j]
                    if np.isinf(xr):
                        #Vertical line
                        ys = np.array([0, ylims[1]])
                        xs = np.array([xl, xl])
                    else:
                        r = (xr-xl)/2.0
                        xs = xl+r+XSemi[:, 0]*r
                        ys = r*XSemi[:, 1]
                    idx = np.arange(len(xs))
                    idx = idx[(xs >= 0)*(xs <= z[-1])*(ys<=ylims[-1])]
                    if j == N:
                        plt.plot(xs[idx], ys[idx], color = c['color'], linestyle = ':', linewidth = 3)
                    elif i < j:
                        plt.plot(xs[idx], ys[idx], color = c['color'])
                    else:
                        plt.plot(xs[idx], ys[idx], color = c['color'], linestyle=':')
        if plotBoundary:
            plt.plot([0, 0], [0, ylims[1]], 'k')
            plt.plot([z[-1], z[-1]], [0, ylims[1]], 'k')
            for i in range(N-1):
                r = (z[i+1] - z[i])/2.0
                plt.plot(r*XSemi[:, 0] + z[i] + r, r*XSemi[:, 1], 'k')
        plt.axis('equal')
        return {'xlims':xlims, 'ylims':ylims}
    
    def renderVoronoiRegionsOneByOne(self, fileprefix, drawText = False):
        regions = self.getVoronoiDiagram()
        N = len(self.z)
        color_cycle = plt.rcParams['axes.prop_cycle']
        for i, c in zip(range(N), color_cycle()):
            plt.clf()
            res = self.render(drawOnly = [i])
            [xlims, ylims] = [res['xlims'], res['ylims']]
            xs = []
            ys = []
            for (arcnum, (circle, arc)) in enumerate(regions[i]):
                arc = np.array(arc)
                arc[:, 1] = np.minimum(arc[:, 1], ylims[1])
                plt.scatter(arc[:, 0], arc[:, 1], 20, color=c['color'])
                if drawText:
                    if arcnum == 0:
                        plt.text(arc[0, 0], arc[0, 1], "%i_0"%arcnum)
                    plt.text(arc[1, 0]+0.2, arc[1, 1]+0.2, "%i_1"%arcnum)
                if arc[0, 0] == arc[1, 0]:
                    #Vertical line
                    xs += arc[:, 0].tolist()
                    ys += arc[:, 1].tolist()
                else:
                    [a, b] = circle
                    r = (b-a)/2.0
                    theta1 = acoscrop((arc[0, 0]-(a+r))/r)
                    theta2 = acoscrop((arc[1, 0]-(a+r))/r)
                    t = np.linspace(theta1, theta2, 100)
                    xs += (a+r+r*np.cos(t)).tolist()
                    ys += (r*np.sin(t)).tolist()
                plt.plot(xs, ys, linewidth=2, color=c['color'])
                #plt.fill(xs, ys, color=c['color'])
            plt.savefig("%s_%i.svg"%(fileprefix, i), bbox_inches = 'tight')


    def setEqualLengthArcs(self, rInfty = None):
        if self.radii.size > 0:
            return self.radii
        if not rInfty:
            print("Error: Horocycle radii were not specified in advance")
            return None
        z = np.array(self.z, dtype = np.float64)
        rInfty = 1.0*rInfty
        if len(z) < 2:
            print("Error: Can't compute horocycle radii if fewer than two points + point at infinity")
            return None
        N = len(self.z)
        rs = np.zeros(N+1)
        rs[-1] = rInfty
        rs[0] = z[-1]/(2.0*rInfty) #r_{-1} in Francis's notes
        rs[-2] = z[-1]*(z[-1] - z[-2])/(2.0*rInfty) #r_n
        rs[1:-2] = z[-1]*(z[1:-1]-z[0:-2])*(z[2::]-z[1:-1])/(2.0*rInfty*(z[2::]-z[0:-2])) #r_k
        self.radii = rs
        return rs

    def getBisectorPoints(self):
        z = self.z
        rs = self.radii
        rsSqrt = np.sqrt(rs)
        N = len(self.z)
        #Right and left points on horizontal circle diameter
        PL = np.inf*np.ones((N+1, N+1))
        PR = np.inf*np.ones((N+1, N+1))
        #First fill in the bisectors between H_{\infty} and H_k
        rprime = np.sqrt(2)*rsSqrt[0:-1]*rsSqrt[-1]
        PL[0:-1, N] = z - rprime
        PR[0:-1, N] = z + rprime
        #Now fill in the bisectors between all other points
        for i1 in range(N):
            d = rsSqrt[i1]
            for i2 in range(i1+1, N):
                a = rsSqrt[i2]*z[i1]
                b = rsSqrt[i1]*z[i2]
                c = rsSqrt[i2]
                res = getPointsNumDenom(a, b, c, d)
                PL[i1, i2] = res[0]
                PR[i1, i2] = res[1]
        #Symmetrize
        PL = np.minimum(PL, PL.T)
        PR = np.minimum(PR, PR.T)
        return (PL, PR)
    
    def getVoronoiDiagram(self):
        z = self.z
        rs = self.radii
        (PL, PR) = self.getBisectorPoints()
        N = len(z)
        #For each vertex i, maintain a list of arcs in CCW order
        regions = []
        for i in range(N):
            #Step 1: Start with the boundary arcs (i-1, i) and (i, i+1)
            #Arc (i-1, i)
            endpts1 = np.zeros(2)
            x1 = np.zeros((2, 2))
            if i == 0:
                #Vertical halfline line boundary geodesic on left
                endpts1 = [z[0], np.inf]
                x1[:, 0] = z[0]
                x1[0, 1] = np.inf
            else:
                #Ordinary semicircle geodesic
                endpts1 = [z[i-1], z[i]]
                x1[:, 0] = endpts1
            #Arc (i, i+1)
            endpts2 = np.zeros(2)
            x2 = np.zeros((2, 2))
            if i == N-1:
                #Vertical halfline line boundary geodesic on right
                endpts2 = [z[i], np.inf]
                x2[:, 0] = z[i]
                x2[1, 1] = np.inf
            else:
                endpts2 = [z[i], z[i+1]]
                x2[:, 0] = endpts2
            #region will consist of a list of tuples 
            #   (circle x endpoints, arc 2x2 matrix endpoints)
            region = [(endpts1, x1), (endpts2, x2)]

            #Step 2: Go through bisectors one by one and narrow down region
            for j in range(N+1):
                if j == i:
                    continue
                #Setup the current bisector
                endptsbis = [PL[i, j], PR[i, j]]
                xbis = np.zeros((2, 2))
                if np.isinf(endptsbis[1]):
                    xbis[:, 0] = endptsbis[0]
                    xbis[1, 1] = np.inf
                else:
                    xbis[:, 0] = endptsbis

                #Compare with all existing arcs in CCW order
                intersections = []
                for (arcnum, (endpts, x)) in enumerate(region):
                    #print("Intersect %s with %s"%(endptsbis, endpts))
                    #print("%s\n%s"%(xbis, x))
                    res = intersectArcs(endptsbis, endpts, xbis, x)
                    if res:
                        intersections.append((arcnum, np.array([res[0], res[1]])))
                if len(intersections) == 1:
                    print("1 intersection for region %i bisector %i"%(i, j))
                    (i1, xint1) = intersections[0]
                    #Figure out if arcing left or right to determine endpoints
                    #of the new arc
                    x = np.zeros((2, 2))
                    x[0, :] = xint1
                    if endptsbis[0] < xint1[0] or np.isinf(endptsbis[1]):
                        #Arcing right or vertical
                        x[1, 0] = endptsbis[1]
                    else:
                        #Arcing left
                        x[1, 0] = endptsbis[0]
                    if i == 1:
                        print(x)
                    #If z[i] is to the right of the intersection point, march right
                    #otherwise, march left
                    if z[i] > xint1[0]:
                        x = np.flipud(x)
                        region[i1][1][0, :] = xint1
                        region = [(endptsbis, x)] + region[i1::]
                    else:
                        region[i1][1][1, :] = xint1
                        region = region[0:i1+1] + [(endptsbis, x)]
                        
                elif len(intersections) == 2:
                    print("2 intersections for region %i bisector %i"%(i, j))
                    [(i1, xint1), (i2, xint2)] = intersections
                    if xint1[0] < xint2[0]:
                        [(i2, xint2), (i1, xint1)] = [(i1, xint1), (i2, xint2)]
                    x = np.array([xint1, xint2]) #The endpoints of the new arc
                    #Now update the 2 arcs that it intersected
                    region[i1][1][1, :] = xint1
                    region[i2][1][0, :] = xint2
                    #Delete all of the arcs in between
                    newregion = []
                    k = i2
                    while not (k == i1):
                        newregion.append(region[k])
                        k = (k+1)%len(region)
                    newregion += [region[i1], (endptsbis, x)]
                    region = newregion
                elif len(intersections) > 2:
                    print("Warning: More than 2 intersections for region %i bisector %i"%(i, j))
            regions.append(region)
        return regions
            


if __name__ == '__main__':
    HMT = HypMergeTree()
    HMT.z = np.array([0, 1, 2, 4, 7])
    HMT.radii = np.array([0.7, 0.6, 0.5, 0.6, 0.45, 2.0])
    #HMT.z = np.array([0, 1, 2])
    #HMT.radii = np.array([0.5, 0.25, 0.5, 2])
    s = "0"
    for z in HMT.z[1::]:
        s += "_%g"%z
    HMT.renderVoronoiRegionsOneByOne(s, drawText = False)
    s += ".svg"
    plt.clf()
    HMT.render()
    plt.savefig(s, bbox_inches = 'tight')
