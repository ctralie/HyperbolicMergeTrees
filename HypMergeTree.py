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
    
    def render(self, plotVertices = True, plotBoundary = True, plotVedges = True, drawOnly = []):
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
        NCircPts = 500
        t = np.linspace(0, 2*np.pi, NCircPts)
        XCirc = np.zeros((len(t), 2))
        XCirc[:, 0] = np.cos(t)
        XCirc[:, 1] = np.sin(t)
        t = np.linspace(0, np.pi, NCircPts/2)
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
        
        #Plot vedges
        if plotVedges:
            (PL, PR) = self.getVedgePoints()
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
        #plt.axis('equal')
        plt.xlim(xlims)
        plt.ylim(ylims)
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
            for (arcnum, (circle, arc, idxvedge)) in enumerate(regions[i]):
                arc = np.array(arc)
                arc[:, 1] = np.minimum(arc[:, 1], ylims[1])
                plt.scatter(arc[:, 0], arc[:, 1], 20, color=c['color'])
                if drawText:
                    #if arcnum == 0:
                    plt.text(arc[0, 0]+0.1, arc[0, 1]+0.1, "%i_0"%arcnum)
                    plt.text(arc[1, 0], arc[1, 1], "%i_1"%arcnum, color = 'r')
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


    def setEqualLengthArcs(self, rInfty):
        z = np.array(self.z, dtype = np.float64)
        rInfty = 1.0*rInfty
        if len(z) < 2:
            print("Error: Can't compute horocycle radii if fewer than two points + point at infinity")
            return None
        N = len(self.z)
        rs = np.zeros(N+1)
        rs[-1] = rInfty
        rs[0] = z[-1]/rInfty #r_{-1} in Francis's notes
        rs[-2] = z[-1]*(z[-1] - z[-2])/rInfty #r_n
        rs[1:-2] = z[-1]*(z[1:-1]-z[0:-2])*(z[2::]-z[1:-1])/(rInfty*(z[2::]-z[0:-2])) #r_k
        self.radii = rs
        return rs

    def getVedgePoints(self):
        z = self.z
        rs = self.radii
        rsSqrt = np.sqrt(rs)
        N = len(self.z)
        #Right and left points on horizontal circle diameter
        PL = np.inf*np.ones((N+1, N+1))
        PR = np.inf*np.ones((N+1, N+1))
        #First fill in the vedge between H_{\infty} and H_k
        rprime = np.sqrt(2)*rsSqrt[0:-1]*rsSqrt[-1]
        PL[0:-1, N] = z - rprime
        PR[0:-1, N] = z + rprime
        #Now fill in the vedges between all other points
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
    
    def getInternalVoronoiVertices(self):
        #An O(N^4) algorithm for internal vertices
        z = self.z
        rs = self.radii
        (PL, PR) = self.getVedgePoints()
        N = len(z)
        #Setup all regions for each vertex
        regions = []
        vedges = {}
        for i in range(N+1):
            #Step 1: Start with the boundary arcs (i-1, i) and (i, i+1)
            #Arc (i-1, i)
            endpts1 = np.zeros(2)
            x1 = np.zeros((2, 2))
            if i == 0:
                #Vertical halfline boundary geodesic on left
                endpts1 = [z[0], np.inf]
                x1[:, 0] = z[0]
                x1[0, 1] = np.inf
            elif i == N:
                #Vertical halfline boundary geodesic on right
                endpts1 = [z[-1], np.inf]
                x1[:, 0] = z[-1]
                x1[0, 1] = np.inf
            else:
                #Ordinary semicircle geodesic
                endpts1 = [z[i-1], z[i]]
                x1[:, 0] = endpts1
            #Arc (i, i+1)
            endpts2 = np.zeros(2)
            x2 = np.zeros((2, 2))
            if i == N-1:
                #Vertical halfline boundary geodesic on right
                endpts2 = [z[i], np.inf]
                x2[:, 0] = z[i]
                x2[1, 1] = np.inf
            elif i == N:
                #Vertical halfline boundary geodesic on left
                endpts2 = [z[0], np.inf]
                x2[:, 0] = z[0]
                x2[0, 1] = np.inf
            else:
                #Ordinary semicircle geodesic
                endpts2 = [z[i], z[i+1]]
                x2[:, 0] = endpts2
            #region will consist of a list of tuples 
            #   (circle x endpoints, 
            #   arc 2x2 matrix endpoints, 
            #   set([index 1, index 2 of points between which vedge is formed]))
            region = [(endpts1, x1, set([-1, i])), (endpts2, x2, set([-2, i]))]

            #Step 2: Add all vedges between this vertex and all other vertices
            for j in range(N+1):
                if i == j:
                    continue
                #Setup the current vedge
                endpts = [PL[i, j], PR[i, j]]
                x = np.zeros((2, 2))
                if np.isinf(endpts[1]):
                    x[:, 0] = endpts[0]
                    x[1, 1] = np.inf
                else:
                    x[:, 0] = endpts
                vedges[(i, j)] = (endpts, x, set([i, j]))
                region += [vedges[(i, j)]]
            regions.append(region)
        
        #Check all triple wise intersections
        TripleVertices = []
        for i in range(N+1):
            for j in range(i+1, N+1):
                for k in range(j+1, N+1):
                    (end1, x1, set1) = vedges[(i, j)]
                    (end2, x2, set2) = vedges[(j, k)]
                    p = intersectArcs(end1, end2, x1, x2)
                    if not p:
                        continue
                    #Now make sure that the point is in every region
                    ini = pointInRegion(regions[i], z, i, p)
                    inj = pointInRegion(regions[j], z, j, p)
                    inz = pointInRegion(regions[k], z, k, p)
                    TripleVertices.append({'triple':(i, j, k), \
                                        'ins':(ini, inj, inz), 'p':p})
        return {'TripleVertices':TripleVertices, 'vedges':vedges}


if __name__ == '__main__':
    HMT = HypMergeTree()
    HMT.z = np.array([0, 1, 2, 4, 7])
    #HMT.radii = np.array([0.7, 0.6, 0.5, 0.55, 0.45, 2.0])
    HMT.radii = np.array([0.3, 0.4, 0.3, 0.55, 0.45, 3.0])
    #HMT.z = np.array([0, 1, 2])
    #HMT.radii = np.array([0.5, 0.25, 0.5, 2])
    s = "0"
    for z in HMT.z[1::]:
        s += "_%g"%z
    plt.title("%s, %s"%(HMT.z, HMT.radii))
    res = HMT.getInternalVoronoiVertices()
    HMT.render()
    for i, V in enumerate(res['TripleVertices']):
        [triple, ins, p] = [V['triple'], V['ins'], V['p']]
        if ins[0] and ins[1] and ins[2]:
            plt.scatter(p[0], p[1], 40, 'r', zorder=100)
        else:
            plt.scatter(p[0], p[1], 20, 'k', zorder=100)
        ts = ""
        for k in range(3):
            ts += "%i (%s) "%(triple[k], ins[k])
        #plt.title(ts)
    plt.savefig("%s.svg"%(s), bbox_inches = 'tight')
