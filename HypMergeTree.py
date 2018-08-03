import numpy as np 
import matplotlib.pyplot as plt
import itertools
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
        
        #Voronoi diagram structures
        self.Ps = None
        self.vedges = None
        self.Ps2P_Vedge = None
        self.refreshNeeded = True #Whether Voronoi structure needs to be refreshed
    
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
                    #idx = idx[(xs >= 0)*(xs <= z[-1])*(ys<=ylims[-1])]
                    idx = idx[ys<=ylims[-1]]
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
        #plt.xlim(xlims)
        #plt.ylim(ylims)
        return {'xlims':xlims, 'ylims':ylims}
    
    def renderVoronoiDiagram(self, xlims = None, ylims = None, plotLabelNums = False, clipendpts = True):
        if not self.Ps or self.refreshNeeded:
            self.computeVoronoiGraph(clipendpts = clipendpts)
            self.refreshNeeded = False
        Ps = self.Ps
        Ps2P_Vedge = self.Ps2P_Vedge
        vedges = self.vedges
        N = len(self.z)
        PsLocs = np.array([p for [p, idxs] in Ps]) 
        color_cycle = plt.rcParams['axes.prop_cycle']
        colors = []
        for c in zip(range(N), color_cycle()):
            colors.append(c[1]['color'])

        res = self.render(plotVedges = False)
        if not xlims:
            [xlims, ylims] = [res['xlims'], res['ylims']]
        #First plot Voronoi points
        plt.scatter(PsLocs[:, 0], PsLocs[:, 1], 40, 'k', zorder=100)
        if plotLabelNums:
            for i in range(PsLocs.shape[0]):
                plt.text(PsLocs[i, 0]+0.01, PsLocs[i, 1]+0.01, "%s"%i)

        #Now plot voronoi segments
        for i in range(len(Ps)):
            for (j, (i1, i2)) in Ps2P_Vedge[i]:
                if i > j:
                    continue
                arc = np.array(PsLocs[[i, j], :])
                arc[:, 1] = np.minimum(arc[:, 1], ylims[1])
                [a, b] = vedges[(i1, i2)][0]
                if np.isinf(b):
                    #Vertical line
                    xs = arc[:, 0]
                    ys = arc[:, 1]
                else:
                    r = (b-a)/2.0
                    theta1 = acoscrop((arc[0, 0]-(a+r))/r)
                    theta2 = acoscrop((arc[1, 0]-(a+r))/r)
                    t = np.linspace(theta1, theta2, 100)
                    xs = a+r+r*np.cos(t)
                    ys = r*np.sin(t)
                if i2 == N:
                    plt.plot(xs, ys, linewidth=3, color=colors[i1])
                else:
                    plt.plot(xs, ys, linewidth=2, color=colors[i1])
                    plt.plot(xs, ys, color=colors[i2], linestyle=':')
        plt.ylim(ylims)
        plt.axis('equal')

    def getVedgePoints(self):
        z = self.z
        rs = self.radii
        rsSqrt = np.sqrt(rs)
        N = len(self.z)
        #Right and left points on horizontal circle diameter
        PL = np.inf*np.ones((N+1, N+1))
        PR = np.inf*np.ones((N+1, N+1))
        #First fill in the vedge between H_{\infty} and H_k
        rprime = rsSqrt[0:-1]*rsSqrt[-1]
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
    
    def getInternalVoronoiVertices(self, clipToPolygon = True, clipendpts = True):
        """An O(N^4) algorithm for internal vertices"""
        z = self.z
        rs = self.radii
        (PL, PR) = self.getVedgePoints()
        N = len(z)
        #Setup all regions for each vertex
        regions = []
        vedges = {}
        for i in range(N+1):
            region = []
            if clipToPolygon:
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
                region += [(endpts1, x1, set([-1, i])), (endpts2, x2, set([-2, i]))]

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
                    TripleVertices.append({'idxs':(i, j, k), \
                                        'ins':(ini, inj, inz), 'p':np.array(list(p))})
        
        #Check all vedge endpoints (terminal points of Voronoi diagram)

        VedgeEndpts = []
        for i in range(N):
            for j in range(i+1, N+1):
                (end, x, setij) = vedges[(i, j)]
                for k in [0, 1]:
                    p = x[k, :]
                    ini = pointInRegion(regions[i], z, i, p)
                    inj = pointInRegion(regions[j], z, j, p)
                    #For the valid terminal vertices, come up with an endpoint
                    #that's on the boundary of the polygon
                    if ini and inj and clipendpts:
                        end2 = [z[i]]
                        x2 = np.zeros((2, 2))
                        x2[0, 0] = z[i]
                        if j < N:
                            end2.append(z[j])
                            x2[1, 0] = z[j]
                        else:
                            end2.append(np.inf)
                            x2[1, 0] = z[i]
                            x2[1, 1] = np.inf
                        xres, yres = intersectArcs(end, end2, x, x2)
                        p = np.array([xres, yres])
                    VedgeEndpts.append({'idxs':(i, j), 'ins':(ini, inj), 'p':p})
        
        return {'TripleVertices':TripleVertices, 'VedgeEndpts':VedgeEndpts, 'vedges':vedges}
    
    def computeVoronoiGraph(self, eps = 1e-7, clipendpts = True):
        """
        Compute the horocycle-based Voronoi graph
        :param eps: A numerical precision tolerance level for merging 
            close internal Voronoi vertices together
        :stores {'Ps': An Nx2 array of tuples \
                        (Voronoi Vertices, involved ideal point indices), \
                        where incident vedges can be inferred from point indices}, 
                'Ps2P_Vedge': A dictionary of bi-directional vedge segments \
                    between Voronoi points: idx -> (idxother, vedge) \
                'vedges': A dictionary of (i, j) tuples to (endpts, x, set([i, j])) \
                }
        """
        res = self.getInternalVoronoiVertices(clipToPolygon = False, clipendpts = clipendpts)
        vedges = res['vedges']
        VedgeEndpts = res['VedgeEndpts']
        TripleVertices = res['TripleVertices']
        #Step 1: Come up with a pruned (point, edge) incidence structure
        #(prune for non-generic cases)
        Ps = []
        for V in TripleVertices + VedgeEndpts:
            [idxs, ins, p1] = [V['idxs'], V['ins'], V['p']]
            allInside = True
            for ini in ins:
                if not ini:
                    allInside = False
                    break
            if not allInside:
                continue
            #Check to see if this is numerically close enough
            #to one of the vertices already there
            closeToOther = False
            for i, [p2, edges] in enumerate(Ps):
                if np.sum((p1-p2)**2) < eps:
                    closeToOther = True
                    Ps[i][1].union(set(idxs))
                    break
            if not closeToOther:
                Ps.append([p1, set(idxs)])
        #Convert (point, set) array into (point, sorted list) array
        for i in range(len(Ps)):
            Ps[i][1] = sorted(list(Ps[i][1]))
        PsLocs = np.array([p for [p, idxs] in Ps]) 
        
        #Step 2: Derive vedge-point incidence structures
        Vedge2Ps = {}
        for i, (p, idxs) in enumerate(Ps):
            for (i1, i2) in itertools.combinations(idxs, 2):
                if not (i1, i2) in Vedge2Ps:
                    Vedge2Ps[(i1, i2)] = []
                Vedge2Ps[(i1, i2)].append(i)
        
        #Step 3: Sort the points along each vedge
        (PL, PR) = self.getVedgePoints()
        Ps2P_Vedge = {}
        for i in range(len(Ps)):
            Ps2P_Vedge[i] = []
        for (i1, i2) in Vedge2Ps:
            e1 = PL[i1, i2]
            e2 = PR[i1, i2]
            idxs = np.array(Vedge2Ps[(i1, i2)])
            #Compute angles
            thisPs = PsLocs[idxs, :]
            if np.isinf(e2):
                #Vertical line
                idxs = idxs[np.argsort(thisPs[:, 1])]
            else:
                #Points along arc; sort by angle
                thisPs[:, 0] = thisPs[:, 0] - e1 + (e2-e1)/2.0
                angles = np.arctan2(thisPs[:, 1], thisPs[:, 0])
                idxs = idxs[np.argsort(angles)]
            for i in range(len(idxs)-1):
                [idx1, idx2] = [idxs[i], idxs[i+1]]
                Ps2P_Vedge[idx1].append((idx2, (i1, i2)))
                Ps2P_Vedge[idx2].append((idx1, (i1, i2)))
        self.Ps = Ps
        self.Ps2P_Vedge = Ps2P_Vedge
        self.vedges = vedges
        return {'Ps':Ps, 'Ps2P_Vedge':Ps2P_Vedge, 'vedges':vedges}

def testEdgeFlip():
    HMT = HypMergeTree()
    HMT.z = np.array([0, 1, 2], dtype = np.float64)
    HMT.radii = np.array([0.25, 0.125, 0.25, 4.0])
    for i, z2 in enumerate(np.linspace(1.5, 2.5, 50)):
        HMT.z[-1] = z2
        plt.clf()
        HMT.refreshNeeded = True
        HMT.renderVoronoiDiagram()
        plt.scatter([-2, 5], [1, 1], 20, 'w')
        plt.axis('equal')
        plt.title("z2 = %.3g"%z2)
        plt.savefig("%i.png"%i, bbox_inches = 'tight')

def testQuadWeights(NSamples = 200):
    w1, w2, w4, w5 = 2.0, 1.0, 1.0, 1.0
    z1 = 3.0
    getz2 = lambda w1, w2, w3, w4, w5: z1*(w2+w3+w5)/(w1+w3+w4)
    #First do branch on right
    HMT = HypMergeTree()
    framenum = 0
    w3s = 2.0-2*np.linspace(0, 1, NSamples)
    for w3 in w3s:
        z2 = getz2(w1, w2, w3, w4, w5)
        print("z2 = %g"%z2)
        alpha_inf = w1 + w3 + w5
        alpha0 = w1 + w2
        alpha1 = w2 + w3 + w4
        alpha2 = w4 + w5
        A = z2*alpha1/(z1+z2)
        rinf = (z1+z2)/alpha_inf
        r0 = z1*alpha0
        r1 = z1*A
        r2 = z2*alpha2
        HMT.z = np.array([0, z1, z1+z2], dtype = np.float64)
        HMT.radii = np.array([r0, r1, r2, rinf])
        plt.clf()
        HMT.refreshNeeded = True
        HMT.renderVoronoiDiagram()
        plt.title("z2 = %.3g, r0 = %.3g, r1 = %.3g, r2 = %.3g\n$r_{\infty}$ = %.3g, w3 = %.3g"%(z2, r0, r1, r2, rinf, w3))
        plt.xlim([-1, 6])
        plt.ylim([0, 7])
        ax = plt.gca()
        ax.set(xlim=[-1, 6], ylim=[0, 7], aspect=1)
        plt.savefig("%i.png"%framenum, bbox_inches='tight')
        framenum += 1


if __name__ == '__main__':
    #testQuadConfigurations()
    testQuadWeights()
    """
    HMT = HypMergeTree()
    HMT.z = np.array([0, 0.75, 1.25, 2.5], dtype = np.float64)
    HMT.radii = np.array([0.4, 0.25, 0.2, 0.4, 2.0])
    HMT.renderVoronoiDiagram()
    plt.title("%s, %s"%(HMT.z, HMT.radii))
    plt.savefig("0_1_2.svg", bbox_inches = 'tight')
    """
