import numpy as np 
import matplotlib.pyplot as plt

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
    
    def render(self, plotVertices = True, plotBoundary = True, plotBisectors = True, hRInfty = 0.0):
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

        if hRInfty > 0:
            rs = self.getHorocycleRadii(hRInfty)
            #First draw horocycle at infinity
            plt.plot([0, z[-1]], [2*rs[-1], 2*rs[-1]], 'gray')
            ylims[1] = max(ylims[1], 2.2*rs[-1])
            #Now draw all other horocycles
            for i in range(N):
                plt.plot(rs[i]*XCirc[:, 0] + z[i], rs[i]*XCirc[:, 1]+rs[i], 'gray')
        
        #Plot bisectors
        if plotBisectors:
            (PL, PR) = self.getBisectorPoints()
            printPLPR(z, PL, PR)
            for i, c in zip(range(N), color_cycle()):
                for j in [N] + np.arange(N).tolist():
                    if i == j:
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
                    idx = idx[(xs >= 0)*(xs <= z[-1])]
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
                
    def getHorocycleRadii(self, rInfty):
        z = self.z
        if len(z) < 2:
            print("Error: Can't compute horocycle radii if fewer than two points + point at infinity")
            return None
        N = len(self.z)
        rs = np.zeros(N+1)
        rs[-1] = rInfty
        rs[0] = z[-1]/(4*rInfty) #r_{-1} in Francis's notes
        rs[-2] = z[-1]*(z[-1] - z[-2])/(4*rInfty) #r_n
        rs[1:-2] = z[-1]*(z[1:-1]-z[0:-2])*(z[2::]-z[1:-1])/(4*rInfty*(z[2::]-z[0:-2])) #r_k
        return rs

    def getBisectorPoints(self):
        z = self.z
        N = len(self.z)
        #Right and left points on horizontal circle diameter
        PL = np.inf*np.ones((N+1, N+1))
        PR = np.inf*np.ones((N+1, N+1))
        #First fill in the bisectors between H_{\infty} and H_k
        #(Corollary 1 in Francis's writeup)
        zn = z[-1]
        for k in range(N):
            if k == 0:
                PL[k, N] = -np.sqrt(zn)
                PR[k, N] = -PL[k, N]
            elif k < N-1:
                rad = np.sqrt(zn*(z[k]-z[k-1])*(z[k+1]-z[k])/(z[k+1]-z[k-1]))
                PL[k, N] = z[k] - rad
                PR[k, N] = z[k] + rad
            else:
                rad = np.sqrt(zn*(zn-z[-2]))
                PL[k, N] = z[k] - rad
                PR[k, N] = z[k] + rad
        #Corollary 2 in Francis's writeup
        rsSqrt = np.sqrt(self.getHorocycleRadii(1.0))
        print(rsSqrt)
        for i in range(N):
            d = rsSqrt[i]
            for j in range(i+1, N):
                a = rsSqrt[j]*z[i]
                b = rsSqrt[i]*z[j]
                c = rsSqrt[j]
                res = getPointsNumDenom(a, b, c, d)
                PL[i, j] = res[0]
                PR[i, j] = res[1]
        #Symmetrize
        PL = np.minimum(PL, PL.T)
        PR = np.minimum(PR, PR.T)
        return (PL, PR)
        


if __name__ == '__main__':
    HMT = HypMergeTree()
    HMT.z = np.array([0, 1, 3])
    HMT.render(hRInfty = 1.0)
    s = "0"
    for z in HMT.z[1::]:
        s += "_%g"%z
    s += ".svg"
    plt.savefig(s, bbox_inches = 'tight')
