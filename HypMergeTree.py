import numpy as np 
import matplotlib.pyplot as plt

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
            for i, c in zip(range(N), color_cycle()):
                for j in range(N+1):
                    if i == j:
                        continue
                    xl = PL[i, j]
                    xr = PR[i, j]
                    if np.isinf(xl) or np.isinf(xr):
                        continue
                    r = (xr-xl)/2.0
                    xs = xl+r+XSemi[:, 0]*r
                    ys = r*XSemi[:, 1]
                    idx = np.arange(len(xs))
                    idx = idx[(xs >= 0)*(xs <= z[-1])]
                    if j == N:
                        plt.plot(xs[idx], ys[idx], color = c['color'], linestyle = '--')
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
        rs[1:-2] = (z[1:-1]-z[0:-2])*(z[2::]-z[1:-1])/(4*rInfty*(z[2::]-z[1:-1])) #r_k
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
        #Corollary 2.i (bisector between -1 and z_n)
        zs = [zn/(np.sqrt(zn-z[-2]) + i) for i in [1.0, -1.0]]
        PL[0, N-1] = min(zs)
        PR[0, N-1] = max(zs)
        #Corollary 2.ii (bisector between -1 and -1 < k < N)
        for k in range(1, N-1):
            num = z[k]*np.sqrt(z[k+1]-z[k-1])
            a = np.sqrt(z[k+1]-z[k-1])
            b = np.sqrt((z[k+1]-z[k])*(z[k]-z[k-1]))
            zs = [num/(a+b), num/(a-b)]
            PL[0, k] = min(zs)
            PR[0, k] = max(zs)
        #Corollary 2.iii (bisector between -1 < k < N and N)
        for k in range(1, N-1):
            a = np.sqrt((z[k+1]-z[k-1])*(zn-z[-2]))
            b = np.sqrt((z[k+1]-z[k])*(z[k]-z[k-1]))
            num = a*z[k] + b*zn
            zs = [num/(a+b), num/(a-b)]
            PL[k, N-1] = min(zs)
            PR[k, N-1] = max(zs)
        #Corollary 2.iv (bisector between -1 < k < N and k < j < N)
        for k in range(1, N-1):
            for j in range(k+1, N-1):
                a = np.sqrt((z[j]-z[j-1])*(z[j+1]-z[j])*(z[k+1]-z[k-1]))
                b = np.sqrt((z[k]-z[k-1])*(z[k+1]-z[k])*(z[j+1]-z[j-1]))
                num = a*z[k] + b*z[j]
                zs = [num/(a+b), num/(a-b)]
                PL[k, j] = min(zs)
                PR[k, j] = max(zs)
        #Symmetrize
        PL = np.minimum(PL, PL.T)
        PR = np.minimum(PR, PR.T)
        return (PL, PR)
        


if __name__ == '__main__':
    HMT = HypMergeTree()
    HMT.z = np.array([0, 1, 3])
    HMT.render(hRInfty = 2.0)
    plt.show()