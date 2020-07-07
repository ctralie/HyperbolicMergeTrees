"""
Programmer: Chris Tralie
Purpose: To provide functions to help create illustrative figures
"""
import numpy as np
import matplotlib.pyplot as plt
from persim import plot_diagrams
from MergeTree import *

def loadSVGPaths(filename = "paths.svg"):
    """
    Given an SVG file, find all of the paths and load them into
    a dictionary of svg.path objects
    :param filename: Path to svg file
    :returns Paths: A dictionary of svg path objects indexed by ID
        specified in the svg file
    """
    from svg.path import parse_path
    import xml.etree.ElementTree as ET
    tree = ET.parse(filename)
    root = tree.getroot()
    Paths = {}
    for c in root.getchildren():
        if c.tag == '{http://www.w3.org/2000/svg}g':
            for elem in c.getchildren():
                if elem.tag == '{http://www.w3.org/2000/svg}path':
                    Paths[elem.attrib['id']] = parse_path(elem.attrib['d'])
    return Paths

def paramSVGPath(P, t):
    """
    Parameterize an SVG path and return it as a numpy array
    :param P: The SVG path object
    :param t: N values between [0, 1] for parameterizing path
    :returns X: An Nx2 array of path points
    """
    N = t.size
    X = np.zeros((N, 2))
    for i in range(N):
        res = P.point(t[i])
        X[i, 0] = res.real
        X[i, 1] = res.imag
    return X

def makeMergeTreeConceptFigure():
    import scipy.interpolate as interp
    path = loadSVGPaths("curve.svg")
    path = path[list(path.keys())[0]]
    N = 40
    x = paramSVGPath(path, np.linspace(0, 1, N))
    x[:, 1] *= -1
    x -= np.min(x, axis=0)[None, :]
    x[:, 1] /= np.max(x[:, 1])
    x[:, 1] += 0.2

    (MT, PS, I) = mergeTreeFrom1DTimeSeries(x[:, 1])

    critidx = [k for k in MT.keys()]
    for k in MT.keys():
        critidx += MT[k]
    critidx = np.unique(np.array(critidx))

    T = wrapMergeTreeTimeSeries(MT, PS, x)

    T.render(offset=np.array([0, 0]))
    plt.plot(x[:, 0], x[:, 1])
    plt.scatter(x[:, 0], x[:, 1])
    plt.plot(x[:, 0], 0*x[:, 0])
    plt.scatter(x[:, 0], 0*x[:, 0])

    labels = [chr(ord('a')+i) for i in range(len(critidx))]
    plt.xticks(x[critidx, 0], labels)
    plt.yticks(x[critidx, 1], ['f(%s)'%a for a in labels])
    plt.grid(linewidth=2, linestyle='--')

    plt.show()

if __name__ == '__main__':
    makeMergeTreeConceptFigure()