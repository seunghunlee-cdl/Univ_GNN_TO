import numpy as np
from matplotlib.tri import Triangulation
from scipy.interpolate import griddata
from scipy.spatial import Voronoi, voronoi_plot_2d


def map_mesh(src_mesh, dst_mesh, values, method: str='linear'):
    assert len(src_mesh) == len(values), \
        "Size of value mush match to either number of elements or nodes."
    
    mapped = griddata(src_mesh, values, dst_mesh, method=method)
    invalid = np.isnan(mapped).any(-1)
    if invalid.any():
        mapped[invalid] = griddata(src_mesh, values, dst_mesh[invalid], method='nearest')

    return mapped

def map_density(rhoh, rhohC, mesh, meshC, v2d, v2dC):
    rhohC.vector()[v2dC] = map_mesh(
        mesh.coordinates(),
        meshC.coordinates(),
        rhoh.vector()[v2d])

def compute_theta_error(dc, dc_pred):
    v1 = dc.vector()[:]
    v2 = dc_pred.vector()[:]
    therr = np.arccos(np.dot(v1, v2)/np.linalg.norm(v1)/np.linalg.norm(v2))*180/np.pi
    return therr

def calculate_center(mesh):
    coords = mesh.coordinates()
    trias = mesh.cells()
    T = Triangulation(*coords.T, triangles = trias)
    center = coords[trias].mean(1)
    return center

def generate_density_graph(mesh):
    coords = mesh.coordinates()
    trias = mesh.cells()
    q = np.zeros((len(trias), len(coords)), dtype = bool)
    for i, tria in enumerate(trias):
        q[i, tria] = True
    v = Voronoi(coords)
    edge = np.array(v.ridge_vertices)
    edge  = edge[(edge != -1).all(1)]
    y = v.vertices
    return edge, y