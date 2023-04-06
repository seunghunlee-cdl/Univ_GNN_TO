import numpy as np
from matplotlib.tri import Triangulation
from scipy.interpolate import griddata
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree


def map_mesh(src_mesh, dst_mesh, values, method: str='nearest'):
    assert len(src_mesh) == len(values), \
        "Size of value mush match to either number of elements or nodes."
    
    mapped = griddata(src_mesh, values, dst_mesh, method=method)
    invalid = np.isnan(mapped).any(-1)
    if invalid.any():
        mapped[invalid] = griddata(src_mesh, values, dst_mesh[invalid], method='nearest')

    return mapped

def map_density(rhoh, rhohC, mesh, meshC, v2d=None, v2dC=None):
    src_coords = mesh.coordinates()
    dst_coords = meshC.coordinates()
    if len(rhoh.vector()[:]) != mesh.coordinates().shape[0]:
        src_coords = src_coords[mesh.cells()].mean(1)
    if v2d is None:
        v2d = np.arange(src_coords.shape[0])
    rhohC.vector()[v2dC] = map_mesh(
        src_coords,
        dst_coords,
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

def filter(H,Hs,x):
    return (H@x.vector()[:])/Hs

def convert_neighors_to_edges(eid, neighbors):
    valid_neighbors = np.setdiff1d(neighbors, -1)
    return np.array([(eid, i) for i in valid_neighbors])

def convolution_operator(center, rmin):
    tree = cKDTree(center)
    pairs = np.array(list(tree.query_pairs(rmin)))
    distances, _ = tree.query(center[pairs[:,0]],k=1)
    data = rmin-distances
    num_points = len(center)
    H = coo_matrix((data, (pairs[:, 0], pairs[:, 1])), shape=(num_points, num_points))
    H = H + H.T
    return H