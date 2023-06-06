import random
from collections import defaultdict

import fenics as fe
import numpy as np
from fenics_adjoint import Constant
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
    return (H@x)/Hs

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
    H.setdiag(rmin)
    return H

def compute_triangle_area(triangles):
    x1, y1 = triangles[:, 0, 0], triangles[:, 0, 1]
    x2, y2 = triangles[:, 1, 0], triangles[:, 1, 1]
    x3, y3 = triangles[:, 2, 0], triangles[:, 2, 1]
    area = 0.5 * np.abs((x1*y2 + x2*y3 + x3*y1) - (y1*x2 + y2*x3 + y3*x1))
    return area

def compute_tetra_area(tetra):
    x1, y1, z1 = tetra[:, 0, 0], tetra[:, 0, 1], tetra[:, 0, 2]
    x2, y2, z2 = tetra[:, 1, 0], tetra[:, 1, 1], tetra[:, 1, 2]
    x3, y3, z3 = tetra[:, 2, 0], tetra[:, 2, 1], tetra[:, 2, 2]
    x4, y4, z4 = tetra[:, 3, 0], tetra[:, 3, 1], tetra[:, 3, 2]
    area = abs((1/6) * ((x2-x1)*(y3-y1)*(z4-z1) + (y2-y1)*(z3-z1)*(x4-x1) + (z2-z1)*(x3-x1)*(y4-y1) - (z2-z1)*(y3-y1)*(x4-x1) - (y2-y1)*(x3-x1)*(z4-z1) - (x2-x1)*(z3-z1)*(y4-y1)))
    return area

def tree_maker(center, meshC):
    tree = cKDTree(center)
    _, fcc2cn = tree.query(meshC.coordinates())
    return fcc2cn

def find_adjacent_tetrahedra(mesh):
    tdim = mesh.topology().dim()  # Topological dimension (3 for tetrahedra)
    mesh.init(tdim, tdim - 1)  # Initialize connectivity between cells and faces
    mesh.init(tdim - 1, tdim)  # Initialize connectivity between faces and cells
    num_cells = mesh.num_cells()
    adjacent_tetrahedra = defaultdict(set)

    for ci in range(num_cells):
        cell_faces = mesh.topology()(tdim, tdim - 1)(ci)  # Get the faces of the current cell
        for face_index in cell_faces:
            neighbors = mesh.topology()(tdim - 1, tdim)(face_index)  # Get the neighbors of the current face
            for neighbor in neighbors:
                if neighbor != ci:  # Exclude the current tetrahedron
                    adjacent_tetrahedra[ci].add(neighbor)

    return adjacent_tetrahedra

def create_adjacent_tetrahedra_matrix(adjacent_tetrahedra):
    num_tetrahedra = len(adjacent_tetrahedra)
    max_adjacents = max(len(adj_set) for adj_set in adjacent_tetrahedra.values())
    
    matrix = -np.ones((num_tetrahedra, max_adjacents), dtype=int)

    for tetrahedron_index, adj_set in adjacent_tetrahedra.items():
        matrix[tetrahedron_index, :len(adj_set)] = list(adj_set)

    return matrix

def line_indices(mesh, flag):
    idx = np.where(flag==True)[0]
    line_info = []
    for i in range(mesh.num_entities(1)):
        line_info.append(fe.Edge(mesh,i).entities(0).tolist())
    line_info = np.array(line_info)
    
    matching_indices = [index for index, line in enumerate(line_info) if set(line).issubset(idx)]
    return matching_indices

def dropping(part_info,x):  ####### only dropout:0
    N = part_info['elems']
    den_patch = []
    for i in range(len(N)):
        partelem = N[i]
        if all(x.vector()[partelem]==0) and (random.random()<0.9):
            den_patch.append(False)
        else:
            den_patch.append(True)
    return den_patch

def dropping2(part_info,x):  ####### both dropout:0 and 1
    N = part_info['elems']
    den_patch = []
    for i in range(len(N)):
        partelem = N[i]
        if (all(x.vector()[partelem]==0) or all(x.vector()[partelem]==1)) and (random.random()<0.9):
            den_patch.append(False)
        else:
            den_patch.append(True)
    return den_patch