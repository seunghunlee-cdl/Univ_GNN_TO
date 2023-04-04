import networkx as nx
import numpy as np
import torch
from matplotlib.tri import Triangulation
from scipy.interpolate import griddata
from scipy.spatial import KDTree, Voronoi, voronoi_plot_2d
from torch_geometric.data import Data
from torch_geometric.utils import (from_networkx, subgraph, to_networkx,
                                   to_undirected)


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
    edge = edge[(edge != -1).all(1)]
    return edge, v.vertices

def filter(H,Hs,x):
    return (H@x.vector()[:])/Hs

def graph_part_index(graph_edge, graph_coords, center, part_info):
    graph_part_info = []
    target = KDTree(graph_coords)
    _, index = target.query(center)
    for i in range(len(part_info['elems'])):
        graph_part_info.append(index[part_info['elems'][i]])
    return graph_part_info

def generate_part_graph(graph_part_info, graph_edge, graph_coords):
    graph = {'nodes':[], 'edge_index':[]}
    for i in range(len(graph_part_info)):
        subset = torch.tensor(graph_part_info[i])
        edge_index, _ = subgraph(subset, torch.tensor(graph_edge.T), return_edge_mask = False)
        edge_index = to_undirected(edge_index)
        subdata = Data(x = graph_coords[graph_part_info[i]],edge_index = edge_index)
        g = to_networkx(subdata, remove_self_loops=True).to_undirected()
        all_nodes = set(g.nodes)
        connected_nodes = set(g.nodes).difference(set(nx.isolates(g)))
        isolated_nodes = list(all_nodes.difference(connected_nodes))
        for node in isolated_nodes:
            g.remove_node(node)
        graph['nodes'].append(np.array(list(g.nodes)))
        graph['edge_index'].append(edge_index)
    return graph