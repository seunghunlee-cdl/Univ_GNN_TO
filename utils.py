import numpy as np
from scipy.interpolate import griddata


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

