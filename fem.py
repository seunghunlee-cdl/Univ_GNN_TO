import fenics as fe
import fenics_adjoint as adj
import numpy as np
from matplotlib.tri import LinearTriInterpolator
from scipy.interpolate import LinearNDInterpolator
from scipy.sparse.linalg import factorized
from sklearn.preprocessing import MinMaxScaler

from utils import filter, map_mesh

# fe.parameters["linear_algebra_backend"] = "Eigen"


def epsilon(u):
    if u.ufl_shape[0] == 2:
        return fe.as_vector([
            u[0].dx(0), u[1].dx(1), u[0].dx(1) + u[1].dx(0)
        ])
    else:
        return fe.as_vector([
            u[0].dx(0), u[1].dx(1), u[2].dx(2), 
            u[0].dx(1) + u[1].dx(0),
            u[0].dx(2) + u[2].dx(0),
            u[1].dx(2) + u[2].dx(0)
        ])


def sigma(u, rho, penal, E1=adj.Constant(1.0), nu=adj.Constant(1/3)):
    e = epsilon(u)
    E0 = 1e-9*E1
    E = E0 + rho**penal*(E1 - E0)
    if u.ufl_shape[0] == 2:
        C = E/(1 - nu**2)*fe.as_tensor([
            [1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, (1 - nu)/2]
        ])
    else:
        C = E/(1 + nu)/(1 - 2*nu)*fe.as_tensor([
            [1.0 - nu, nu, nu, 0.0, 0.0, 0.0],
            [nu, 1.0 - nu, nu, 0.0, 0.0, 0.0],
            [nu, nu, 1.0 - nu, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, (1 - 2*nu)/2, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, (1 - 2*nu)/2, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, (1 - 2*nu)/2]
        ])
    return fe.dot(C, e)


def build_weakform_filter(rho, drho, phih, rmin):
    aH = (rmin**2*fe.inner(fe.grad(rho), fe.grad(drho)) + fe.inner(rho, drho))*fe.dx
    LH = fe.inner(phih, drho)*fe.dx
    return aH, LH


def build_weakform_struct(u, du, rhoh, t, ds, penal, subdomain_id=2):
    loadArea = adj.assemble(adj.Constant(1.0)*ds(2))
    scaledLoad = t / loadArea
    # a = fe.inner(sigma(u,rhoh,penal), epsilon(du))*dx(0) + fe.inner(sigma(u,adj.Constant(1.0),penal), epsilon(du))*dx(1)
    # L = fe.dot(scaledLoad, du)*ds(subdomain_id)
    a = fe.inner(sigma(u,rhoh,penal), epsilon(du))*fe.dx
    L = fe.dot(scaledLoad, du)*ds(subdomain_id)
    return a, L


def displacement(u):
    return fe.sqrt(u[0]**2 + u[1]**2)


def input_assemble(rhoh, uhC, V, F, FC, v2dC, center, coordsC=None, T=None, scaler=None):
    eC = epsilon(uhC)
    e_mapped = np.zeros((F.mesh().num_cells(), eC.ufl_shape[0]))
        # uhbar = adj.interpolate(uhC,V)
    # strain = epsilon(uhbar)
    # for i in range(3):
        # e_mapped[:,i]=adj.project(strain[i],F).vector()[:]
    if center.shape[1] == 2:
        for i in range(3):
            coarse_node2fine_cell = LinearTriInterpolator(T, adj.project(eC[i], FC).vector()[v2dC])
            e_mapped[:, i] = coarse_node2fine_cell(*center.T).data
    else:
        for i in range(6):
            interpolator = LinearNDInterpolator(coordsC,adj.project(eC[i], FC).vector()[v2dC])
            e_mapped[:, i] = interpolator(center).data

    if scaler is None:
        scaler = MinMaxScaler(feature_range=(-1,1))
        scaler.fit(e_mapped)

    e_mapped = scaler.transform(e_mapped)
    x = np.c_[rhoh.vector()[:], e_mapped]
    return x, scaler


def output_assemble(dc, loop, scalers = None,  lb = None, k = 5):
    box = dc.vector()[:].copy()
    if lb is None:
        q1, q3 = np.percentile(box, [25, 75])
        iqr = q3 - q1
        lb = q1 - k*iqr
    box[box < lb] = box[box >= lb].min()
    if scalers is None:
        scalers = MinMaxScaler(feature_range=(-1,0))
        scalers.fit(box.reshape(-1,1))
    box = scalers.transform(box.reshape(-1,1))
    dc.vector()[:] = box.ravel()
    return dc.vector()[:].reshape(-1,1), scalers, lb

def oc(density,dc,dv,mesh,H,Hs,volfrac,areas):
    l1 = 0
    l2 = 1e9
    move = 0.1
    while l2 - l1 > 1e-4:
        lmid = 0.5*(l2+l1)
        phi_new = np.maximum(0.0, np.maximum(density.vector()[:] - move, np.minimum(1.0, np.minimum(density.vector()[:] + move, density.vector()[:] * np.sqrt(-dc.vector()[:] / dv.vector()[:] /lmid)))))
        rho_new = filter(H,Hs,phi_new)
        l1, l2 = (lmid, l2) if (rho_new*areas).sum() - volfrac*areas.sum() > 0 else (l1, lmid)
    return phi_new