import fenics as fe
import fenics_adjoint as adj
import numpy as np
from scipy.sparse.linalg import factorized
from sklearn.preprocessing import MinMaxScaler

from utils import filter, map_mesh

# fe.parameters["linear_algebra_backend"] = "Eigen"


def epsilon(u):
    return fe.as_vector([
        u[0].dx(0), u[1].dx(1), u[0].dx(1) + u[1].dx(0)
    ])


def sigma(u, rho, penal, E1=adj.Constant(1.0), nu=adj.Constant(1/3)):
    e = epsilon(u)
    E0 = 1e-9*E1
    E = E0 + rho**penal*(E1 - E0)
    C = E/(1 - nu**2)*fe.as_tensor([
        [1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, (1 - nu)/2]
    ])
    return fe.dot(C, e)


def build_weakform_filter(rho, drho, phih, rmin):
    aH = (rmin**2*fe.inner(fe.grad(rho), fe.grad(drho)) + fe.inner(rho, drho))*fe.dx
    LH = fe.inner(phih, drho)*fe.dx
    return aH, LH


def build_weakform_struct(u, du, rhoh, t, ds, penal, subdomain_id=2):
    a = fe.inner(sigma(u,rhoh,penal), epsilon(du))*fe.dx
    L = fe.inner(t, du)*ds(subdomain_id)
    return a, L


def displacement(u):
    return fe.sqrt(u[0]**2 + u[1]**2)


def input_assemble(rhoh, uhC, V, F, FC, v2dC, loop, center, scaler=None):
    # eC = epsilon(uhC)
    uhbar = adj.interpolate(uhC,V)
    strain = epsilon(uhbar)

    e_mapped = np.zeros((F.mesh().num_cells(), 3))
    # for i in range(3):
    #     e_mapped[:, i] = map_mesh(
    #         FC.mesh().coordinates(),
    #         F.mesh().coordinates(),
    #         adj.project(eC[i], FC).vector()[v2dC]
    #     )
    for i in range(3):
        e_mapped[:,i]=adj.project(strain[i],F).vector()[:]

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

def oc(density,dc,dv,mesh,H,Hs,volfrac):
    l1 = 0
    l2 = 1e9
    move = 0.2
    while l2 - l1 > 1e-4:
        lmid = 0.5*(l2+l1)
        density_new = np.maximum(0.0, np.maximum(density.vector()[:] - move, np.minimum(1.0, np.minimum(density.vector()[:] + move, density.vector()[:] * np.sqrt(-dc.vector()[:] / dv.vector()[:] /lmid)))))
        # xphys = (H@density_new)/Hs
        l1, l2 = (lmid, l2) if sum(density_new) - volfrac * mesh.num_cells()>0 else (l1, lmid)
    return density_new