import fenics as fe
import fenics_adjoint as adj
import numpy as np
from scipy.sparse.linalg import factorized
from sklearn.preprocessing import MinMaxScaler

from utils import map_mesh

# fe.parameters["linear_algebra_backend"] = "Eigen"


def epsilon(u):
    return fe.as_vector([
        u[0].dx(0), u[1].dx(1), u[0].dx(1) + u[1].dx(0)
    ])


def sigma(u, rho, penal=adj.Constant(3.0), E1=adj.Constant(1.0), nu=adj.Constant(1/3)):
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


def build_weakform_struct(u, du, rhoh, t, ds, subdomain_id=2):
    a = fe.inner(sigma(u,rhoh), epsilon(du))*fe.dx
    L = fe.inner(t, du)*ds(subdomain_id)
    return a, L


def displacement(u):
    return fe.sqrt(u[0]**2 + u[1]**2)


def input_assemble(rhoh, uhC, V, F, FC, v2dC,  loop, scaler=None):
    eC = epsilon(uhC)
    e_mapped = np.zeros((len(F.mesh().coordinates()), 3))
    for i in range(3):
        e_mapped[:, i] = map_mesh(
            FC.mesh().coordinates(),
            F.mesh().coordinates(),
            adj.project(eC[i], FC).vector()[v2dC]
        )
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(-1,1))
        scaler.fit(e_mapped)

    e_mapped = scaler.transform(e_mapped)
    x = np.c_[rhoh.vector(), e_mapped]
    # x /= count
    # x = np.c_[x, count]
    return x, scaler


def output_assemble(dc, v2d, loop, scalers = None,  lb = None, k = 5):
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
    return dc.vector()[v2d].reshape(-1,1), scalers, lb

