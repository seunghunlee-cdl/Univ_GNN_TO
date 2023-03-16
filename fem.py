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


def input_assemble(rhoh, uhC, V, F, FC, v2d, v2dC, count, loop, scaler=None):
    eC = epsilon(uhC)
    e_mapped = np.zeros((len(F.mesh().coordinates()), 3))
    for i in range(3):
        e_mapped[:, i] = map_mesh(
            FC.mesh().coordinates(),
            F.mesh().coordinates(),
            adj.project(eC[i], FC).vector()[v2dC]
        )
    # u_mapped = np.zeros((len(F.mesh().coordinates()), 2))
    # for i in range(2):
    #     u_mapped[:, i] = map_mesh(
    #         FC.mesh().coordinates(),
    #         F.mesh().coordinates(),
    #         adj.project(uhC[i], FC).vector()[v2dC]
    #     )
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(-1,1))
        scaler.fit(e_mapped)
    e_mapped = scaler.transform(e_mapped)
    x = np.c_[rhoh.vector()[v2d], e_mapped]
    return x, scaler


def output_assemble(dc,v2d,loop):
    box = dc.vector()[:]
    if loop == 0:
        global scalers
        scalers = MinMaxScaler(feature_range=(-1,0))
        scalers.fit(box.reshape(-1,1))
    box = scalers.transform(box.reshape(-1,1))
    dc.vector()[:] = box.ravel()
    return dc.vector()[v2d].reshape(-1,1)


def oc(density,volfrac,dc,dv,mesh):
    l1 = 0
    l2 = 1e16
    move = 0.2
    # reshape to perform vector operations
    while (l2-l1)/(l1+l2)>1e-4:
        lmid = 0.5*(l2+l1)
        density_new = np.maximum(0.0,np.maximum(density.vector()[:]-move,np.minimum(1.0,np.minimum(density.vector()[:]+move,density.vector()[:]*np.sqrt(-dc/dv.vector()[:]/lmid)))))
        l1, l2 = (lmid, l2) if (sum(density_new) - volfrac * mesh.num_entities(0))>0 else (l1, lmid)
    return density_new