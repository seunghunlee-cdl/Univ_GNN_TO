import fenics as fe
import fenics_adjoint as adj
from scipy.sparse.linalg import factorized


fe.parameters["linear_algebra_backend"] = "Eigen"


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