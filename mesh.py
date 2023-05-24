from time import time

import fenics as fe
import fenics_adjoint as adj
import gmsh
import meshio
import numpy as np

from utils import line_indices

TEMP_MESH_PATH = "/workspace/output/tmp_mesh_output.xdmf"


def generate_fenics_mesh(N=None):
    # Synchronize
    gmsh.model.geo.synchronize()

    # Define physical groups
    gmsh.model.addPhysicalGroup(2, [1], 1, name='domain')

    # Generate mesh
    gmsh.model.mesh.generate(2)

    # Geometric dimension
    ndim = gmsh.model.getDimension()

    # Get node coordinates
    idx, coords, _ = gmsh.model.mesh.getNodes()
    coords = coords.reshape(-1, 3)[:, :ndim]

    # Get elements
    _, tag = gmsh.model.getPhysicalGroups(ndim)[0]
    _, elementTags, elems = map(lambda x: x[0], gmsh.model.mesh.getElements(ndim, tag))
    elems = elems.reshape(-1, ndim+1) - 1
    max_node_idx = elems.ravel().max()
    coords = coords[idx-1 <= max_node_idx]

    # Partition mesh
    if N:
        tic = time()
        numElements = len(gmsh.model.mesh.getElements(2)[1][0])
        numPartitions = numElements // N
        gmsh.model.mesh.partition(numPartitions)

        part_info = {'nodes': [], 'elems': []}
        for _, tag in gmsh.model.getEntities(ndim):
            _, elementTags_, elementNodeTags = gmsh.model.mesh.getElements(ndim, tag)
            if len(elementTags_):
                part_info['nodes'].append(np.unique(elementNodeTags[0].ravel().astype(int)) - 1)
                _, comm, _ = np.intersect1d(elementTags, elementTags_, return_indices=True)
                part_info['elems'].append(comm)
        t_part_info = time()-tic
    else:
        part_info = None
        t_part_info = None

    # Generate fenics mesh from gmsh
    mesh_out = meshio.Mesh(
        points=coords,
        cells={'triangle': elems}
    )
    meshio.write(TEMP_MESH_PATH, mesh_out)

    mesh = adj.Mesh()    
    with fe.XDMFFile(TEMP_MESH_PATH) as file:
        file.read(mesh)

    return mesh, part_info, t_part_info
def generate_fenics_mesh_3d(N=None):
    gmsh.model.geo.synchronize()
    gmsh.model.occ.synchronize()
    
    gmsh.model.addPhysicalGroup(3, list(range(1, gmsh.model.geo.getMaxTag(3)+1)), 1, name='domain')
    gmsh.model.mesh.generate(3)
    ndim = gmsh.model.getDimension()
    idx, coords, _ = gmsh.model.mesh.getNodes()
    coords = coords.reshape(-1,3)[:, :ndim]
    # _, tag = gmsh.model.getPhysicalGroups(ndim)[0]
    _, elementTags, elems = map(lambda x: x[0], gmsh.model.mesh.getElements(ndim))
    elems = elems.reshape(-1,ndim+1) -1
    max_node_idx = elems.ravel().max()
    coords = coords[idx-1 <= max_node_idx]
    if N:
        tic = time()
        numElements = len(gmsh.model.mesh.getElements(3)[1][0])
        numPartitions = numElements // N
        gmsh.model.mesh.partition(numPartitions)
        part_info = {'nodes': [], 'elems': []}
        for _, tag in gmsh.model.getEntities(ndim):
            _, elementTags_, elementNodeTags = gmsh.model.mesh.getElements(ndim,tag)
            if len(elementTags_):
                part_info['nodes'].append(np.unique(elementNodeTags[0].ravel().astype(int)) - 1)
                _, comm, _ = np.intersect1d(elementTags, elementTags_, return_indices=True)
                part_info['elems'].append(comm)
        t_part_info = time()-tic
    else:
        part_info = None
        t_part_info = None
    
    mesh_out = meshio.Mesh(
        points = coords,
        cells = {"tetra": elems}
    )
    meshio.write(TEMP_MESH_PATH, mesh_out)
    mesh = adj.Mesh()
    with fe.XDMFFile(TEMP_MESH_PATH) as file:
        file.read(mesh)
    return mesh, part_info, t_part_info

def get_clever2d_mesh(L=2, H=1, hmax=0.1, N=None):
    # Initialize
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add('clever2d')

    # Add points
    gmsh.model.geo.addPoint(0, 0, 0, hmax, 1)
    gmsh.model.geo.addPoint(L, 0, 0, hmax, 2)
    gmsh.model.geo.addPoint(L, 0.45*H, 0, hmax, 3)
    gmsh.model.geo.addPoint(L, 0.55*H, 0, hmax, 4)
    gmsh.model.geo.addPoint(L, H, 0, hmax, 5)
    gmsh.model.geo.addPoint(0, H, 0, hmax, 6)

    # Add lines
    gmsh.model.geo.addLine(1, 2, 1)
    gmsh.model.geo.addLine(2, 3, 2)
    gmsh.model.geo.addLine(3, 4, 3)
    gmsh.model.geo.addLine(4, 5, 4)
    gmsh.model.geo.addLine(5, 6, 5)
    gmsh.model.geo.addLine(6, 1, 6)

    # Add surfaces
    gmsh.model.geo.addCurveLoop([1, 2, 3, 4, 5, 6], 1)
    gmsh.model.geo.addPlaneSurface([1], 1)
    
    # Convert gmsh to fenics mesh
    mesh, part_info, t_part_info = generate_fenics_mesh(N)
    gmsh.finalize()

    # Function spaces
    if N:
        V = fe.VectorFunctionSpace(mesh,"CG", 1)
        F = fe.FunctionSpace(mesh,"DG", 0)
    else:
        V = fe.VectorFunctionSpace(mesh, "CG", 1)
        F = fe.FunctionSpace(mesh, "CG", 1)

    u = fe.TrialFunction(V)
    du = fe.TestFunction(V)

    rho = fe.TrialFunction(F)
    drho = fe.TestFunction(F)

    # Boundary marker
    boundaries = fe.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(0)

    # Dirichlet boundary
    class DirBd(fe.SubDomain):
        def inside(self, x, on_boundary):
            return (fe.near(x[0], 0.0) and on_boundary)
    dirBd = DirBd()
    dirBd.mark(boundaries, 1)
    bcs = [adj.DirichletBC(V, (0.0,0.0), dirBd)]

    # Traction boundary
    class TracBd(fe.SubDomain):
        def inside(self, x, on_boundary):
            return ((fe.near(x[0], L) and (x[1] >= 0.45*H and x[1] <= 0.55*H)) and on_boundary)
    tracBd = TracBd()
    tracBd.mark(boundaries, 2)
    t = adj.Constant((0.0, -1.0))
    ds = fe.Measure("ds")(mesh, subdomain_data=boundaries)

    return mesh, V, F, bcs, t, ds, u, du, rho, drho, part_info, t_part_info
    
def get_clever3d_mesh(L = 2, H = 1, W = 0.5, hmax = 0.1, N=None):
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add('mbb3d')
    T = 0.01
    
    # gmsh.model.geo.addPoint(L, 0, 0, hmax, 1)
    # gmsh.model.geo.addPoint(L, T, 0, hmax, 2)
    # gmsh.model.geo.addPoint(L-T, T, 0, hmax, 3)
    # gmsh.model.geo.addPoint(L-T, 0, 0, hmax, 4)
    # gmsh.model.geo.addPoint(L, H, 0, hmax, 5)
    # gmsh.model.geo.addPoint(0, H, 0, hmax, 6)
    # gmsh.model.geo.addPoint(0, 0, 0, hmax, 7)

    # gmsh.model.geo.addLine(1, 2, 1)
    # gmsh.model.geo.addLine(2, 3, 2)
    # gmsh.model.geo.addLine(3, 4, 3)
    # gmsh.model.geo.addLine(4, 1, 4)
    # gmsh.model.geo.addLine(2, 5, 5)
    # gmsh.model.geo.addLine(5, 6, 6)
    # gmsh.model.geo.addLine(6, 7, 7)
    # gmsh.model.geo.addLine(7, 4, 8)

    # gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 1)
    # gmsh.model.geo.addCurveLoop([5, 6, 7, 8, -3, -2], 2)
    
    # gmsh.model.geo.addPlaneSurface([1], 1)
    # gmsh.model.geo.addPlaneSurface([2], 2)

    # gmsh.model.geo.extrude([(2, 1)], 0, 0, W)
    # gmsh.model.geo.extrude([(2, 2)], 0, 0, W)
    gmsh.model.geo.addPoint(0, 0, 0, hmax, 1)
    gmsh.model.geo.addPoint(0, 0, W, hmax, 2)
    gmsh.model.geo.addPoint(L, 0, W, hmax, 3)
    gmsh.model.geo.addPoint(L, 0, 0, hmax, 4)


    gmsh.model.geo.addLine(1, 2, 1)
    gmsh.model.geo.addLine(2, 3, 2)
    gmsh.model.geo.addLine(3, 4, 3)
    gmsh.model.geo.addLine(4, 1, 4)
    gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 1)
    gmsh.model.geo.addPlaneSurface([1], 1)
    gmsh.model.geo.extrude([(2, 1)], 0, H, 0)
    mesh, part_info, t_part_info = generate_fenics_mesh_3d(N)
    gmsh.finalize()
    

    if N:
        V = fe.VectorFunctionSpace(mesh, "CG", 1)
        F = fe.FunctionSpace(mesh, "DG", 0)
    else:
        V = fe.VectorFunctionSpace(mesh, "CG", 1)
        F = fe.FunctionSpace(mesh, "CG", 1)
    
    u = fe.TrialFunction(V)
    du = fe.TestFunction(V)

    rho = fe.TrialFunction(F)
    drho = fe.TestFunction(F)

    domains = fe.MeshFunction("size_t", mesh, mesh.topology().dim())
    domains.set_all(0)
    boundaries = fe.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(0)

    tol = 1E-15
    class DirBdSupp(fe.SubDomain):
        def inside(self, x, on_boundary):
            return (x[0] <= tol) and on_boundary
    class DirBdSym(fe.SubDomain):
        def inside(self, x, on_boundary):
            return (x[2] <= tol) and on_boundary
        
    dirBdSupp, dirBdSym = [DirBdSupp(), DirBdSym()]
    # dirBdSupp = DirBdSupp()
    dirBdSym.mark(boundaries, 1)
    dirBdSupp.mark(boundaries, 1)
    bdsupp = adj.DirichletBC(V, adj.Constant((0.0, 0.0, 0.0)), dirBdSupp)
    bdsym = adj.DirichletBC(V.sub(2), adj.Constant(0.0), dirBdSym)
    bcs = [bdsupp, bdsym]
    # bcs = [bdsupp]
    
    class TracBd(fe.SubDomain):
        def inside(self, x, on_boundary):
            return (x[0]>=L-tol) and (x[1]<=hmax+tol) and on_boundary  ## side
            # return (x[0]>=L-(hmax+tol)) and (x[1]<=tol) and on_boundary ## bottom

    tracBd = TracBd()
    tracBd.mark(boundaries,2)

    # flag1 = np.logical_and(mesh.coordinates()[:, 0] >= 2, mesh.coordinates()[:, 1] <= 0)
    # flag2 = np.logical_or(mesh.coordinates()[:, 2] <= 0, mesh.coordinates()[:, 2] >= 0.5)
    # flag3 = np.logical_and(flag1, flag2)

    # flag = np.logical_and(flag1, np.logical_not(flag2))
    # line_index = line_indices(mesh, flag)
    # edge_function.array()[line_index] = 2
    t = adj.Constant((0.0,-1.0,0.0))
    ds = fe.Measure("ds", domain = mesh, subdomain_data=boundaries)
    
    return mesh, V, F, bcs, t, ds, u, du, rho, drho, part_info, t_part_info
def get_mbb2d_mesh(L=3, H=1, hmax=0.1, N=None):
    # Initialize
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add('mbb2d')

    # Add points
    gmsh.model.geo.addPoint(0, 0, 0, hmax, 1)
    gmsh.model.geo.addPoint(L, 0, 0, hmax, 2)
    gmsh.model.geo.addPoint(L, H, 0, hmax, 3)
    gmsh.model.geo.addPoint(0, H, 0, hmax, 4)

    # Add lines
    gmsh.model.geo.addLine(1, 2, 1)
    gmsh.model.geo.addLine(2, 3, 2)
    gmsh.model.geo.addLine(3, 4, 3)
    gmsh.model.geo.addLine(4, 1, 4)

    # Add surfaces
    gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 1)
    gmsh.model.geo.addPlaneSurface([1], 1)

    # Synchronize
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)
    
    # Convert gmsh to fenics mesh
    mesh, part_info,t_part_info = generate_fenics_mesh(N)

    gmsh.finalize()

    # Function spaces
    if N:
        V = fe.VectorFunctionSpace(mesh,"CG", 1)
        F = fe.FunctionSpace(mesh,"DG", 0)
    else:
        V = fe.VectorFunctionSpace(mesh, "CG", 1)
        F = fe.FunctionSpace(mesh, "CG", 1)

    u = fe.TrialFunction(V)
    du = fe.TestFunction(V)
    
    rho = fe.TrialFunction(F)
    drho = fe.TestFunction(F)

    # Boundary marker
    boundaries = fe.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(0)
    # Dirichlet boundary
    tol = 1E-5
    class DirBdSym(fe.SubDomain):
        def inside(self, x, on_boundary):
            return fe.near(x[0], 0.0) 
            # return (x[0]<tol) and on_boundary
    class DirBdSupp(fe.SubDomain):
        def inside(self, x, on_boundary):
            # return (fe.near(x[1], 0.0) and x[0] >= L - 0.05*H) and on_boundary
            return abs(x[0]-L) < 1e-14 and abs(x[1]) < 1e-14 
            # return (x[0]>(L-tol)) and (x[1]<tol) and on_boundary
    dirBdSym, dirBdSupp = [DirBdSym(),DirBdSupp()]
    # dirBdSupp = DirBdSupp()
    dirBdSym.mark(boundaries, 1)
    dirBdSupp.mark(boundaries, 1)
    # bcs = [adj.DirichletBC(V.sub(0), 0.0, dirBdSym), adj.DirichletBC(V.sub(1), 0.0, dirBdSupp)]
    bcs = [adj.DirichletBC(V.sub(0), 0.0, dirBdSym), adj.DirichletBC(V.sub(1), 0.0, dirBdSupp,method = 'pointwise')]
    # Traction boundary
    class TracBd(fe.SubDomain):
        def inside(self, x, on_boundary):
            return (fe.near(x[1], H) and (x[0] <= hmax*1.5) and on_boundary)
            # return (x[1]>=(H-tol)) and (x[0]<=hmax+tol) and on_boundary
    tracBd = TracBd()
    tracBd.mark(boundaries, 2)
    t = adj.Constant((0.0, -10.0))
    ds = fe.Measure("ds")(mesh, subdomain_data=boundaries)
    return mesh, V, F, bcs, t, ds, u, du, rho, drho, part_info, t_part_info


def get_mbb3d_mesh(L=3, H=1, W=0.5, hmax=0.1, N=None):
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add('mbb3d')
    T = 0.01
    # add points
    # gmsh.model.geo.addPoint(0, H, 0, hmax, 1)
    # gmsh.model.geo.addPoint(0, H, T, hmax, 2)
    # gmsh.model.geo.addPoint(T, H, T, hmax, 3)
    # gmsh.model.geo.addPoint(T, H, 0, hmax, 4)
    # gmsh.model.geo.addPoint(0, H, W, hmax, 5)
    # gmsh.model.geo.addPoint(L, H, W, hmax, 6)
    # gmsh.model.geo.addPoint(L, H, 0, hmax, 7)

    # gmsh.model.geo.addPoint(L-T, hmax, W, hmax, 8)
    # gmsh.model.geo.addPoint(L-T, 0, W, hmax, 9)
    # gmsh.model.geo.addPoint(L, 0, W, hmax, 10)
    # gmsh.model.geo.addPoint(L, hmax, W, hmax, 11)
    # gmsh.model.geo.addPoint(0, hmax, W, hmax, 12)
    # gmsh.model.geo.addPoint(0, 0, W, hmax, 13)

    # gmsh.model.geo.addLine(1, 2, 1)
    # gmsh.model.geo.addLine(2, 3, 2)
    # gmsh.model.geo.addLine(3, 4, 3)
    # gmsh.model.geo.addLine(4, 1, 4)
    # gmsh.model.geo.addLine(2, 5, 5)
    # gmsh.model.geo.addLine(5, 6, 6)
    # gmsh.model.geo.addLine(6, 7, 7)
    # gmsh.model.geo.addLine(7, 4, 8)

    # gmsh.model.geo.addLine(8, 9, 9)
    # gmsh.model.geo.addLine(9, 10, 10)
    # gmsh.model.geo.addLine(10, 11, 11)
    # gmsh.model.geo.addLine(11, 8, 12)
    # gmsh.model.geo.addLine(8, 12, 13)
    # gmsh.model.geo.addLine(12, 13, 14)
    # gmsh.model.geo.addLine(13, 9, 15)

    # gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 1)
    # gmsh.model.geo.addCurveLoop([5, 6, 7, 8, -3, -2], 2)

    # gmsh.model.geo.addCurveLoop([9, 10, 11, 12], 3)
    # gmsh.model.geo.addCurveLoop([13, 14, 15, -9], 4)

    # gmsh.model.geo.addPlaneSurface([1], 1)
    # gmsh.model.geo.addPlaneSurface([2], 2)
    # gmsh.model.geo.addPlaneSurface([3], 3)
    # gmsh.model.geo.addPlaneSurface([4], 4)

    # gmsh.model.geo.extrude([(2, 1)], 0, -H, 0)
    # gmsh.model.geo.extrude([(2, 2)], 0, -H, 0)
    # gmsh.model.geo.extrude([(2, 3)], 0, 0, -W)
    # gmsh.model.geo.extrude([(2, 4)], 0, 0, -W)

    # gmsh.model.geo.addPoint(0, 0, 0, hmax, 1)
    # gmsh.model.geo.addPoint(0, 0, T, hmax, 2)
    # gmsh.model.geo.addPoint(T, 0, T, hmax, 3)
    # gmsh.model.geo.addPoint(T, 0, 0, hmax, 4)
    # gmsh.model.geo.addPoint(L-T, 0, W-T, hmax, 5)
    # gmsh.model.geo.addPoint(L-T, 0, W, hmax, 6)
    # gmsh.model.geo.addPoint(L, 0, W, hmax, 7)
    # gmsh.model.geo.addPoint(L, 0, W-T, hmax, 8)
    # gmsh.model.geo.addPoint(0, 0, W, hmax, 9)
    # gmsh.model.geo.addPoint(L, 0, 0, hmax, 10)

    # gmsh.model.geo.addLine(1, 2, 1)
    # gmsh.model.geo.addLine(2, 3, 2)
    # gmsh.model.geo.addLine(3, 4, 3)
    # gmsh.model.geo.addLine(4, 1, 4)
    # gmsh.model.geo.addLine(5, 6, 5)
    # gmsh.model.geo.addLine(6, 7, 6)
    # gmsh.model.geo.addLine(7, 8, 7)
    # gmsh.model.geo.addLine(8, 5, 8)
    # gmsh.model.geo.addLine(2, 9, 9)
    # gmsh.model.geo.addLine(9, 6, 10)
    # gmsh.model.geo.addLine(8, 10, 11)
    # gmsh.model.geo.addLine(10, 4, 12)

    # gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 1)
    # gmsh.model.geo.addCurveLoop([5, 6, 7, 8], 2)
    # gmsh.model.geo.addCurveLoop([9, 10, -5, -8, 11, 12, -3, -2], 3)
    
    # gmsh.model.geo.addPlaneSurface([1], 1)
    # gmsh.model.geo.addPlaneSurface([2], 2)
    # gmsh.model.geo.addPlaneSurface([3], 3)
    # gmsh.model.geo.extrude([(2, 1)], 0, H, 0)
    # gmsh.model.geo.extrude([(2, 2)], 0, H, 0)
    # gmsh.model.geo.extrude([(2, 3)], 0, H, 0)
    gmsh.model.geo.addPoint(0, 0, 0, hmax, 1)
    gmsh.model.geo.addPoint(0, 0, W, hmax, 2)
    gmsh.model.geo.addPoint(L, 0, W, hmax, 3)
    gmsh.model.geo.addPoint(L, 0, 0, hmax, 4)


    gmsh.model.geo.addLine(1, 2, 1)
    gmsh.model.geo.addLine(2, 3, 2)
    gmsh.model.geo.addLine(3, 4, 3)
    gmsh.model.geo.addLine(4, 1, 4)
    gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 1)
    gmsh.model.geo.addPlaneSurface([1], 1)
    gmsh.model.geo.extrude([(2, 1)], 0, H, 0)

    mesh, part_info, t_part_info = generate_fenics_mesh_3d(N)
    gmsh.finalize()

    if N:
        V = fe.VectorFunctionSpace(mesh, "CG", 1)
        F = fe.FunctionSpace(mesh, "DG", 0)
    else:
        V = fe.VectorFunctionSpace(mesh, "CG", 1)
        F = fe.FunctionSpace(mesh, "CG", 1)
    
    u = fe.TrialFunction(V)
    du = fe.TestFunction(V)
    
    rho = fe.TrialFunction(F)
    drho = fe.TestFunction(F)

    boundaries = fe.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(0)
    tol = 1E-5
    class DirBdX(fe.SubDomain):
        def inside(self, x, on_boundary):
            return (x[0] <= tol) and on_boundary
    class DirBdY(fe.SubDomain):
        def inside(self, x, on_boundary):
            # return (x[0] >= L-(T+tol)) and (x[1] <= tol) and (x[2] <= W-(T+tol)) and on_boundary
            # return (x[0] >= L-T-tol) and (x[1] <= tol) and (x[2] >= W-T-tol) and on_boundary
            return (x[0] >= L-(hmax+tol)) and (x[1] <= tol) and (x[2] >= W-(hmax*1.5+tol)) and on_boundary
    class DirBdZ(fe.SubDomain):
        def inside(self, x, on_boundary):
            return (x[2] <= tol) and on_boundary
        
    dirBdX, dirBdY, dirBdZ = [DirBdX(), DirBdY(), DirBdZ()]
    dirBdX.mark(boundaries, 1)
    dirBdY.mark(boundaries, 1)
    dirBdZ.mark(boundaries, 1)
    dbX = adj.DirichletBC(V.sub(0), adj.Constant(0.0), dirBdX)
    dbY = adj.DirichletBC(V.sub(1), adj.Constant(0.0), dirBdY)
    dbZ = adj.DirichletBC(V.sub(2), adj.Constant(0.0), dirBdZ)
    bcs = [dbX, dbY, dbZ]

    class TracBd(fe.SubDomain):
        def inside(self, x, on_boundary):
            return (x[0] <= hmax+tol) and (x[2] <= hmax+tol) and (x[1] >= H-tol) and on_boundary
        
    tracBd = TracBd()
    tracBd.mark(boundaries, 2)
    t = adj.Constant((0.0, -1.0, 0.0))
    ds = fe.Measure("ds", domain = mesh, subdomain_data = boundaries)
    return mesh, V, F, bcs, t, ds, u, du, rho, drho, part_info, t_part_info

def get_wrench2d_mesh(L: float = 2, R1: float = 0.5, R2: float = 0.3, r1: float = 0.3, r2: float = 0.175, hmax: float = 0.1, N = None):
    # Initialize
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add('wrench2d')

    # Add points
    x1 = R1*(R1 - R2)/L
    x2 = (L**2 - R2**2 + R1*R2)/L
    y1 = R1*np.sqrt((L + R1 - R2)*(L - R1 + R2))/L
    y2 = R2*np.sqrt((L + R1 - R2)*(L - R1 + R2))/L
    gmsh.model.geo.addPoint(0, 0, 0, hmax, 1)
    gmsh.model.geo.addPoint(r1, 0, 0, hmax, 2)
    gmsh.model.geo.addPoint(-r1, 0, 0, hmax, 3)
    gmsh.model.geo.addPoint(L, 0, 0, hmax, 4)
    gmsh.model.geo.addPoint(L + r2, 0, 0, hmax, 5)
    gmsh.model.geo.addPoint(L - r2, 0, 0, hmax, 6)
    gmsh.model.geo.addPoint(x1, -y1, 0, hmax, 7)
    gmsh.model.geo.addPoint(x2, -y2, 0, hmax, 8)
    gmsh.model.geo.addPoint(x2, y2, 0, hmax, 9)
    gmsh.model.geo.addPoint(x1, y1, 0, hmax, 10)
    gmsh.model.geo.addPoint(-R1, 0, 0, hmax, 11)
    
    # Add lines
    gmsh.model.geo.addCircleArc(2, 1, 3, 1)
    gmsh.model.geo.addCircleArc(3, 1, 2, 2)
    gmsh.model.geo.addCircleArc(5, 4, 6, 3)
    gmsh.model.geo.addCircleArc(6, 4, 5, 4)
    gmsh.model.geo.addLine(7, 8, 5)
    gmsh.model.geo.addCircleArc(8, 4, 9, 6)
    gmsh.model.geo.addLine(9, 10, 7)
    gmsh.model.geo.addCircleArc(10, 1, 11, 8) 
    gmsh.model.geo.addCircleArc(11, 1, 7, 9) 

    # Add surfaces
    gmsh.model.geo.addCurveLoop([1, 2], 1)
    gmsh.model.geo.addCurveLoop([3, 4], 2)
    gmsh.model.geo.addCurveLoop([5, 6, 7, 8, 9], 3)
    gmsh.model.geo.addPlaneSurface([3, 1, 2], 1)

    # Convert gmsh to fenics mesh
    mesh, part_info,t_part_info = generate_fenics_mesh(N)
    gmsh.finalize()

    if N:
        V = fe.VectorFunctionSpace(mesh,"CG", 1)
        F = fe.FunctionSpace(mesh,"DG", 0)
    else:
        V = fe.VectorFunctionSpace(mesh, "CG", 1)
        F = fe.FunctionSpace(mesh, "CG", 1)

    u = fe.TrialFunction(V)
    du = fe.TestFunction(V)

    rho = fe.TrialFunction(F)
    drho = fe.TestFunction(F)

    boundaries = fe.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(0)
    tol = 1E-5
    #### Dirichlet boundary
    class DirBd(fe.SubDomain):
        def inside(self, x, on_boundary):
            return (x[0]**2 + x[1]**2 <= (r1 + tol)**2) and on_boundary
    dirBd = DirBd()
    dirBd.mark(boundaries,1)
    bcs = [adj.DirichletBC(V, (0.0,0.0), dirBd)]

    ##### Traction boundary
    class TracBd(fe.SubDomain):
        def inside(self, x, on_boundary):
            return (((x[0] - L)**2 + x[1]**2 <= (r2 + tol)**2) and x[1] <= 0) and on_boundary
    tracBd = TracBd()
    tracBd.mark(boundaries,2)

    t = adj.Constant((0.0, -1.0))
    ds = fe.Measure("ds")(mesh, subdomain_data=boundaries)
    return mesh, V, F, bcs, t, ds, u, du, rho, drho, part_info,t_part_info

def get_lshape2d_mesh(L = 2, H = 2, hmax = 0.1, N =None):
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add('lshape')

    #add points
    gmsh.model.geo.addPoint(0, 0, 0, hmax, 1)
    gmsh.model.geo.addPoint(L, 0, 0, hmax, 2)
    gmsh.model.geo.addPoint(L, 1, 0, hmax, 3)
    gmsh.model.geo.addPoint(1, 1, 0, hmax, 4)
    gmsh.model.geo.addPoint(1, H, 0, hmax, 5)
    gmsh.model.geo.addPoint(0, H, 0, hmax, 6)

    #add lines
    gmsh.model.geo.addLine(1, 2, 1)
    gmsh.model.geo.addLine(2, 3, 2)
    gmsh.model.geo.addLine(3, 4, 3)
    gmsh.model.geo.addLine(4, 5, 4)
    gmsh.model.geo.addLine(5, 6, 5)
    gmsh.model.geo.addLine(6, 1, 6)

    #add surface
    gmsh.model.geo.addCurveLoop([1,2,3,4,5,6], 1)
    gmsh.model.geo.addPlaneSurface([1], 1)

    #synchronize
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)

    mesh, part_info,t_part_info = generate_fenics_mesh(N)
    gmsh.finalize()

    if N:
        V = fe.VectorFunctionSpace(mesh,"CG", 1)
        F = fe.FunctionSpace(mesh,"DG", 0)
    else:
        V = fe.VectorFunctionSpace(mesh, "CG", 1)
        F = fe.FunctionSpace(mesh, "CG", 1)

    u = fe.TrialFunction(V)
    du = fe.TestFunction(V)

    rho = fe.TrialFunction(F)
    drho = fe.TestFunction(F)

    boundaries = fe.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(0)

    #dirichlet boundary
    class DirBd(fe.SubDomain):
        def inside(self, x, on_boundary):
            return(fe.near(x[1], 2.0) and on_boundary)
    dirBd = DirBd()
    dirBd.mark(boundaries, 1)
    bcs = [adj.DirichletBC(V, (0.0, 0.0), dirBd)]

    #traction boundary
    class TracBd(fe.SubDomain):
        def inside(self, x, on_boundary):
            # return((fe.near(x[0], L) and (fe.near(x[1], 1))) and on_boundary)
            return (fe.near(x[1], 1) and x[0] >= 2-hmax)
    tracBd = TracBd()
    tracBd.mark(boundaries, 2)
    t = adj.Constant((0.0, -1.0))
    ds = fe.Measure("ds")(mesh, subdomain_data=boundaries)
    return mesh, V, F, bcs, t, ds, u, du, rho, drho, part_info, t_part_info

def get_halfcircle2d_mesh(R= 1, alpha= 0.1, hmax= 0.1, N= None):
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add('halfcircle2d')

    # Add points
    gmsh.model.geo.addPoint(-R, 0, 0, hmax, 1)
    gmsh.model.geo.addPoint(-R + alpha, 0, 0, hmax, 2)
    gmsh.model.geo.addPoint(-alpha/2, 0, 0, hmax, 3)
    gmsh.model.geo.addPoint(0, 0, 0, hmax, 4)
    gmsh.model.geo.addPoint(alpha/2, 0, 0, hmax, 5)
    gmsh.model.geo.addPoint(R - alpha, 0, 0, hmax, 6)
    gmsh.model.geo.addPoint(R, 0, 0, hmax, 7)

    # Add lines
    gmsh.model.geo.addCircleArc(7, 4, 1, 1)
    gmsh.model.geo.addLine(1, 2, 2)
    gmsh.model.geo.addLine(2, 3, 3)
    gmsh.model.geo.addLine(3, 5, 4)
    gmsh.model.geo.addLine(5, 6, 5)
    gmsh.model.geo.addLine(6, 7, 6)

    # Add surfaces
    gmsh.model.geo.addCurveLoop([1, 2, 3, 4, 5, 6], 1)
    gmsh.model.geo.addPlaneSurface([1], 1)

    # Synchronize
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)

    mesh, part_info = generate_fenics_mesh(N)
    gmsh.finalize()

    # Function spaces
    V = fe.VectorFunctionSpace(mesh, "CG", 1)
    u = fe.TrialFunction(V)
    du = fe.TestFunction(V)
    F = fe.FunctionSpace(mesh, "CG", 1)
    rho = fe.TrialFunction(F)
    drho = fe.TestFunction(F)

    # Boundary marker
    boundaries = fe.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(0)

    # Dirichlet boundary
    class DirBd(fe.SubDomain):
        def inside(self, x, on_boundary):
            return (fe.near(x[0], -1) and fe.near(x[0],1) and on_boundary)
    dirBd = DirBd()
    dirBd.mark(boundaries, 1)
    bcs = [adj.DirichletBC(V, (0.0,0.0), dirBd)]

    # Traction boundary
    class TracBd(fe.SubDomain):
        def inside(self, x, on_boundary):
            return (fe.near(x[0], 0) and (fe.near(x[1],0)) and on_boundary)
    tracBd = TracBd()
    tracBd.mark(boundaries, 2)
    t = adj.Constant((0.0, -1.0))
    ds = fe.Measure("ds")(mesh, subdomain_data=boundaries)

    return mesh, V, F, bcs, t, ds, u, du, rho, drho, part_info
def get_hook2d_mesh(hmax = 0.1, N = None):
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add("hook2d")
    p1 = gmsh.model.geo.addPoint(0, 8.06226, 0, hmax)
    p2 = gmsh.model.geo.addPoint(-1.99852218, 8.35604405, 0, hmax)
    p3 = gmsh.model.geo.addPoint(2.015169545, 8.201872682, 0, hmax)

    c1 = gmsh.model.geo.addCircleArc(p2,p1,p3)
    p4 = gmsh.model.geo.addPoint(2.02, 8.06226, 0, hmax)
    c2 = gmsh.model.geo.addCircleArc(p3,p1,p4)
    p5 = gmsh.model.geo.addPoint(1.977358411, 7.649399598, 0, hmax)
    c3 = gmsh.model.geo.addCircleArc(p4, p1, p5)

    p6 = gmsh.model.geo.addPoint(5.48716, 7.68672, 0, hmax)
    p7 = gmsh.model.geo.addPoint(3.588057725, 4.734854394, 0, hmax)
    c4 = gmsh.model.geo.addCircleArc(p5, p6, p7)

    p8 = gmsh.model.geo.addPoint(4.094842924, 4.464678231, 0, hmax) #point 13
    l1 = gmsh.model.geo.addLine(p7, p8)


    p9 = gmsh.model.geo.addPoint(1.48842,0.18605, 0, hmax)
    p10 = gmsh.model.geo.addPoint(6.49842, 0.18605, 0, hmax)
    p11 = gmsh.model.geo.addPoint(6.489182725, -0.118042363, 0, hmax)
    c5 = gmsh.model.geo.addCircleArc(p8, p9, p10)
    # p9 = gmsh.model.geo.addPoint(14.8842,1.8605, 0, 0, hmax) #point 8


    l2 = gmsh.model.geo.addLine(p10, p11)

    p12 = gmsh.model.geo.addPoint(-3.51493010, -0.07199491029, 0, hmax) #point16
    c6 = gmsh.model.geo.addCircleArc(p11, p9, p12)
    p13 = gmsh.model.geo.addPoint(-3.5181199919, -0.00011473716, 0, hmax) #point 14
    l3 = gmsh.model.geo.addLine(p12, p13)

    p14 = gmsh.model.geo.addPoint(-2.70406, 0, 0, hmax)
    p15 = gmsh.model.geo.addPoint(-1.90405294261, 0.1506067453, 0, hmax)
    c7 = gmsh.model.geo.addCircleArc(p13, p14, p15)
    p16 = gmsh.model.geo.addPoint(-1.904052942, -0.1506067453, 0, hmax)
    l4 = gmsh.model.geo.addLine(p15, p16)
    p17 = gmsh.model.geo.addPoint(0,0,0,hmax)
    p18 = gmsh.model.geo.addPoint(1.91, 0, 0, hmax) #point 8
    c8 = gmsh.model.geo.addCircleArc(p16, p17, p18)
    p19 = gmsh.model.geo.addPoint(1.344501352, 1.35661936, 0, hmax)
    c9 = gmsh.model.geo.addCircleArc(p18, p17, p19)
    p20 = gmsh.model.geo.addPoint(0.9499630023, 1.6570064255, 0, hmax)
    c10 = gmsh.model.geo.addCircleArc(p19, p17, p20)
    p21 = gmsh.model.geo.addPoint(-1.21950878021, 4.40116823337, 0, hmax)
    p22 = gmsh.model.geo.addPoint(5.99713,7.87683, 0, hmax)
    c11 = gmsh.model.geo.addCircleArc(p20,p22, p21)
    p23 = gmsh.model.geo.addPoint(-2.012854658, 7.89250687859, 0, hmax)
    c12 = gmsh.model.geo.addCircleArc(p21, p22, p23)
    l5 = gmsh.model.geo.addLine(p23, p2)
    p25 = gmsh.model.geo.addPoint(1.01,8.06226,0,hmax)
    p26 = gmsh.model.geo.addPoint(-1.01,8.06226,0,hmax)
    c13 = gmsh.model.geo.addCircleArc(p25,p1,p26)
    c14 = gmsh.model.geo.addCircleArc(p26,p1,p25)

    cL1 = gmsh.model.geo.addCurveLoop([c1,c2,c3,c4,l1,c5,l2,c6,l3,c7,l4,c8,c9,c10,c11,c12,l5],1)
    cL2 = gmsh.model.geo.addCurveLoop([c13,c14],2)

    gmsh.model.geo.addPlaneSurface([1,2],1)

    mesh, part_info,t_part_info = generate_fenics_mesh(N)
    
    gmsh.finalize()

    if N:
        V = fe.VectorFunctionSpace(mesh,"CG", 1)
        F = fe.FunctionSpace(mesh,"DG", 0)
    else:
        V = fe.VectorFunctionSpace(mesh, "CG", 1)
        F = fe.FunctionSpace(mesh, "CG", 1)

    u = fe.TrialFunction(V)
    du = fe.TestFunction(V)
    
    rho = fe.TrialFunction(F)
    drho = fe.TestFunction(F)

    # Boundary marker
    boundaries = fe.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(0)
    # Dirichlet boundary
    class DirBdSupp(fe.SubDomain):
        def inside(self, x, on_boundary):
            # return (fe.near(x[1], 0.0) and x[0] >= L - 0.05*H) and on_boundary
            return (
                x[0]**2 + (x[1] - 8.06226)**2 < (1.01 + 1e-4)**2 and \
                x[1] > 8.06226 - 1e-4
                ) and on_boundary
    dirBdSupp = DirBdSupp()
    # dirBdSupp = DirBdSupp()
    dirBdSupp.mark(boundaries, 1)
    # bcs = [adj.DirichletBC(V.sub(0), 0.0, dirBdSym), adj.DirichletBC(V.sub(1), 0.0, dirBdSupp)]
    # bcs = [adj.DirichletBC(V.sub(0), 0.0, dirBdSym), adj.DirichletBC(V.sub(1), 0.0, dirBdSupp,method = 'pointwise')]
    bcs = [adj.DirichletBC(V, (0.0, 0.0), dirBdSupp)]
    # Traction boundary
    class TracBd(fe.SubDomain):
        def inside(self, x, on_boundary):
            return (
                x[0]**2 + x[1]**2 < (1.91 + 1e-4)**2 and \
                x[1] < 1e-4
            ) and on_boundary
    tracBd = TracBd()
    tracBd.mark(boundaries, 2)
    t = adj.Constant((0.0, -0.1))
    ds = fe.Measure("ds")(mesh, subdomain_data=boundaries)
    return mesh, V, F, bcs, t, ds, u, du, rho, drho, part_info, t_part_info
def get_dof_map(F):
    v2d = fe.vertex_to_dof_map(F)
    d2v = fe.dof_to_vertex_map(F)
    return v2d, d2v

def get_hook3d_mesh(hmax = 0.1, N = None):
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add("hook2d")
    p1 = gmsh.model.occ.addPoint(0, 8.06226/2, 0, hmax)
    p2 = gmsh.model.occ.addPoint(-1.99852218/2, 8.35604405/2, 0, hmax)
    p3 = gmsh.model.occ.addPoint(2.015169545/2, 8.201872682/2, 0, hmax)

    c1 = gmsh.model.occ.addCircleArc(p2,p1,p3)
    p4 = gmsh.model.occ.addPoint(2.02/2, 8.06226/2, 0, hmax)
    c2 = gmsh.model.occ.addCircleArc(p3,p1,p4)
    p5 = gmsh.model.occ.addPoint(1.977358411/2, 7.649399598/2, 0, hmax)
    c3 = gmsh.model.occ.addCircleArc(p4, p1, p5)

    p6 = gmsh.model.occ.addPoint(5.48716/2, 7.68672/2, 0, hmax)
    p7 = gmsh.model.occ.addPoint(3.588057725/2, 4.734854394/2, 0, hmax)
    c4 = gmsh.model.occ.addCircleArc(p5, p6, p7)

    p8 = gmsh.model.occ.addPoint(4.094842924/2, 4.464678231/2, 0, hmax) #point 13
    l1 = gmsh.model.occ.addLine(p7, p8)


    p9 = gmsh.model.occ.addPoint(1.48842/2,0.18605/2, 0, hmax)
    p10 = gmsh.model.occ.addPoint(6.49842/2, 0.18605/2, 0, hmax)
    p11 = gmsh.model.occ.addPoint(6.489182725/2, -0.118042363/2, 0, hmax)
    c5 = gmsh.model.occ.addCircleArc(p8, p9, p10)
    # p9 = gmsh.model.occ.addPoint(14.8842,1.8605, 0, 0, hmax) #point 8


    l2 = gmsh.model.occ.addLine(p10, p11)

    p12 = gmsh.model.occ.addPoint(-3.51493010/2, -0.07199491029/2, 0, hmax) #point16
    c6 = gmsh.model.occ.addCircleArc(p11, p9, p12)
    p13 = gmsh.model.occ.addPoint(-3.5181199919/2, -0.00011473716/2, 0, hmax) #point 14
    l3 = gmsh.model.occ.addLine(p12, p13)

    p14 = gmsh.model.occ.addPoint(-2.70406/2, 0, 0, hmax)
    p15 = gmsh.model.occ.addPoint(-1.90405294261/2, 0.1506067453/2, 0, hmax)
    c7 = gmsh.model.occ.addCircleArc(p13, p14, p15)
    p16 = gmsh.model.occ.addPoint(-1.904052942/2, -0.1506067453/2, 0, hmax)
    l4 = gmsh.model.occ.addLine(p15, p16)
    p17 = gmsh.model.occ.addPoint(0,0,0,hmax)
    p18 = gmsh.model.occ.addPoint(1.91/2, 0, 0, hmax) #point 8
    c8 = gmsh.model.occ.addCircleArc(p16, p17, p18)
    p19 = gmsh.model.occ.addPoint(1.344501352/2, 1.35661936/2, 0, hmax)
    c9 = gmsh.model.occ.addCircleArc(p18, p17, p19)
    p20 = gmsh.model.occ.addPoint(0.9499630023/2, 1.6570064255/2, 0, hmax)
    c10 = gmsh.model.occ.addCircleArc(p19, p17, p20)
    p21 = gmsh.model.occ.addPoint(-1.21950878021/2, 4.40116823337/2, 0, hmax)
    p22 = gmsh.model.occ.addPoint(5.99713/2,7.87683/2, 0, hmax)
    c11 = gmsh.model.occ.addCircleArc(p20,p22, p21)
    p23 = gmsh.model.occ.addPoint(-2.012854658/2, 7.89250687859/2, 0, hmax)
    c12 = gmsh.model.occ.addCircleArc(p21, p22, p23)
    l5 = gmsh.model.occ.addLine(p23, p2)
    p25 = gmsh.model.occ.addPoint(1.01/2,8.06226/2,0,hmax)
    p26 = gmsh.model.occ.addPoint(-1.01/2,8.06226/2,0,hmax)
    c13 = gmsh.model.occ.addCircleArc(p25,p1,p26)
    c14 = gmsh.model.occ.addCircleArc(p26,p1,p25)

    cL1 = gmsh.model.occ.addCurveLoop([c1,c2,c3,c4,l1,c5,l2,c6,l3,c7,l4,c8,c9,c10,c11,c12,l5],1)
    cL2 = gmsh.model.occ.addCurveLoop([c13,c14],2)

    gmsh.model.occ.addPlaneSurface([1],1)
    gmsh.model.occ.addPlaneSurface([2],2)
    gmsh.model.occ.cut([(2,1)],[(2,2)])
    # gmsh.model.occ.fuse([1,2],[1,1])
    gmsh.model.occ.extrude([(2, 1)], 0, 0, 0.5)
    # gmsh.model.occ.dilate([(3,1)],0,0,0,0.5,0.5,1)
    

    mesh, part_info,t_part_info = generate_fenics_mesh_3d(N)
    gmsh.finalize()
    if N:
        V = fe.VectorFunctionSpace(mesh,"CG", 1)
        F = fe.FunctionSpace(mesh,"DG", 0)
    else:
        V = fe.VectorFunctionSpace(mesh, "CG", 1)
        F = fe.FunctionSpace(mesh, "CG", 1)

    u = fe.TrialFunction(V)
    du = fe.TestFunction(V)
    
    rho = fe.TrialFunction(F)
    drho = fe.TestFunction(F)

    # Boundary marker
    boundaries = fe.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(0)
    # Dirichlet boundary
    tol = 1E-5
    class DirBdSupp(fe.SubDomain):
        def inside(self, x, on_boundary):
            # return (fe.near(x[1], 0.0) and x[0] >= L - 0.05*H) and on_boundary
            return (
                x[0]**2 + (x[1] - 8.06226/2)**2 < (1.01/2 + tol)**2 and \
                x[1] > 8.06226/2 - tol
                ) and on_boundary
    class DirBdsym(fe.SubDomain):
        def inside(self, x, on_boundary):
            return (x[2] <= tol) and on_boundary
    dirBdSupp, dirBdsym = [DirBdSupp(), DirBdsym()]

    dirBdSupp.mark(boundaries, 1)
    dirBdsym.mark(boundaries,1)
    
    dbx = adj.DirichletBC(V, adj.Constant((0.0, 0.0, 0.0)), dirBdSupp)
    dbsym = adj.DirichletBC(V.sub(2), adj.Constant(0.0), dirBdsym)
    bcs = [dbx, dbsym]
    # Traction boundary
    class TracBd(fe.SubDomain):
        def inside(self, x, on_boundary):
            return (
                x[0]**2 + x[1]**2 < (1.91/2 + tol)**2 and \
                x[1] < tol
            ) and on_boundary
    tracBd = TracBd()
    tracBd.mark(boundaries, 2)
    t = adj.Constant((0.0, -1.0, 0.0))
    ds = fe.Measure("ds")(mesh, subdomain_data=boundaries)
    return mesh, V, F, bcs, t, ds, u, du, rho, drho, part_info, t_part_info
def get_dof_map(F):
    v2d = fe.vertex_to_dof_map(F)
    d2v = fe.dof_to_vertex_map(F)
    return v2d, d2v

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation

    mesh, V, F, bcs, t, ds, u, du, rho, drho, part_info = get_mbb2d_mesh(hmax=0.02, N=4)

    for n, e in zip(part_info['nodes'], part_info['elems']):
        T = Triangulation(*mesh.coordinates().T, triangles=mesh.cells()[e])
        plt.triplot(T)
    count = np.zeros((len(mesh.coordinates()), 1))  ### num of patches overlab by node
    for pn in part_info['nodes']:
        count[pn] += 1
    plt.axis('image')
    plt.show()
    pass