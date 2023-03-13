import fenics as fe
import fenics_adjoint as adj
import gmsh
import meshio
import numpy as np

TEMP_MESH_PATH = "/workspace/output/tmp_mesh_output.xdmf"


def load_from_gmsh(model, N=None):
    ndim = model.getDimension()
    # Get node coordinates
    idx, x, _ = model.mesh.getNodes()
    x = x.reshape(-1, 3)[:, :ndim]

    # Get elements
    _, elems_raw, cells = map(lambda x: x[0], model.mesh.getElements(dim=2))
    cells = cells.reshape(-1, 3) - 1
    num_cells = len(cells)
    cell_data = np.zeros(num_cells, dtype=int)

    part_info = None
    if N is None:
        mesh_out = meshio.Mesh(
            points=x,
            cells={'triangle': cells},
            cell_data={'name_to_read': [cell_data]}
        )
        meshio.write(TEMP_MESH_PATH, mesh_out)

        mesh = adj.Mesh()    
        with fe.XDMFFile(TEMP_MESH_PATH) as file:
            file.read(mesh)
            mvc = fe.MeshValueCollection('size_t', mesh, mesh.topology().dim())
            file.read(mvc, 'name_to_read')
    else:
        numElements = len(model.mesh.getElements(2)[1][0])
        numNodes = len(model.mesh.getNodes()[0])
        numPartitions = numNodes // N
        model.mesh.partition(numPartitions)

        num_parts = model.getNumberOfPartitions()
        elems_per_part = [[] for _ in range(num_parts)]  ###elem num
        entities = model.getEntities(2)
        for dim, tag in entities:
            part_id = model.getPartitions(dim, tag)
            if part_id:
                elem_id = model.mesh.getElements(dim, tag)[1][0]
                _, comm, _ = np.intersect1d(elems_raw, elem_id, return_indices = True)
                elems_per_part[part_id[0]-1] = comm
        nodes_per_part = []
        for part_id, part in enumerate(elems_per_part):
            cell_data[part] = part_id
            nodes_per_part.append(np.unique(cells[part]))
        mesh_out = meshio.Mesh(
            points = x,
            cells = {'triangle' : cells},
            cell_data = {'name_to_read': [cell_data]}
        )
        meshio.write(TEMP_MESH_PATH, mesh_out)

        mesh = adj.Mesh()
        with fe.XDMFFile(TEMP_MESH_PATH) as file:
            file.read(mesh)
            mvc = fe.MeshValueCollection('size_t', mesh, mesh.topology().dim())
            file.read(mvc, 'name_to_read')
        # mf = fe.cpp.mesh.MeshFunctionSizet(mesh,mvc)

        part_info = dict(
            nodes=nodes_per_part,
            elems=elems_per_part
        )

    return mesh, part_info


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

    # Synchronize
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)
    
    # Convert gmsh to fenics mesh
    mesh, part_info = load_from_gmsh(gmsh.model, N)
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

    return mesh, V, F, bcs, t, ds, u, du, rho, drho, part_info
    

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
    mesh, part_info = load_from_gmsh(gmsh.model, N)
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
    class DirBdSym(fe.SubDomain):
        def inside(self, x, on_boundary):
            return (fe.near(x[0], 0.0) and on_boundary)
    class DirBdSupp(fe.SubDomain):
        def inside(self, x, on_boundary):
            return (fe.near(x[1], 0.0) and x[0] >= L - 0.1*H) and on_boundary
    dirBdSym = DirBdSym()
    dirBdSupp = DirBdSupp()
    dirBd.mark(boundaries, 1)
    bcs = [adj.DirichletBC(V.sub(0), 0.0, dirBdSym), adj.DirichletBC(V.sub(1), 0.0, dirBdSupp)]

    # Traction boundary
    class TracBd(fe.SubDomain):
        def inside(self, x, on_boundary):
            return (fe.near(x[1], H) and x[0] <= 0.1*H) and on_boundary
    tracBd = TracBd()
    tracBd.mark(boundaries, 2)
    t = adj.Constant((0.0, -1.0))
    ds = fe.Measure("ds")(mesh, subdomain_data=boundaries)

    return mesh, V, F, bcs, t, ds, u, du, rho, drho, part_info


def get_dof_map(F):
    v2d = fe.vertex_to_dof_map(F)
    d2v = fe.dof_to_vertex_map(F)
    return v2d, d2v