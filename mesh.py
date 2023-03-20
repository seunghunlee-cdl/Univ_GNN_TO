import fenics as fe
import fenics_adjoint as adj
import gmsh
import meshio
import numpy as np

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
    else:
        part_info = None

    # Generate fenics mesh from gmsh
    mesh_out = meshio.Mesh(
        points=coords,
        cells={'triangle': elems}
    )
    meshio.write(TEMP_MESH_PATH, mesh_out)

    mesh = adj.Mesh()    
    with fe.XDMFFile(TEMP_MESH_PATH) as file:
        file.read(mesh)

    return mesh, part_info


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
    
    # Convert gmsh to fenics mesh
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
            return (fe.near(x[1], 0.0) and x[0] >= L - 0.03*H) and on_boundary
    dirBdSym, dirBdSupp = [DirBdSym(),DirBdSupp()]
    # dirBdSupp = DirBdSupp()
    dirBdSym.mark(boundaries, 1)
    dirBdSupp.mark(boundaries, 1)
    bcs = [adj.DirichletBC(V.sub(0), 0.0, dirBdSym), adj.DirichletBC(V.sub(1), 0.0, dirBdSupp)]
    # Traction boundary
    class TracBd(fe.SubDomain):
        def inside(self, x, on_boundary):
            return (fe.near(x[1], H) and x[0] <= 0.05*H)
    tracBd = TracBd()
    tracBd.mark(boundaries, 2)
    t = adj.Constant((0.0, -1.0))
    ds = fe.Measure("ds")(mesh, subdomain_data=boundaries)
    return mesh, V, F, bcs, t, ds, u, du, rho, drho, part_info

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
    mesh, part_info = generate_fenics_mesh(N)
    gmsh.finalize()

    V = fe.VectorFunctionSpace(mesh, "CG", 1)
    u = fe.TrialFunction(V)
    du = fe.TestFunction(V)
    F = fe.FunctionSpace(mesh, "CG", 1)
    rho = fe.TrialFunction(F)
    drho = fe.TestFunction(F)

    boundaries = fe.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(0)

    #### Dirichlet boundary
    class DirBd(fe.SubDomain):
        def inside(self, x, on_boundary):
            return (x[0]**2 + x[1]**2 <= (r1 + 1e-9)**2) and on_boundary
    dirBd = DirBd()
    dirBd.mark(boundaries,1)
    bcs = [adj.DirichletBC(V, (0.0,0.0), dirBd)]

    ##### Traction boundary
    class TracBd(fe.SubDomain):
        def inside(self, x, on_boundary):
            return (((x[0] - L)**2 + x[1]**2 <= (r2 + 1e-9)**2) and x[1] <= 0) and on_boundary
    tracBd = TracBd()
    tracBd.mark(boundaries,2)

    t = adj.Constant((0.0, -1.0))
    ds = fe.Measure("ds")(mesh, subdomain_data=boundaries)
    return mesh, V, F, bcs, t, ds, u, du, rho, drho, part_info

def halfcircle2d(R= 1, alpha= 0.1, hmax= 0.1, N= None):
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


def get_dof_map(F):
    v2d = fe.vertex_to_dof_map(F)
    d2v = fe.dof_to_vertex_map(F)
    return v2d, d2v


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation

    mesh, V, F, bcs, t, ds, u, du, rho, drho, part_info = get_wrench2d_mesh(hmax=0.08, N=20)

    for n, e in zip(part_info['nodes'], part_info['elems']):
        T = Triangulation(*mesh.coordinates().T, triangles=mesh.cells()[e])
        plt.triplot(T)
    count = np.zeros((len(mesh.coordinates()), 1))  ### num of patches overlab by node
    for pn in part_info['nodes']:
        count[pn] += 1
    plt.axis('image')
    plt.show()
    pass