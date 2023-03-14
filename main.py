from time import time

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import torch
import torch_geometric as pyg
from fenics import (XDMFFile, as_backend_type, dx, grad, inner, parameters,
                    plot, set_log_active)
from fenics_adjoint import (Constant, Control, Function, assemble,
                            compute_gradient, interpolate, project, solve)
from scipy.sparse.linalg import factorized

from fem import (build_weakform_filter, build_weakform_struct, epsilon,
                 input_assemble, output_assemble, sigma)
from mesh import get_clever2d_mesh, get_dof_map
from MMA import mmasub, subsolv
from model import MyGNN, generate_data, training

t_start = time()
parameters["linear_algebra_backend"] = "Eigen"
set_log_active(False)

save_dir = XDMFFile("output/result.xdmf")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## parameters
rmin = 0.01
volfrac = 0.5
maxiter = 100
N = 20    ## number of node in patch
hmax = 0.05
hmaxC = 0.1
Ni = 10
Nf = 10
Wi = 10
Wu = 10
batch_size = 300
epochs = 20
n_hidden = 200
n_layer = 5
lr = 0.005

## time
t_filter = []
t_input = []
t_fine = []
t_dcdv = []
input_apd = []
output_apd = []

mesh, V, F, bcs, t, ds, u, du, rho, drho, part_info = get_clever2d_mesh(hmax=hmax, N=N)
meshC, VC, FC, bcsC, tC, dsC, uC, duC, _, _, _ = get_clever2d_mesh(hmax = hmaxC)
print("fine :", mesh.num_entities(0),",","Coarse :", meshC.num_entities(0),",","Patch_elem :", N)

v2d, d2v = get_dof_map(F)

uh = Function(V)
phih = Function(F)   ## density
phih.vector()[:] = volfrac
dc_pred = Function(F)
m = Control(phih)
rhoh = Function(F)   ## Filtered density

## MMA parameters
mm = 1
n = mesh.num_entities(0)
xmin = np.zeros((n,1))
xmax = np.ones((n,1))
xval = phih.vector()[:][np.newaxis].T
xold1 = xval.copy()
xold2 = xval.copy()
low = np.ones((n,1))
upp = np.ones((n,1))
a0 = 1.0
aa = np.zeros((mm,1))
c = 10000*np.ones((mm,1))
d = np.zeros((mm,1))
move = 0.2

aH, LH = build_weakform_filter(rho, drho, phih, rmin)
a, L = build_weakform_struct(u, du, rhoh, t, ds)

uhC = Function(VC)
rhohC = Function(FC)
rhohC.vector()[:] = volfrac

aC, LC = build_weakform_struct(uC, duC, rhohC, tC, dsC)

### solve FE
tic = time()
rhohC.vector()[:] = project(rhoh, FC).vector()[:]
toc = time()
print(f'mapping: {toc - tic} sec')
tic = time()
solve(aC == LC, uhC, bcs=bcsC, solver_parameters={'linear_solver': 'umfpack'})
toc = time()
print(f'solving(coarse): {toc - tic} sec')

loop = 0
while loop < maxiter:
    tic = time()
    solve(aH == LH, rhoh, solver_parameters={'linear_solver': 'umfpack'}) ## density Filtering
    t_filter.append(time()-tic)

    tic = time()
    x = input_assemble(uhC, FC, F, v2d, rhoh)
    t_input.append(time()-tic)
    x_last = x  ##있어야할지 잘 모르겠지만 우선 넣어본다

    if(loop<Ni+Wi) or (divmod(max(loop-Ni-Wi,1),Nf)[1]==0):
        tic = time()
        solve(a == L, uh, bcs=bcs, solver_parameters={'linear_solver': 'umfpack'})  ## fine
        t_fine.append(time()-tic)

        ## objective, constraint
        Ws = inner(sigma(uh,rhoh), epsilon(uh))
        comp = assemble(Ws*dx)
        vol = assemble(rhoh*dx)
        tic = time()
        dc = compute_gradient(comp, m)   ### fine sensitivity
        dv = compute_gradient(vol, m)
        t_dcdv.append(time()-tic)

        ## Store
        y = output_assemble(dc, v2d)
        
        input_apd.append(x)
        output_apd.append(y)

        if loop == Ni + Wi -1:
            data_list = [generate_data(input_apd[-Wi:], output_apd[-Wi:], node_ids, cell_ids,mesh) for node_ids, cell_ids in zip(part_info['nodes'], part_info['elems'])]
            lose_hist, loader, net  = training(data_list, batch_size, n_hidden, n_layer, lr, epochs, device)
        elif divmod(max(loop-Ni-Wi,1), Nf)[1] == 0:
            data_list = [generate_data(input_apd[-Wu:], output_apd[-Wu:], node_ids, cell_ids,mesh) for node_ids, cell_ids in zip(part_info['nodes'], part_info['elems'])]
            lose_hist, loader, net = training(data_list, batch_size, n_hidden, n_layer, lr, epochs, device)
        
        ## MMA parameters
        mu0 = 1.0
        mu1 = 1.0
        f0val = comp
        df0dx = dc.vector()[:].reshape(-1,1)
        fval = np.array([[rhoh.vector()[:].sum()/n-volfrac]])
        dfdx = dv.vector()[:].reshape(1,-1)
        xval = rhoh.vector()[:].reshape(-1,1)
        xmma,ymma,zmma,lam,xsi,eta,mu,zet,s,low,upp = \
            mmasub(mm,n,loop,xval,xmin,xmax,xold1,xold2,f0val,df0dx,fval,dfdx,low,upp,a0,aa,c,d,move)
        xold2 = xold1.copy()
        xold1 = xval.copy()
        phih.vector()[:] = xmma.ravel()
    
    else:
        new_input = x_last
        with torch.no_grad():
            for batch in loader:
                yhat = net(batch.x.to(device), batch.edge_index.to(device())).cpu()
                dc_pred.vector()[v2d[batch.global_idx]] = yhat.numpy()[:,0]
        
        vol = assemble(rhoh*dx)
        dv = compute_gradient(vol, m)

        ## MMA parameters
        mu0 = 1.0
        mu1 = 1.0
        f0val = comp
        df0dx = dc_pred.vector()[:].reshape(-1,1)
        fval = np.array([[rhoh.vector()[:].sum()/n-volfrac]])
        dfdx = dv.vector()[:].reshape(1,-1)
        xval = rhoh.vector()[:].reshape(-1,1)
        xmma,ymma,zmma,lam,xsi,eta,mu,zet,s,low,upp = \
            mmasub(mm,n,loop,xval,xmin,xmax,xold1,xold2,f0val,df0dx,fval,dfdx,low,upp,a0,aa,c,d,move)
        xold2 = xold1.copy()
        xold1 = xval.copy()
        phih.vector()[:] = xmma.ravel()
    loop += 1
    print(f"it.: {loop}", f",obj.: {comp}")
save_dir.write(rhoh)
print("total time : ", time()-t_start)
print("filter time : ", sum(t_filter))
print("input time : ", sum(t_input))
print("fine time : ", sum(t_fine))
print("adj time : ", sum(t_dcdv))