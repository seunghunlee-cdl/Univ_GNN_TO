import os
import random
import shutil
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
from matplotlib.tri import Triangulation
from torch_geometric.data import Data

from fem import (build_weakform_filter, build_weakform_struct, epsilon,
                 input_assemble, oc, output_assemble, sigma)
from mesh import (get_clever2d_mesh, get_dof_map, get_mbb2d_mesh,
                  get_wrench2d_mesh, halfcircle2d)
from MMA import mmasub
from model import (MyGNN, generate_data, graph_partitioning, pred_input,
                   training)
from utils import (compute_theta_error, compute_triangle_area,
                   convolution_operator, filter, map_density, tree_maker)

set_log_active(False)
torch.cuda.empty_cache()
if os.path.exists("/workspace/output"):
    shutil.rmtree("/workspace/output")
SAVE_DIR = XDMFFile("output/result.xdmf")
os.mkdir("/workspace/output")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main(volfrac, maxiter, N, hmax, hamxC, rmin, Ni, Nf, Wi, Wu, target_step_per_epoch, epochs, n_hidden, n_layer, lr, optimizer, continuation):
    t_start = time()
    ## time
    t_filter = []
    t_input  = []
    t_output = []
    t_fine   = []
    t_coarse = []
    t_append = []
    t_dcdv   = []
    t_training=[]
    t_pred=[]
    t_optimizer=[]
    t_mesh = []
    t_pre = []
    input_apd=[]
    output_apd=[]

    tic = time()
    mesh, V, F, bcs, t, ds, u, du, rho, drho, part_info = get_mbb2d_mesh(hmax=hmax, N=N)
    meshC, VC, FC, bcsC, tC, dsC, uC, duC, _, _, _ = get_mbb2d_mesh(hmax = hmaxC)
    t_mesh.append(time()-tic)

    print("fine :", mesh.num_cells(),",","Coarse :", meshC.num_entities(0),",","Patch :", len(part_info['nodes']))

    batch_size = np.ceil(len(part_info['nodes'])/target_step_per_epoch).astype(int).item()
    # batch_size = 300
    tic = time()
    v2dC, d2vC = get_dof_map(FC)
    
    coords = mesh.coordinates()
    trias = mesh.cells()
    center = coords[trias].mean(1)

    areas = compute_triangle_area(coords[trias])
    fcc2cn = tree_maker(center, meshC)

    if continuation == True:
        penal = Constant(1.0)
    else:
        penal = Constant(3.0)

    H = convolution_operator(center, rmin)
    Hs = H@np.ones(mesh.num_cells())

    partitioned_graphs = graph_partitioning(coords, trias, part_info, center)

    uh = Function(V)
    phih = Function(F)   ## density
    phih.vector()[:] = volfrac
    dc_pred = Function(F)

    dc_bar = Function(F)
    dv_bar = Function(F)
    dc_pred_bar = Function(F)
    
    rhoh = Function(F)   ## Filtered density
    m = Control(phih)

    ## MMA parameters
    if optimizer == 0:
        mm = 1
        n = mesh.num_cells()
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

    T = Triangulation(*meshC.coordinates().T, triangles=meshC.cells())

    aH, LH = build_weakform_filter(rho, drho, phih, rmin) #### filter equation
    a, L = build_weakform_struct(u, du, rhoh, t, ds, penal) #### FEA-fine

    uhC = Function(VC)
    rhohC = Function(FC)

    aC, LC = build_weakform_struct(uC, duC, rhohC, tC, dsC, penal) #### FEA-coarse

    t_pre.append(time()-tic)
    loop = 0
    iteration = 0
    tict = time()
    while iteration < 40 and continuation:
        tic = time()
        rhoh.assign(phih)
        rhoh.vector()[:] = filter(H,Hs,rhoh.vector()[:])
        t_filter.append(time()-tic)

        tic = time()
        a, L = build_weakform_struct(u, du, rhoh, t, ds, penal)
        solve(a == L, uh, bcs = bcs)
        t_fine.append(time()-tic)

        tic = time()
        Ws = inner(sigma(uh,rhoh,penal), epsilon(uh))
        comp = assemble(Ws*dx)
        vol = (rhoh.vector()[:]*areas).sum()
        dc = compute_gradient(comp, m)
        t_dcdv.append(time()-tic)

        tic = time()
        dc_bar.vector()[:] = filter(H,Hs,dc.vector()[:])
        dv_bar.vector()[:] = filter(H,Hs,areas)
        t_filter.append(time()-tic)

        tic = time()
        if optimizer == 0:
            mu0 = 1.0
            mu1 = 1.0
            f0val = comp
            df0dx = dc_bar.vector()[:].reshape(-1,1)
            fval = np.array([[vol - volfrac*areas.sum()]])
            dfdx = dv_bar.vector()[:].reshape(1,-1)
            xval = phih.vector()[:].reshape(-1,1)
            xmma,ymma,zmma,lam,xsi,eta,mu,zet,s,low,upp = \
                mmasub(mm,n,iteration,xval,xmin,xmax,xold1,xold2,f0val,df0dx,fval,dfdx,low,upp,a0,aa,c,d,move)
            xold2 = xold1.copy()
            xold1 = xval.copy()
            phih.vector()[:] = xmma.ravel()
        elif optimizer == 1:
            phih.vector()[:] = oc(phih,dc_bar,dv_bar,mesh,H,Hs,volfrac,areas)
        t_optimizer.append(time()-tic)

        # plot(rhoh, cmap="gray_r")
        # plt.savefig("test.png")
        if iteration == 19:
            penal = Constant(2.0)
        iteration += 1
        print(f"it.: {iteration: 3d},\tobj.: {comp:.4e},\tvol.: {vol/areas.sum():.3f},\tpenal.: {penal.values()[0]}")

    penal_time = time()-tict
    penal = Constant(3.0)
    if continuation:
        a, L = build_weakform_struct(u, du, rhoh, t, ds, penal)
    while loop < maxiter:
        tic = time()
        rhoh.assign(phih)
        rhoh.vector()[:] = filter(H,Hs,rhoh.vector()[:])
        t_filter.append(time()-tic)

        # map_density(rhoh, rhohC, mesh, meshC, None, v2dC)
        rhohC.vector()[v2dC] = rhoh.vector()[fcc2cn] ## density mapping

        tic = time()
        solve(aC == LC, uhC, bcs=bcsC)  ##  Coarse FE
        t_coarse.append(time()-tic)

        tic = time()
        x, scaler = input_assemble(T, rhoh, uhC, V, F, FC, v2dC, center, scaler if loop > 0 else None)
        x_last = x  
        input_apd.append(x)
        t_input.append(time()-tic)
                
        if(loop<Ni+Wi) or (divmod(max(loop-Ni-Wi,1),Nf)[1]==0):
            tic = time()
            solve(a == L, uh, bcs=bcs)  ## fine
            t_fine.append(time()-tic)

            tic = time()
            Ws = inner(sigma(uh,rhoh,penal), epsilon(uh))
            comp = assemble(Ws*dx)
            vol = (rhoh.vector()[:]*areas).sum()
            dc = compute_gradient(comp, m)   ### fine sensitivity
            # dv = compute_gradient(vol, m)
            t_dcdv.append(time()-tic)

            tic = time()
            dc_bar.vector()[:] = filter(H,Hs,dc.vector()[:])
            dv_bar.vector()[:] = filter(H,Hs,areas)
            t_filter.append(time()-tic)

            ## Store
            tic = time()
            y, scalers, lb = output_assemble(
                dc, loop, scalers if loop > 0 else None, lb if loop > 0 else None,
                k=2)
            output_apd.append(y)
            t_output.append(time()-tic)

        
            if loop == Ni + Wi -1:
                data_list = []
                tic = time()
                for i in range(Wi):
                    data_list.append([generate_data(input_apd[-(i+1)], output_apd[-(i+1)], edge_ids, elem_ids, mesh) for edge_ids, elem_ids in zip(partitioned_graphs, part_info['elems'])])
                dataset = sum(data_list,[])
                t_append.append(time()-tic)

                tic = time()
                train_hist, val_hist, net  = training(dataset, batch_size, n_hidden, n_layer, lr, epochs, device)
                t_training.append(time()-tic)
            elif divmod(max(loop-Ni-Wi,1), Nf)[1] == 0:
                data_list = []
                tic = time()
                for i in range(Wu):
                    data_list.append([generate_data(input_apd[-(i+1)], output_apd[-(i+1)], edge_ids, elem_ids,mesh) for edge_ids, elem_ids in zip(partitioned_graphs, part_info['elems'])])
                dataset = sum(data_list,[])
                t_append.append(time()-tic)

                tic = time()
                train_hist, val_hist, net = training(dataset, batch_size, n_hidden, n_layer, lr, epochs, device, net)
                t_training.append(time()-tic)

            ## Optimizer parameters
            tic = time()
            if optimizer == 0:
                mu0 = 1.0
                mu1 = 1.0
                f0val = comp
                df0dx = dc_bar.vector()[:].reshape(-1,1)
                fval = np.array([[vol - volfrac*areas.sum()]])
                dfdx = dv_bar.vector()[:].reshape(1,-1)
                xval = phih.vector()[:].reshape(-1,1)
                xmma,ymma,zmma,lam,xsi,eta,mu,zet,s,low,upp = \
                    mmasub(mm,n,loop,xval,xmin,xmax,xold1,xold2,f0val,df0dx,fval,dfdx,low,upp,a0,aa,c,d,move)
                xold2 = xold1.copy()
                xold1 = xval.copy()
                phih.vector()[:] = xmma.ravel()
            elif optimizer == 1:
                phih.vector()[:] = oc(phih, dc_bar, dv_bar, mesh, H, Hs, volfrac,areas)
            t_optimizer.append(time()-tic)
        
        else:
            pred_input_data = [pred_input(x_last, edge_ids, elem_ids, mesh) for edge_ids, elem_ids in zip(partitioned_graphs, part_info['elems'])]
            pred_loader = pyg.loader.DataLoader(pred_input_data, batch_size = len(pred_input_data))
            tic = time()
            with torch.no_grad():
                net.eval()
                for batch in pred_loader:
                    yhat = net(batch.x.to(device), batch.edge_index.to(device)).cpu()
                    dc_pred.vector()[batch.global_idx] = yhat.numpy()[:, 0]

            dc_pred.vector()[:] = scalers.inverse_transform(dc_pred.vector()[:].reshape(-1,1)).ravel()
            dc_pred.vector()[np.where(dc_pred.vector()[:]>0)[0]]=0
            dc_pred_bar.vector()[:] = filter(H,Hs,dc_pred.vector()[:])
            t_pred.append(time()-tic)

            # therr = compute_theta_error(dc,dc_pred)    ###### theta_error
            # print(f'theta={therr:.3f}')

            tic = time()
            vol = (rhoh.vector()[:]*areas).sum()
            t_dcdv.append(time()-tic)

            tic = time()
            dv_bar.vector()[:] = filter(H,Hs,areas)
            t_filter.append(time()-tic)

            ## Optimizer parameters
            tic = time()
            if optimizer ==0:
                mu0 = 1.0
                mu1 = 1.0
                f0val = comp
                df0dx = dc_pred_bar.vector()[:].reshape(-1,1)
                fval = np.array([[vol - volfrac*areas.sum()]])
                dfdx = dv_bar.vector()[:].reshape(1,-1)
                xval = phih.vector()[:].reshape(-1,1)
                xmma,ymma,zmma,lam,xsi,eta,mu,zet,s,low,upp = \
                    mmasub(mm,n,loop,xval,xmin,xmax,xold1,xold2,f0val,df0dx,fval,dfdx,low,upp,a0,aa,c,d,move)
                xold2 = xold1.copy()
                xold1 = xval.copy()
                phih.vector()[:] = xmma.ravel()
            elif optimizer == 1:
                phih.vector()[:] = oc(phih, dc_pred_bar, dv_bar, mesh, H, Hs, volfrac,areas)
            t_optimizer.append(time()-tic)

        plt.cla()
        # plot(rhoh, cmap="gray_r")
        # plt.savefig("test.png")
        loop += 1
        # print(f"it.: {loop}", f",obj.: {comp}")
        print(f"it.: {loop: 3d},\tobj.: {comp:.4e},\tvol.: {vol/areas.sum():.3f}")
        
    # SAVE_DIR.write(rhoh)
    plot(rhoh, cmap = "gray_r")
    plt.savefig("test"+'.png')

    print("total time :", time()-t_start)
    print("mesh time :", sum(t_mesh))
    print("pre time :", sum(t_pre))
    print("filter time :", sum(t_filter))
    print("input time :", sum(t_input), "call :", len(t_input), "once :", sum(t_input)/len(t_input))
    print("output time :", sum(t_output))
    print("fine time :", sum(t_fine), "call :", len(t_fine), "once :", sum(t_fine)/len(t_fine))
    print("coarse time :", sum(t_coarse), "call :", len(t_coarse), "once :", sum(t_coarse)/len(t_coarse))
    print("sens time :", sum(t_dcdv))
    print("stack time :", sum(t_append))
    print("training time :", sum(t_training), "call :", len(t_training), "once :", sum(t_training)/len(t_training))
    print("pred time :", sum(t_pred), "call :", len(t_pred), "once :", sum(t_pred)/len(t_pred))
    print("optimizer time :", sum(t_optimizer), "call :", len(t_optimizer), "once :", sum(t_optimizer)/len(t_optimizer))


if __name__ == "__main__":
    # import sys
    # num_n = [5,50,100,150,200,250,350,450]
    # num_q = ('5.txt','50.txt','100.txt','150.txt','200.txt','250.txt','350.txt','450.txt')
    # for i in range(len(num_n)):

    ## parameters
    volfrac = 0.5
    maxiter = 200
    N = 100    ## number of elem in patch
    hmax = 0.0075
    hmaxC = hmax*2
    rmin = hmax*3
    Ni = 1
    Nf = 10
    Wi = 10
    Wu = 5
    target_step_per_epoch = 15
    epochs = 3
    n_hidden = 128
    n_layer = 4
    lr = 0.0005
    optimizer = 1   ####   0 --> MMA,   1 --> OC
    continuation = True
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
   
    
        # f = open(num_q[i],'w')
        # nnn = num_q[i]
        # N = num_n[i]
    main(volfrac, maxiter, N, hmax, hmaxC, rmin, Ni, Nf, Wi, Wu, target_step_per_epoch, epochs, n_hidden, n_layer, lr, optimizer, continuation)