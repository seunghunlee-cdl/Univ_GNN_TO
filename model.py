# import matplotlib.pyplot as plt
import numpy as np
import torch
import torch_geometric as pyg
from matplotlib.tri import Triangulation
from torch.utils.data import random_split
from tqdm.auto import tqdm


def generate_data(x, y, node_ids, cell_ids, mesh):
    coords = mesh.coordinates()
    num_nodes = len(coords)
    cells = mesh.cells()

    tmp = np.arange(num_nodes)
    tmp[node_ids] = np.arange(len(node_ids))
    x_by_part = torch.tensor(x[node_ids], dtype = torch.float)
    y_by_part = torch.tensor(y[node_ids], dtype = torch.float)
    cell_by_part = tmp[cells[cell_ids]]

    T = Triangulation(*coords[node_ids].T, triangles=cell_by_part)
    src, dst = T.edges.T
    edge_index = torch.tensor(np.c_[np.r_[src, dst], np.r_[dst, src]].T, dtype=torch.long)

    return pyg.data.Data(x=x_by_part, y=y_by_part, edge_index=edge_index, global_idx=torch.tensor(node_ids.astype(int), dtype=torch.long))

def pred_input(x, node_ids, cell_ids, mesh):
    coords = mesh.coordinates()
    num_nodes = len(coords)
    cells = mesh.cells()

    tmp = np.arange(num_nodes)
    tmp[node_ids] = np.arange(len(node_ids))
    x_by_part = torch.tensor(x[node_ids], dtype = torch.float)
    cell_by_part = tmp[cells[cell_ids]]
    T = Triangulation(*coords[node_ids].T, triangles=cell_by_part)
    src, dst = T.edges.T
    edge_index = torch.tensor(np.c_[np.r_[src, dst], np.r_[dst, src]].T, dtype=torch.long)

    return pyg.data.Data(x=x_by_part, edge_index=edge_index, global_idx=torch.tensor(node_ids.astype(int), dtype=torch.long))


class MyGNN(torch.nn.Module):
    def __init__(self, n_input, n_hidden, n_layer, dropout):
        super().__init__()
        
        self.input = pyg.nn.GCNConv(n_input, n_hidden)
        self.input_act = torch.nn.ReLU()

        self.dropout = torch.nn.ModuleList()
        self.hidden = torch.nn.ModuleList()
        self.hidden_act = torch.nn.ModuleList()
        for _ in range(n_layer):
            self.hidden.append(pyg.nn.GCNConv(n_hidden, n_hidden))
            self.dropout.append(torch.nn.Dropout(p=dropout))
            self.hidden_act.append(torch.nn.ReLU())
        self.output = pyg.nn.GCNConv(n_hidden, 1)
        
    def forward(self, x, edge_index):
        x = self.input_act(self.input(x, edge_index))
        for layer, drop, act in zip(self.hidden, self.dropout, self.hidden_act):
            x = layer(x, edge_index) + x
            x = drop(x)
            x = act(x)
        return self.output(x, edge_index)
    
def training(dataset, batch_size, n_hidden, n_layer, lr, epochs, device, net=None):
    dataset_size = len(dataset)
    train_size = int(dataset_size*0.8)
    validataion_size = int(dataset_size-train_size)
    train_dataset, validataion_dataset = random_split(dataset, [train_size, validataion_size])

    train_loader = pyg.loader.DataLoader(train_dataset, batch_size = batch_size)
    validation_loader = pyg.loader.DataLoader(validataion_dataset, batch_size = batch_size)
    if net is None:
        net = MyGNN(4, n_hidden, n_layer, 0.5).to(device)
    optim = torch.optim.Adam(net.parameters(), lr=lr)
    # criterion = torch.nn.MSELoss()
    criterion = torch.nn.L1Loss()
    # def criterion(yhat, y):
    #     umag = torch.norm(yhat[:, 0])
    #     vmag = torch.norm(y[:, 0])
    #     return 1 - torch.sum(yhat*y)/umag/vmag
    train_history = []
    val_history = []
    pbar = tqdm(range(epochs))
    for epoch in pbar:
        net.train()
        running_loss = 0.0
        for batch in train_loader:
            optim.zero_grad()
            yhat = net(batch.x.to(device), batch.edge_index.to(device))
            loss = criterion(yhat, batch.y.to(device))
            loss.backward()
            optim.step()
            running_loss += loss.item()
        train_loss = running_loss/len(train_loader)
        train_history.append(train_loss)
        with torch.no_grad():
            net.eval()
            running_loss = 0.0
            for batch in validation_loader:
                yhat = net(batch.x.to(device), batch.edge_index.to(device))
                loss = criterion(yhat, batch.y.to(device))
                running_loss += loss.item()
        val_loss = running_loss/len(train_loader)
        val_history.append(val_loss)
        pbar.set_postfix_str(f'loss={train_loss:.3e}/{val_loss:.3e}')
    return train_history, val_history, net