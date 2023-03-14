# import matplotlib.pyplot as plt
import numpy as np
import torch
import torch_geometric as pyg
from matplotlib.tri import Triangulation
from tqdm.auto import tqdm


def generate_data(x, y, node_ids, cell_ids, mesh):
    coords = mesh.coordinates()
    num_nodes = len(coords)
    cells = mesh.cells()

    tmp = np.arange(num_nodes)
    tmp[node_ids] = np.arange(len(node_ids))
    x_by_part = torch.tensor(x[node_ids], dtype=torch.float)
    y_by_part = torch.tensor(y[node_ids], dtype=torch.float)
    cell_by_part = tmp[cells[cell_ids]]

    T = Triangulation(*coords[node_ids].T, triangles=cell_by_part)
    src, dst = T.edges.T
    edge_index = torch.tensor(np.c_[np.r_[src, dst], np.r_[dst, src]].T, dtype=torch.long)

    return pyg.data.Data(x=x_by_part, y=y_by_part, edge_index=edge_index, global_idx=torch.tensor(node_ids.astype(int), dtype=torch.long))

class MyGNN(torch.nn.Module):
    def __init__(self, n_hidden, n_layer):
        super().__init__()
        
        self.input = pyg.nn.GCNConv(4, n_hidden)
        self.hidden = torch.nn.ModuleList()
        for _ in range(n_layer):
            self.hidden.append(pyg.nn.GCNConv(n_hidden, n_hidden))
        self.output = pyg.nn.GCNConv(n_hidden, 1)
        
    def forward(self, x, edge_index):
        x = torch.relu(self.input(x, edge_index))
        for layer in self.hidden:
            x = torch.relu(layer(x, edge_index))
        return self.output(x, edge_index)
    
def training(data_list, batch_size, n_hidden, n_layer, lr, epochs, device):
    loader = pyg.loader.DataLoader(data_list, batch_size = batch_size)
    net = MyGNN(n_hidden, n_layer).to(device)
    optim = torch.optim.Adam(net.parameters(), lr = lr)
    criterion = torch.nn.MSELoss()
    net.train()
    loss_history = []
    pbar = tqdm(range(epochs))
    for epoch in pbar:
        running_loss = 0.0
        for batch in loader:
            optim.zero_grad()
            yhat = net(batch.x.to(device), batch.edge_index.to(device))
            loss = criterion(yhat, batch.y.to(device))
            loss.backward()
            optim.step()
            running_loss += loss.item()
        avg_loss = running_loss/len(loader)
        loss_history.append(avg_loss)
        pbar.set_postfix_str(f'loss={avg_loss:.3e}')
    return loss_history, loader, net