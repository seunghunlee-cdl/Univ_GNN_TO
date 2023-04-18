# import matplotlib.pyplot as plt
import numpy as np
import torch
import torch_geometric as pyg
from matplotlib.tri import Triangulation
from torch.utils.data import random_split
from torch_geometric.data import Data
from torch_geometric.utils import subgraph
from tqdm.auto import tqdm

from utils import convert_neighors_to_edges


def generate_data(x, y, edge_ids, elem_ids, mesh):
    x_by_part = torch.tensor(x[elem_ids], dtype = torch.float)
    y_by_part = torch.tensor(y[elem_ids], dtype = torch.float)
    return pyg.data.Data(x=x_by_part, y=y_by_part, edge_index=edge_ids.edge_index, global_idx=torch.tensor(elem_ids.astype(int),dtype=torch.long))

def pred_input(x, edge_ids, elem_ids, mesh):
    x_by_part = torch.tensor(x[elem_ids], dtype = torch.float)
    return pyg.data.Data(x=x_by_part, edge_index=edge_ids.edge_index, global_idx=torch.tensor(elem_ids.astype(int), dtype=torch.long))


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
        self.output_act = torch.nn.ReLU()
        
    def forward(self, x, edge_index):
        x = self.input_act(self.input(x, edge_index))
        for layer, drop, act in zip(self.hidden, self.dropout, self.hidden_act):
            x = layer(x, edge_index) + x
            x = drop(x)
            x = act(x)
        x = self.output_act(self.output(x, edge_index))
        return -x
    
def training(dataset, batch_size, n_hidden, n_layer, lr, epochs, device, net=None):
    dataset_size = len(dataset)
    train_size = int(dataset_size*0.8)
    validation_size = int(dataset_size-train_size)
    train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])

    train_loader = pyg.loader.DataLoader(train_dataset, batch_size = batch_size)
    validation_loader = pyg.loader.DataLoader(validation_dataset, batch_size = batch_size)
    if net is None:
        net = MyGNN(4, n_hidden, n_layer, 0.5).to(device)
    optim = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = torch.nn.L1Loss()

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

def partition_graph(subset, data):
    if not isinstance(subset, torch.Tensor):
        subset = torch.tensor(subset, dtype=torch.long)
    dummy = torch.zeros(data.num_nodes, dtype=torch.long)
    edge_index_, _ = subgraph(subset, data.edge_index)
    dummy[subset] = torch.arange(len(subset))
    return Data(
        x=data.x[subset],
        edge_index=dummy[edge_index_]
    )

def graph_partitioning(coords, trias, part_info, center):
    T = Triangulation(*coords.T, triangles=trias)
    edge_index = np.concatenate([convert_neighors_to_edges(eid, neighbors) for eid, neighbors in enumerate(T.neighbors)]).T
    global_graph = Data(
        x=torch.tensor(center), 
        edge_index=torch.tensor(edge_index, dtype=torch.long)
    )
    partitioned_graphs = [partition_graph(subset, global_graph) for subset in part_info['elems']]
    return partitioned_graphs