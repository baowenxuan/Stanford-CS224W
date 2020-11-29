import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils

class GNNStack(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, args, task='node'):
        super(GNNStack, self).__init__()
        conv_model = self.build_conv_model(args.model_type)
        self.convs = nn.ModuleList()
        self.convs.append(conv_model(input_dim, hidden_dim))
        assert (args.num_layers >= 1), 'Number of layers is not >=1'
        for l in range(args.num_layers-1):
            self.convs.append(conv_model(hidden_dim, hidden_dim))

        # post-message-passing
        self.post_mp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Dropout(args.dropout), 
            nn.Linear(hidden_dim, output_dim))

        self.task = task
        if not (self.task == 'node' or self.task == 'graph'):
            raise RuntimeError('Unknown task.')

        self.dropout = args.dropout
        self.num_layers = args.num_layers

    def build_conv_model(self, model_type):
        if model_type == 'GCN':
            return pyg_nn.GCNConv
        elif model_type == 'myGCN':
            return GCN
        elif model_type == 'GraphSage':
            return GraphSage
        elif model_type == 'GAT':
            return GAT

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        ############################################################################
        # TODO: Your code here! 
        # Each layer in GNN should consist of a convolution (specified in model_type),
        # a non-linearity (use RELU), and dropout. 
        # HINT: the __init__ function contains parameters you will need. You may 
        # also find pyg_nn.global_max_pool useful for graph classification.
        # Our implementation is ~6 lines, but don't worry if you deviate from this.
        
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
        if self.task == 'graph':
            x = pyg_nn.global_add_pool(x, batch)

        ############################################################################

        x = self.post_mp(x)

        return F.log_softmax(x, dim=1)

    def loss(self, pred, label):
        return F.nll_loss(pred, label)
    
    
class GCN(pyg_nn.MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCN, self).__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = pyg_utils.add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = pyg_utils.degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j


class GraphSage(pyg_nn.MessagePassing):
    """Non-minibatch version of GraphSage."""
    def __init__(self, in_channels, out_channels, reducer='mean', 
                 normalize_embedding=True):
        super(GraphSage, self).__init__(aggr='mean')

        ############################################################################
        # TODO: Your code here! 
        # Define the layers needed for the forward function. 
        # Our implementation is ~2 lines, but don't worry if you deviate from this.

        self.lin = nn.Linear(in_channels + out_channels, out_channels)
        self.agg_lin = nn.Linear(in_channels, out_channels)

        ############################################################################

        self.normalize_emb = normalize_embedding

    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        ############################################################################
        # TODO: Your code here! 
        # Given x, perform the aggregation and pass it through a MLP with skip-
        # connection. Place the result in out. 
        # HINT: It may be useful to read the pyg_nn implementation of GCNConv,
        # https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html
        # Our implementation is ~4 lines, but don't worry if you deviate from this.
        
        agg_x = self.agg_lin(x)
        agg_x = F.relu(agg_x)
        agg_x = self.propagate(edge_index, size=(num_nodes, num_nodes), x=agg_x)
        out = self.lin(torch.cat((x, agg_x), 1))
        if self.normalize_emb:
            out = F.normalize(out, p=2, dim=1)
            
        ############################################################################
        return out

#     def message(self, x_j, edge_index, size):
#         return x_j


class GAT(pyg_nn.MessagePassing):

    def __init__(self, in_channels, out_channels, num_heads=1, concat=True,
                 dropout=0, bias=True, **kwargs):
        super(GAT, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = num_heads
        self.concat = concat 
        self.dropout = dropout

        ############################################################################
        #  TODO: Your code here!
        # Define the layers needed for the forward function. 
        # Remember that the shape of the output depends the number of heads.
        # Our implementation is ~1 line, but don't worry if you deviate from this.

        self.lin = nn.Linear(in_channels, num_heads * out_channels)

        ############################################################################

        ############################################################################
        #  TODO: Your code here!
        # The attention mechanism is a single feed-forward neural network parametrized
        # by weight vector self.att. Define the nn.Parameter needed for the attention
        # mechanism here. Remember to consider number of heads for dimension!
        # Our implementation is ~1 line, but don't worry if you deviate from this.

        self.att = nn.Parameter(torch.Tensor(num_heads, 2 * out_channels))

        ############################################################################

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(self.heads * out_channels))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        nn.init.xavier_uniform_(self.att)
        nn.init.zeros_(self.bias)

        ############################################################################

    def forward(self, x, edge_index, size=None):
        ############################################################################
        #  TODO: Your code here!
        # Apply your linear transformation to the node feature matrix before starting
        # to propagate messages.
        # Our implementation is ~1 line, but don't worry if you deviate from this.
        
        x = self.lin(x)
        ############################################################################

        # Start propagating messages.
        return self.propagate(edge_index, size=size, x=x)

    def message(self, edge_index_i, x_i, x_j, size_i):
        #  Constructs messages to node i for each edge (j, i).

        ############################################################################
        #  TODO: Your code here! Compute the attention coefficients alpha as described
        # in equation (7). Remember to be careful of the number of heads with 
        # dimension!
        # Our implementation is ~5 lines, but don't worry if you deviate from this.
        
        x_i = x_i.view(-1, self.heads, self.out_channels)
        x_j = x_j.view(-1, self.heads, self.out_channels)
        x_ij = torch.cat([x_i, x_j], dim=2)
        alpha = torch.mul(self.att, x_ij)  # shape (E, num_heads, 2 * out_channels)
        alpha = torch.sum(alpha, dim=2, keepdim=True)  # shape (E, num_heads, 1)
        
        alpha = F.leaky_relu(alpha, negative_slope=0.2)
        alpha = pyg_utils.softmax(alpha, edge_index_i, num_nodes=size_i)

        ############################################################################

        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return (x_j * alpha).view(-1, self.heads * self.out_channels)

    def update(self, aggr_out):
        # Updates node embedings.
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.view(-1, self.heads, self.out_channels)
            aggr_out = aggr_out.mean(dim=1)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out
