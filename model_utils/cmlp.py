import torch.nn as nn
import torch.nn.functional as F

class CategoricalMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout = dropout
        
        self.dims = [self.input_dim]
        self.dims.extend(hidden_dims)
        self.dims.append(output_dim)
        
        self.layers = nn.ModuleList([])
        
        for i in range(len(self.dims) - 1):
            ip_dim = self.dims[i]
            op_dim = self.dims[i+1]
            self.layers.append(
                nn.Linear(ip_dim, op_dim, bias=True)
            )        
            
        self.__init_net_weights__()
        
    def __init_net_weights__(self):
        for m in self.layers:
            m.weight.data.normal_(0.0, 0.1)
            m.bias.data.fill_(0.1)
            
    def forward(self, x, start_layer_idx=0):
        if start_layer_idx < 0:
            return self.mapping(x, start_layer_idx=start_layer_idx)
        x = x.view(-1, self.input_dim)
        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:
                x = layer(x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        out = self.layers[-1](x)
            
        return x, out
    
    def mapping(self, z_input, start_layer_idx=-1):
        z = z_input
        z = self.layers[-1](z)
        out=F.log_softmax(z, dim=1)
        return z, out