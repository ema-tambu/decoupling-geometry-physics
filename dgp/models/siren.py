"""
Shift-modulated SIREN (implicit neural representation) with residual connections.

The first layer embeds the coordinates, h_0 = sin(omega_0 (W_in x + b_in)); each hidden layer is shifted by a linear
function of the global parameters p, h_i = sin(omega (W_i h_{i-1} + b_i + sigma(W_mod,i p + b_mod,i))) + h_{i-1},
and a linear layer gives the output. omega and omega_0 are learnable. The input of forward() is
[p, x] (parameters first, then coordinates).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ShiftModulatedMLP(nn.Module):
    def __init__(self, 
                 in_features:       int = 2,
                 shift_size:        int = 6,
                 out_features:      int = 1,
                 num_hidden_layers: int = 3,
                 hidden_features:   list[int] = [128, 128, 128],
                 omega:             float = 30.0,
                 omega_0:           float = 30.0,
                 activation_fn:     str = "sin",
                 dropout_rate:      float = 0.0,
                 use_resnet:        bool = False,
                 mod_activation_fn: str = "identity"
                 ):
        super(ShiftModulatedMLP, self).__init__()

        self.in_features = in_features
        self.shift_size = shift_size

        self.input_layer_siren = nn.Linear(in_features, hidden_features[0])
        
        self.hidden_layers_siren = nn.ModuleList([
            nn.Linear(hidden_features[i], hidden_features[i + 1]) 
            for i in range(num_hidden_layers - 1)
        ])
        self.hidden_layers_modulation = nn.ModuleList([
            nn.Linear(shift_size, hidden_features[i]) 
            for i in range(num_hidden_layers - 1)
        ])

        self.output_layer_siren = nn.Linear(hidden_features[-1], out_features)
        self.omega = nn.Parameter(torch.tensor(omega))
        self.omega_0 = nn.Parameter(torch.tensor(omega_0))

        # Dropout
        self.dropout_rate = dropout_rate
        self.dropout_layer = nn.Dropout(p=self.dropout_rate)

        self.use_resnet = use_resnet

        # activation functions
        self.activation_dict = {
            "relu": F.relu,
            "sigmoid": F.sigmoid,
            "tanh": F.tanh,
            "leaky_relu": F.leaky_relu,
            "sin": torch.sin,
            "silu": F.silu,
            "identity": nn.Identity()
        }
        if activation_fn not in self.activation_dict:
            raise ValueError(f"Unsupported activation function: {activation_fn}")
        self.activation_fn = self.activation_dict[activation_fn]

        if mod_activation_fn not in self.activation_dict:
            raise ValueError(f"Unsupported modulation activation function: {mod_activation_fn}")
        self.mod_activation_fn = self.activation_dict[mod_activation_fn]

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            # first layer: U(-1/fan_in, 1/fan_in)
            self.input_layer_siren.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            omega_0 = self.omega.item() if isinstance(self.omega, nn.Parameter) else self.omega
            # hidden layers: Uniform(-sqrt(6/fan_in)/omega_0, sqrt(6/fan_in)/omega_0)
            for layer in self.hidden_layers_siren:
                fan_in = layer.in_features
                bound = np.sqrt(6 / fan_in) / omega_0
                layer.weight.uniform_(-bound, bound)
            # output layer
            fan_in = self.output_layer_siren.in_features
            bound = np.sqrt(6 / fan_in) / omega_0
            self.output_layer_siren.weight.uniform_(-bound, bound)

    def forward(self, input):

        # input has shape (batch_size, params_features + in_features)
        params = input[:, :self.shift_size]  # shape (batch_size, shift_size)
        x = input[:, self.shift_size:]       # shape (batch_size, in_features)

        # input layer, no shift modulation
        x = self.activation_fn(self.omega_0 * self.input_layer_siren(x))
        x = self.dropout_layer(x)

        h_prev = x
        # hidden layers with shift modulation
        for i in range(len(self.hidden_layers_siren)):
            
            # shift modulation layer
            layer_mod = self.hidden_layers_modulation[i](params)
            # apply independent activation to modulation
            layer_mod = self.mod_activation_fn(layer_mod)
            # linear layer siren
            layer_linear = self.hidden_layers_siren[i](h_prev)
            # complete shift modulated siren layer
            h_next = self.activation_fn(self.omega * (layer_linear + layer_mod))
            # skip connection IF specified (ResNet)
            if self.use_resnet: h_next = h_next + h_prev
            # dropout (only during training)
            h_next = self.dropout_layer(h_next)
            # update for next layer
            h_prev = h_next

        # output layer linear, no shift modulation
        return self.output_layer_siren(h_prev)