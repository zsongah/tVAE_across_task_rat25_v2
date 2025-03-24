import torch.nn as nn

class MLP(nn.Module):
    '''
    MLP Module for DFINE encoder and decoder in addition to the mapper to behavior for supervised DFINE. 
    Encoder encodes the high-dimensional neural observations into low-dimensional manifold latent factors space 
    and decoder decodes the manifold latent factors into high-dimensional neural observations.
    '''

    def __init__(self, **kwargs):
        '''
        Initializer for an Encoder/Decoder/Mapper object. Note that Encoder/Decoder/Mapper is a subclass of torch.nn.Module.

        Parameters
        ------------
        input_dim: int, Dimensionality of inputs to the MLP, default None
        output_dim: int, Dimensionality of outputs of the MLP , default None
        layer_list: list, List of number of neurons in each hidden layer, default None
        kernel_initializer_fn: torch.nn.init, Hidden layer weight initialization function, default nn.init.xavier_normal_
        activation_fn: torch.nn, Activation function of neurons, default nn.Tanh
        '''
        
        super(MLP, self).__init__()
        self.input_dim = kwargs.get('input_dim', None)
        self.output_dim = kwargs.get('output_dim', None)
        self.layer_list = kwargs.get('layer_list', None)
        self.kernel_initializer_fn = kwargs.get('kernel_initializer_fn', nn.init.xavier_normal_)
        self.activation_fn = kwargs.get('activation_fn', nn.Tanh)

        # Create the ModuleList to stack the hidden layers
        self.layers = nn.ModuleList()

        current_dim = self.input_dim
        for i, dim in enumerate(self.layer_list):
            self.layers.append(nn.Linear(current_dim, dim))
            self.kernel_initializer_fn(self.layers[i].weight)
            current_dim = dim

        # Add the output layer
        self.out_layer = nn.Linear(current_dim, self.output_dim)
        self.kernel_initializer_fn(self.layers[-1].weight)


    def forward(self, inp):    
        '''
        Forward pass function for MLP Module 

        Parameters: 
        ------------
        inp: torch.Tensor, shape: (num_seq * num_steps, input_dim), Flattened batch of inputs

        Returns: 
        ------------
        out: torch.Tensor, shape: (num_seq * num_steps, output_dim),Flattened batch of outputs
        '''

        for layer in self.layers:
            inp = layer(inp)
            inp = self.activation_fn(inp)

        out = self.out_layer(inp)

        return out

        

