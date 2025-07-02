"""
Create a linear layer using the same API as KANLayer so that it can be plotted.
"""
import torch
import torch.nn as nn
import numpy as np
from .spline import *
from .utils import sparse_mask


class LinearLayer(nn.Module):
    """
    LinearLayer class
    

    Attributes:
    -----------
        in_dim: int
            input dimension
        out_dim: int
            output dimension
        mask: tensor in [in_dim, out_dim]
            mask of spline functions. setting some element of the mask to zero means setting the corresponding activation to zero function.
        device: str
            device
    """

    def __init__(self, in_dim=3, out_dim=2, device='cpu'):
        ''''
        initialize a LinearLayer
        
        Args:
        -----
            in_dim : int
                input dimension. Default: 2.
            out_dim : int
                output dimension. Default: 3.
            device : str
                device

        Returns:
        --------
            self
            
        Example
        -------
        >>> from kan.LinearLayer import *
        >>> model = LinearLayer(in_dim=3, out_dim=5)
        >>> (model.in_dim, model.out_dim)
        '''
        super(LinearLayer, self).__init__()
        # size 
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.linear = nn.Linear(in_dim, out_dim)
        self.register_buffer("mask", torch.ones(in_dim, out_dim))

        # Set these as just zeros (they are not used, just adding 
        # in case other methods references them)
        self.register_buffer("coef", torch.zeros((1, 1, 1)))
        self.register_buffer("grid", torch.zeros((1, 1, 1)))
        self.register_buffer("scale_base", torch.zeros((1, 1)))
        self.register_buffer("scale_sp", torch.zeros((1, 1)))
        self.to(device)


    def to(self, device):
        super(LinearLayer, self).to(device)
        self.device = device
        return self


    def forward(self, x):
        '''
        LinearLayer forward given input x
        
        Args:
        -----
            x : 2D torch.float
                inputs, shape (number of samples, input dimension)
            
        Returns:
        --------
            y : 2D torch.float
                outputs, shape (number of samples, output dimension)
            preacts : 3D torch.float
                fan out x into activations, shape (number of samples, output dimension, input dimension)
            postacts : 3D torch.float
                the outputs of activation functions with preacts as inputs
            postspline : 3D torch.float
                the outputs of spline functions with preacts as inputs
        
        Example
        -------
        >>> from kan.KANLayer import *
        >>> model = KANLayer(in_dim=3, out_dim=5)
        >>> x = torch.normal(0,1,size=(100,3))
        >>> y, preacts, postacts, postspline = model(x)
        >>> y.shape, preacts.shape, postacts.shape, postspline.shape
        '''
        batch = x.shape[0]
        preacts = x[:,None,:].clone().expand(batch, self.out_dim, self.in_dim)  # [batch, out_dim, in_dim]
        y_orig = self.linear(x)  # [batch, out_dim]

        # Manually calculate y so that the mask can be applied
        postacts = self.linear.weight * preacts  # weight: [out_dim, in_dim], preacts: [batch, out_dim, in_dim] -> postacts: batch, out_dim, in_dim]
        y = self.mask[None, :, :] * y
        y = y.sum(dim=2) + self.linear.bias  # y: [batch, out_dim]
        assert torch.allclose(y, y_orig+0.5)
        return y, preacts, postacts, postacts


    def get_subset(self, in_id, out_id):
        '''
        get a smaller KANLayer from a larger KANLayer (used for pruning)
        
        Args:
        -----
            in_id : list
                id of selected input neurons
            out_id : list
                id of selected output neurons
            
        Returns:
        --------
            spb : LinearLayer
            
        Example
        -------
        >>> kanlayer_large = LinearLayer(in_dim=10, out_dim=10, num=5, k=3)
        >>> kanlayer_small = kanlayer_large.get_subset([0,9],[1,2,3])
        >>> kanlayer_small.in_dim, kanlayer_small.out_dim
        (2, 3)
        '''
        spb = LinearLayer(len(in_id), len(out_id), device=self.device)  # @joshuafan
        spb.mask.data = self.mask[in_id][:,out_id]

        spb.in_dim = len(in_id)
        spb.out_dim = len(out_id)
        return spb
    
    
    def swap(self, i1, i2, mode='in'):
        '''
        swap the i1 neuron with the i2 neuron in input (if mode == 'in') or output (if mode == 'out') 
        
        Args:
        -----
            i1 : int
            i2 : int
            mode : str
                mode = 'in' or 'out'
            
        Returns:
        --------
            None
            
        Example
        -------
        >>> from kan.KANLayer import *
        >>> model = KANLayer(in_dim=2, out_dim=2, num=5, k=3)
        >>> print(model.coef)
        >>> model.swap(0,1,mode='in')
        >>> print(model.coef)
        '''
        with torch.no_grad():
            def swap_(data, i1, i2, mode='in'):
                if mode == 'in':
                    data[i1], data[i2] = data[i2].clone(), data[i1].clone()
                elif mode == 'out':
                    data[:,i1], data[:,i2] = data[:,i2].clone(), data[:,i1].clone()
            swap_(self.mask.data, i1, i2, mode=mode)
