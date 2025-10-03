import torch
import torch.nn as nn
import numpy as np
from .spline import *
from .utils import sparse_mask

class KANLayer(nn.Module):
    """
    KANLayer class
    

    Attributes:
    -----------
        in_dim: int
            input dimension
        out_dim: int
            output dimension
        num: int
            the number of grid intervals
        k: int
            the piecewise polynomial order of splines
        noise_scale: float
            spline scale at initialization
        coef: 2D torch.tensor
            coefficients of B-spline bases
        scale_base_mu: float
            magnitude of the residual function b(x) is drawn from N(mu, sigma^2), mu = sigma_base_mu
        scale_base_sigma: float
            magnitude of the residual function b(x) is drawn from N(mu, sigma^2), mu = sigma_base_sigma
        scale_sp: float
            mangitude of the spline function spline(x)
        base_fun: fun
            residual function b(x)
        mask: 1D torch.float
            mask of spline functions. setting some element of the mask to zero means setting the corresponding activation to zero function.
        grid_eps: float in [0,1]
            a hyperparameter used in update_grid_from_samples. When grid_eps = 1, the grid is uniform; when grid_eps = 0, the grid is partitioned using percentiles of samples. 0 < grid_eps < 1 interpolates between the two extremes.
            the id of activation functions that are locked
        grid_margin: float
            a hyperparameter used in update_grid_from_samples; applies for UNIFORM grid only. How much of a margin to leave around the data points in x, given in units of x-range (x.max() - x.min()).
            In original KAN repo, this was set to 0. If set to 1, and x originally ranged from [0, 1], the grid would range between [-1, 2] (EXCLUDING the extended points).
        device: str
            device
    """

    def __init__(self, in_dim=3, out_dim=2, num=5, k=3, noise_scale=0.5, scale_base_mu=0.0, scale_base_sigma=1.0, scale_sp=1.0, base_fun=torch.nn.SiLU(), grid_eps=0.02, grid_margin=0.0, grid_range=[-1, 1], sp_trainable=True, sb_trainable=True, save_plot_data = True, device='cpu', sparse_init=False,
                 drop_rate=0.0, drop_mode='postact', drop_scale=True, batch_norm_spline=False):
        ''''
        initialize a KANLayer
        
        Args:
        -----
            in_dim : int
                input dimension. Default: 2.
            out_dim : int
                output dimension. Default: 3.
            num : int
                the number of grid intervals = G. Default: 5.
            k : int
                the order of piecewise polynomial. Default: 3.
            noise_scale : float
                the scale of noise injected at initialization. Default: 0.1.
            scale_base_mu : float
                the scale of the residual function b(x) is intialized to be N(scale_base_mu, scale_base_sigma^2).
            scale_base_sigma : float
                the scale of the residual function b(x) is intialized to be N(scale_base_mu, scale_base_sigma^2).
            scale_sp : float
                the scale of the base function spline(x).
            base_fun : function
                residual function b(x). Default: torch.nn.SiLU()
            grid_eps : float
                When grid_eps = 1, the grid is uniform; when grid_eps = 0, the grid is partitioned using percentiles of samples. 0 < grid_eps < 1 interpolates between the two extremes.
            grid_range : list/np.array of shape (2,)
                setting the range of grids. Default: [-1,1].
            sp_trainable : bool
                If true, scale_sp is trainable
            sb_trainable : bool
                If true, scale_base is trainable
            device : str
                device
            sparse_init : bool
                if sparse_init = True, sparse initialization is applied.

            DropKAN related (see https://github.com/Ghaith81/dropkan/blob/master/dropkan/DropKAN.py)
                drop_rate: list
                    A list of floats indicating the rates of drop for the DropKAN mask. Default: 0.0.
                drop_mode: str
                    Accept the following values 'postspline' the drop mask is applied to the layer's postsplines, 'postact' the drop mask is applied to the layer's postacts, 'dropout' applies a standard dropout layer to the inputs. Default: 'postact'.
                drop_scale: bool
                    If true, the retained postsplines/postacts are scaled by a factor of 1/(1-drop_rate). Default: True
            batch_norm_spline : bool
                If true, batch-normalize the output of each (input-output) spline.

        Returns:
        --------
            self
            
        Example
        -------
        >>> from kan.KANLayer import *
        >>> model = KANLayer(in_dim=3, out_dim=5)
        >>> (model.in_dim, model.out_dim)
        '''
        super(KANLayer, self).__init__()
        # size 
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.num = num
        self.k = k

        grid = torch.linspace(grid_range[0], grid_range[1], steps=num + 1, device=device)[None,:].expand(self.in_dim, num+1)  # [in_dim, G+1]
        grid = extend_grid(grid, k_extend=k)  # [in_dim, G+2k+1]
        # self.grid = torch.nn.Parameter(grid).requires_grad_(False)
        self.register_buffer("grid", grid)  # @joshuafan NOTE: Changed grid to a buffer
        noises = (torch.rand(self.num+1, self.in_dim, self.out_dim, device=device) - 1/2) * noise_scale / num  # [G+1, in_dim, out_dim]
        # print("Noises", noise_scale, noises)
        self.coef = torch.nn.Parameter(curve2coef(self.grid[:,k:-k].permute(1,0), noises, self.grid, k))  # [in_dim, out_dim, G+k]
        # print("Grid", self.grid.shape, self.grid[0, :])
        # print("Noise", noises.shape, noises[:, 0, 0])
        # print("Coefs", self.coef.shape, self.coef[0, 0, :])

        if sparse_init:
            self.register_buffer("mask", sparse_mask(in_dim, out_dim))
            # self.mask = torch.nn.Parameter(sparse_mask(in_dim, out_dim)).requires_grad_(False)
        else:
            self.register_buffer("mask", torch.ones(in_dim, out_dim))
            # self.mask = torch.nn.Parameter(torch.ones(in_dim, out_dim)).requires_grad_(False)

        # Batch-normalized spline
        self.batch_norm_spline = batch_norm_spline
        if batch_norm_spline:
            self.norm = nn.BatchNorm1d(self.in_dim * self.out_dim)

        self.scale_base = torch.nn.Parameter(scale_base_mu * 1 / np.sqrt(in_dim) + \
                         scale_base_sigma * (torch.rand(in_dim, out_dim)*2-1) * 1/np.sqrt(in_dim)).requires_grad_(sb_trainable)
        self.scale_sp = torch.nn.Parameter(torch.ones(in_dim, out_dim) * scale_sp * 1 / np.sqrt(in_dim) * self.mask).requires_grad_(sp_trainable)  # make scale trainable
        self.base_fun = base_fun

        self.grid_eps = grid_eps
        self.grid_margin = grid_margin
        self.sb_trainable = sb_trainable
        self.sp_trainable = sp_trainable
        self.drop_rate = drop_rate
        self.drop_mode = drop_mode
        self.drop_scale = drop_scale
        self.to(device)


    def to(self, device):
        super(KANLayer, self).to(device)
        self.device = device
        return self

    def forward(self, x):
        '''
        KANLayer forward given input x
        
        Args:
        -----
            x : 2D torch.float
                inputs, shape (number of samples, input dimension)
            
        Returns:
        --------
            y : 2D torch.float
                outputs, shape (number of samples, output dimension)
            preacts : 3D torch.float
                fan out x into activations, shape (number of sampels, output dimension, input dimension)
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

        # Standard (node-based) dropout. https://github.com/Ghaith81/dropkan/blob/master/dropkan/DropKANLayer.py
        if self.training:
            if self.drop_mode == 'dropout' and self.drop_rate > 0 and self.drop_scale:
                #print('dropout with scale')
                mask = torch.empty(x.shape, device=x.device).bernoulli_(1 - self.drop_rate)
                x = x * mask / (1 - self.drop_rate)
            elif self.drop_mode == 'dropout' and self.drop_rate > 0 and not self.drop_scale:
                mask = torch.empty(x.shape, device=x.device).bernoulli_(1 - self.drop_rate)
                x = x * mask

        preacts = x[:,None,:].clone().expand(batch, self.out_dim, self.in_dim)

        base = self.base_fun(x) # (batch, in_dim)
        y = coef2curve(x_eval=x, grid=self.grid, coef=self.coef, k=self.k)  # y: (batch, in_dim, other_out_dim)

        # Batch-normalization of post-splines
        if self.batch_norm_spline:
            y = y.reshape(y.shape[0], -1)  # rearrange(y, ('b i o -> b (i o)'))
            y = self.norm(y) / np.sqrt(self.in_dim)
            y = y.reshape(y.shape[0], self.in_dim, self.out_dim)  # rearrange(y, ('b (i o) -> b i o'), i=self.in_dim)

        postspline = y.clone().permute(0,2,1)

        # Post-spline dropout (excluding the base function). https://github.com/Ghaith81/dropkan/blob/master/dropkan/DropKANLayer.py
        if self.training:
            if self.drop_mode == 'postspline' and self.drop_rate > 0 and self.drop_scale:
                mask = torch.empty(y.shape, device=y.device).bernoulli_(1 - self.drop_rate)
                y = y * mask / (1 - self.drop_rate)
            elif self.drop_mode == 'postspline' and self.drop_rate > 0 and not self.drop_scale:
                mask = torch.empty(y.shape, device=y.device).bernoulli_(1 - self.drop_rate)
                y = y * mask

        y = self.scale_base[None,:,:] * base[:,:,None] + self.scale_sp[None,:,:] * y
        y = self.mask[None,:,:] * y  # [batch, in_dim, out_dim]

        postacts = y.clone().permute(0,2,1)  # [batch, out_dim, in_dim]
        # print("postacts dist", postacts.mean(dim=0), postacts.std(dim=0))
        # print("postacts dist per output", postacts.mean(dim=(0, 2)), postacts.std(dim=(0, 2)))

        # Post-activation dropout (including base function). https://github.com/Ghaith81/dropkan/blob/master/dropkan/DropKANLayer.py
        if self.training:
            if self.drop_mode == 'postact' and self.drop_rate > 0 and self.drop_scale:
                mask = torch.empty(y.shape, device=y.device).bernoulli_(1 - self.drop_rate)
                y = y * mask / (1 - self.drop_rate)
            elif self.drop_mode == 'postact' and self.drop_rate > 0 and not self.drop_scale:
                mask = torch.empty(y.shape, device=y.device).bernoulli_(1 - self.drop_rate)
                y = y * mask
            if self.drop_mode == 'postact_input' and self.drop_rate > 0:
                mask = torch.empty((y.shape[0], y.shape[1], 1), device=y.device).bernoulli_(1 - self.drop_rate).repeat(1, 1, y.shape[2])
                y = y * mask
                if self.drop_scale:
                    y = y / (1 - self.drop_rate)

        y = torch.sum(y, dim=1)
        # print("Dist of KANLayer out", y.mean(dim=0), y.std(dim=0))
        return y, preacts, postacts, postspline

    def update_grid_from_samples(self, x, mode='sample'):
        '''
        update grid from samples
        
        Args:
        -----
            x : 2D torch.float
                inputs, shape (number of samples, input dimension)

        Returns:
        --------
            None
        
        Example
        -------
        >>> model = KANLayer(in_dim=1, out_dim=1, num=5, k=3)
        >>> print(model.grid.data)
        >>> x = torch.linspace(-3,3,steps=100)[:,None]
        >>> model.update_grid_from_samples(x)
        >>> print(model.grid.data)
        '''
        
        batch = x.shape[0]
        #x = torch.einsum('ij,k->ikj', x, torch.ones(self.out_dim, ).to(self.device)).reshape(batch, self.size).permute(1, 0)
        x_pos = torch.sort(x, dim=0)[0]
        # print("X pos", x_pos[0])
        # print("Grid", self.grid.shape, self.grid[0])
        # print("Coef", self.coef.shape, self.coef[0, 0])
        y_eval = coef2curve(x_pos, self.grid, self.coef, self.k)
        # print("Y eval", y_eval[0])
        num_interval = self.grid.shape[1] - 1 - 2*self.k

        def get_grid(num_interval):
            ids = [int(batch / num_interval * i) for i in range(num_interval)] + [-1]
            grid_adaptive = x_pos[ids, :].permute(1,0)
            # margin = 0.00
            margin = 0.01 + self.grid_margin * (grid_adaptive[:,[-1]] - grid_adaptive[:,[0]]) # NOTE TODO @joshuafan Trying a wider margin
            h = (grid_adaptive[:,[-1]] - grid_adaptive[:,[0]] + 2 * margin)/num_interval
            grid_uniform = grid_adaptive[:,[0]] - margin + h * torch.arange(num_interval+1,)[None, :].to(x.device)
            grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
            return grid

        
        grid = get_grid(num_interval)
        
        if mode == 'grid':
            sample_grid = get_grid(2*num_interval)
            x_pos = sample_grid.permute(1,0)
            y_eval = coef2curve(x_pos, self.grid, self.coef, self.k)
        
        # print("Grid before", self.grid.shape, self.grid[0])
        self.grid.data = extend_grid(grid, k_extend=self.k)
        # print("Grid after", self.grid[0])
        #print('x_pos 2', x_pos.shape)
        #print('y_eval 2', y_eval.shape)
        # print("Update grid before. Coef", self.coef[0,0], "X", x_pos[0], "Y", y_eval[0,0])
        self.coef.data = curve2coef(x_pos, y_eval, self.grid, self.k)
        # print("Update grid after. Coef", self.coef[0,0])


    def initialize_grid_from_parent(self, parent, x, mode='sample'):
        '''
        update grid from a parent KANLayer & samples
        
        Args:
        -----
            parent : KANLayer
                a parent KANLayer (whose grid is usually coarser than the current model)
            x : 2D torch.float
                inputs, shape (number of samples, input dimension)
            
        Returns:
        --------
            None
          
        Example
        -------
        >>> batch = 100
        >>> parent_model = KANLayer(in_dim=1, out_dim=1, num=5, k=3)
        >>> print(parent_model.grid.data)
        >>> model = KANLayer(in_dim=1, out_dim=1, num=10, k=3)
        >>> x = torch.normal(0,1,size=(batch, 1))
        >>> model.initialize_grid_from_parent(parent_model, x)
        >>> print(model.grid.data)
        '''
        
        batch = x.shape[0]
        
        # shrink grid
        x_pos = torch.sort(x, dim=0)[0]
        y_eval = coef2curve(x_pos, parent.grid, parent.coef, parent.k)
        num_interval = self.grid.shape[1] - 1 - 2*self.k
        

        '''
        # based on samples
        def get_grid(num_interval):
            ids = [int(batch / num_interval * i) for i in range(num_interval)] + [-1]
            grid_adaptive = x_pos[ids, :].permute(1,0)
            h = (grid_adaptive[:,[-1]] - grid_adaptive[:,[0]])/num_interval
            grid_uniform = grid_adaptive[:,[0]] + h * torch.arange(num_interval+1,)[None, :].to(x.device)
            grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
            return grid'''
        
        #print('p', parent.grid)
        # based on interpolating parent grid
        def get_grid(num_interval):
            x_pos = parent.grid[:,parent.k:-parent.k]
            #print('x_pos', x_pos)
            sp2 = KANLayer(in_dim=1, out_dim=self.in_dim,k=1,num=x_pos.shape[1]-1,scale_base_mu=0.0, scale_base_sigma=0.0, device=x.device)

            #print('sp2_grid', sp2.grid[:,sp2.k:-sp2.k].permute(1,0).expand(-1,self.in_dim))
            #print('sp2_coef_shape', sp2.coef.shape)
            sp2_coef = curve2coef(sp2.grid[:,sp2.k:-sp2.k].permute(1,0).expand(-1,self.in_dim), x_pos.permute(1,0).unsqueeze(dim=2), sp2.grid[:,:], k=1).permute(1,0,2)
            shp = sp2_coef.shape
            #sp2_coef = torch.cat([torch.zeros(shp[0], shp[1], 1), sp2_coef, torch.zeros(shp[0], shp[1], 1)], dim=2)
            #print('sp2_coef',sp2_coef)
            #print(sp2.coef.shape)
            sp2.coef.data = sp2_coef
            percentile = torch.linspace(-1,1,self.num+1).to(self.device)
            grid = sp2(percentile.unsqueeze(dim=1))[0].permute(1,0)
            #print('c', grid)
            return grid
        
        grid = get_grid(num_interval)
        
        if mode == 'grid':
            sample_grid = get_grid(2*num_interval)
            x_pos = sample_grid.permute(1,0)
            y_eval = coef2curve(x_pos, parent.grid, parent.coef, parent.k)
        
        grid = extend_grid(grid, k_extend=self.k)
        self.grid.data = grid
        self.coef.data = curve2coef(x_pos, y_eval, self.grid, self.k)

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
            spb : KANLayer
            
        Example
        -------
        >>> kanlayer_large = KANLayer(in_dim=10, out_dim=10, num=5, k=3)
        >>> kanlayer_small = kanlayer_large.get_subset([0,9],[1,2,3])
        >>> kanlayer_small.in_dim, kanlayer_small.out_dim
        (2, 3)
        '''
        spb = KANLayer(len(in_id), len(out_id), self.num, self.k, base_fun=self.base_fun, device=self.device,
                       grid_eps=self.grid_eps, grid_margin=self.grid_margin,
                       sp_trainable=self.sp_trainable,sb_trainable=self.sb_trainable,
                       drop_rate=self.drop_rate, drop_mode=self.drop_mode, drop_scale=self.drop_scale, batch_norm_spline=self.batch_norm_spline)  # @joshuafan
        spb.grid.data = self.grid[in_id]
        spb.coef.data = self.coef[in_id][:,out_id]
        spb.scale_base.data = self.scale_base[in_id][:,out_id]
        spb.scale_sp.data = self.scale_sp[in_id][:,out_id]
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

            if mode == 'in':
                swap_(self.grid.data, i1, i2, mode='in')
            swap_(self.coef.data, i1, i2, mode=mode)
            swap_(self.scale_base.data, i1, i2, mode=mode)
            swap_(self.scale_sp.data, i1, i2, mode=mode)
            swap_(self.mask.data, i1, i2, mode=mode)
