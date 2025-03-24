import torch
import torch.nn as nn
import pdb

class LDM(nn.Module):
    '''
    Linear Dynamical Model backbone for DFINE. This module is used for smoothing and filtering
    given a batch of trials/segments/time-series. 

    LDM equations are as follows:
    x_{t+1} = Ax_{t} + w_{t}; cov(w_{t}) = W
    a_{t} = Cx_{t} + r_{t}; cov(r_{t}) = R
    '''
    def __init__(self, **kwargs):
        # 注意torch device的问题，参数传进来的时候就应该有device属性。否则就是cpu。这个属性应该和a的一致。
        '''
        Initializer for an LDM object. Note that LDM is a subclass of torch.nn.Module.

        Parameters
        ------------
        - dim_x: int, Dimensionality of dynamic latent factors, default None  
        - dim_a: int, Dimensionality of manifold latent factors, default None
        - is_W_trainable: bool, Whether dynamics noise covariance matrix (W) is learnt or not, default True
        - is_R_trainable: bool, Whether observation noise covariance matrix (R) is learnt or not, default True
        - A: torch.Tensor, shape: (self.dim_x, self.dim_x), State transition matrix of LDM, default identity
        - C: torch.Tensor, shape: (self.dim_a, self.dim_x), Observation matrix of LDM, default identity
        - mu_0: torch.Tensor, shape: (self.dim_x, ), Dynamic latent factor estimate initial condition (x_{0|-1}) for Kalman filtering, default zeros 
        - Lambda_0: torch.Tensor, shape: (self.dim_x, self.dim_x), Dynamic latent factor estimate error covariance initial condition (P_{0|-1}) for Kalman Filtering, default identity
        - W_log_diag: torch.Tensor, shape: (self.dim_x, ), Log-diagonal of process noise covariance matrix (W, therefore it is diagonal and PSD), default ones
        - R_log_diag: torch.Tensor, shape: (self.dim_a, ), Log-diagonal of observation noise covariance matrix  (R, therefore it is diagonal and PSD), default ones
        '''
        super(LDM, self).__init__()
        self.device = kwargs.pop('device', torch.device('cpu'))
        self.dim_x = kwargs.pop('dim_x', None)
        self.dim_a = kwargs.pop('dim_a',None)

        self.is_W_trainable = kwargs.pop('is_W_trainable', True)
        self.is_R_trainable = kwargs.pop('is_R_trainable', True)

        # Initializer for identity matrix, zero matrix and ones matrix
        self.eye_init = lambda shape, dtype=torch.float32: torch.eye(*shape, dtype=dtype)
        self.zero_init = lambda shape, dtype=torch.float32: torch.zeros(*shape, dtype=dtype)
        self.ones_init = lambda shape, dtype=torch.float32: torch.ones(*shape, dtype=dtype)
        # Get initial values for LDM parameters
        self.A = kwargs.pop('A', self.eye_init((self.dim_x, self.dim_x),dtype=torch.float32))
        self.C = kwargs.pop('C', self.eye_init((self.dim_a, self.dim_x),dtype=torch.float32))
        # Get KF initial conditions
        self.mu_0 = kwargs.pop('mu_0', self.zero_init((self.dim_x,),dtype=torch.float32))
        self.Lambda_0 = kwargs.pop('Lambda_0', self.eye_init((self.dim_x, self.dim_x),dtype=torch.float32))
        # Get initial process and observation noise parameters
        self.W_log_diag = kwargs.pop('W_log_diag', self.ones_init((self.dim_x,),dtype=torch.float32))
        self.R_log_diag = kwargs.pop('R_log_diag', self.ones_init((self.dim_a,),dtype=torch.float32))

        # register trainable parameters to modules
        self._register_params()

    def _register_params(self):
        '''
        Register trainable parameters to the module
        '''
        # Register A and C as trainable parameters

        self._check_matrix_shapes()

        self.A = nn.Parameter(self.A, requires_grad = True)
        self.C = nn.Parameter(self.C, requires_grad = True)

        # Register W and R as trainable parameters
        self.W_log_diag = nn.Parameter(self.W_log_diag, requires_grad = self.is_W_trainable)
        self.R_log_diag = nn.Parameter(self.R_log_diag, requires_grad = self.is_R_trainable)
        
        self.mu_0 = nn.Parameter(self.mu_0, requires_grad = True)
        self.Lambda_0 = nn.Parameter(self.Lambda_0, requires_grad = True)


    def _check_matrix_shapes(self):
        '''
        Checks whether LDM parameters have the correct shapes, which are defined above in the constructor
        '''
        if self.A.shape != (self.dim_x, self.dim_x):
            assert False, 'Shape of A matrix is not (dim_x, dim_x)'

        if self.C.shape != (self.dim_a, self.dim_x):
            assert False, 'Shape of C matrix is not (dim_a, dim_x)'

        if len(self.mu_0.shape) != 1:
            self.mu_0 = self.mu_0.view(-1,)
        
        if self.mu_0.shape != (self.dim_x,):
            assert False, 'Shape of mu_0 is not (dim_x,)'

        if self.Lambda_0.shape != (self.dim_x, self.dim_x):
            assert False, 'Shape of Lambda_0 is not (dim_x, dim_x)'
        
        if len(self.W_log_diag.shape) != 1:
            self.W_log_diag = self.W_log_diag.view(-1,)
        
        if self.W_log_diag.shape != (self.dim_x,):
            assert False, 'Shape of W_log_diag is not (dim_x,)'

        if len(self.R_log_diag.shape) != 1:
            self.R_log_diag = self.R_log_diag.view(-1,)
        
        if self.R_log_diag.shape != (self.dim_a,):
            assert False, 'Shape of R_log_diag is not (dim_a,)'
        



    def forward(self, a, mask=None, do_smoothing=False):
        '''
        Forward pass function for LDM Module

        Parameters:
        ------------
        - a: torch.Tensor, shape: (num_seq, num_steps, dim_a), Batch of projected manifold latent factors (outputs of encoder; nonlinear manifold embedding step)
        - mask: torch.Tensor, shape: (num_seq, num_steps, 1), Mask input which shows whether 
                                                                     observations at each timestep exists (1) or are missing (0)
        do_smoothing: bool, Whether to run RTS Smoothing or not

        Returns:
        ------------
        - mu_pred_all: torch.Tensor, shape: (num_seq, num_steps, dim_x), Dynamic latent factor predictions (t+1|t) where first index of the second dimension has x_{1|0}
        - mu_t_all: torch.Tensor, shape: (num_seq, num_steps, dim_x), Dynamic latent factor filtered estimates (t|t) where first index of the second dimension has x_{0|0}
        - mu_back_all: torch.Tensor, shape: (num_seq, num_steps, dim_x), Dynamic latent factor smoothed estimates (t|T) where first index of the second dimension has x_{0|T}. Ones tensor if do_smoothing is False
        - Lambda_pred_all: torch.Tensor, shape: (num_seq, num_steps, dim_x, dim_x), Dynamic latent factor estimation error covariance predictions (t+1|t) where first index of the second dimension has P_{1|0}
        - Lambda_t_all: torch.Tensor, shape: (num_seq, num_steps, dim_x, dim_x), Dynamic latent factor estimation error covariance filtered estimates (t|t) where first index of the second dimension has P_{0|0}
        - Lambda_back_all: torch.Tensor, shape: (num_seq, num_steps, dim_x, dim_x), Dynamic latent factor estimation error covariance smoothed estimates (t|T) where first index of the second dimension has P_{0|T}. Ones tensor if do_smoothing is False
        '''

        if do_smoothing:
            mu_pred_all, mu_t_all, mu_back_all, Lambda_pred_all, Lambda_t_all, Lambda_back_all = self.smooth(a=a, mask=mask) 
        else:
            mu_pred_all, mu_t_all, Lambda_pred_all, Lambda_t_all = self.filter(a=a, mask=mask)
            mu_back_all = torch.ones_like(mu_t_all, dtype=torch.float32, device=mu_t_all.device)
            Lambda_back_all = torch.ones_like(Lambda_t_all, dtype=torch.float32, device=Lambda_t_all.device)

        return mu_pred_all, mu_t_all, mu_back_all, Lambda_pred_all, Lambda_t_all, Lambda_back_all
    

    def filter(self, a, mask=None):
        '''
        Performs Kalman Filtering  

        Parameters:
        ------------
        - a: torch.Tensor, shape: (num_seq, num_steps, dim_a), Batch of projected manifold latent factors (outputs of encoder; nonlinear manifold embedding step)
        - mask: torch.Tensor, shape: (num_seq, num_steps, 1), Mask input which shows whether 
                                                                     observations at each timestep exists (1) or are missing (0)

        Returns: 
        ------------
        - mu_pred_all: torch.Tensor, shape: (num_seq, num_steps, dim_x), Dynamic latent factor predictions (t+1|t) where first index of the second dimension has x_{1|0}
        - mu_t_all: torch.Tensor, shape: (num_seq, num_steps, dim_x), Dynamic latent factor filtered estimates (t|t) where first index of the second dimension has x_{0|0}
        - Lambda_pred_all: torch.Tensor, shape: (num_seq, num_steps, dim_x, dim_x), Dynamic latent factor estimation error covariance predictions (t+1|t) where first index of the second dimension has P_{1|0}
        - Lambda_t_all: torch.Tensor, shape: (num_seq, num_steps, dim_x, dim_x), Dynamic latent factor estimation error covariance filtered estimates (t|t) where first index of the second dimension has P_{0|0}
        '''
        # Run the forward iteration
        mu_pred_all, mu_t_all, Lambda_pred_all, Lambda_t_all = self.compute_forwards(a=a, mask=mask)

        # Swab num_seq and num_steps dimensions
        mu_pred_all = torch.permute(mu_pred_all, (1, 0, 2))
        mu_t_all = torch.permute(mu_t_all, (1, 0, 2))
        Lambda_pred_all = torch.permute(Lambda_pred_all, (1, 0, 2, 3))
        Lambda_t_all = torch.permute(Lambda_t_all, (1, 0, 2, 3))

        return mu_pred_all, mu_t_all, Lambda_pred_all, Lambda_t_all

    def compute_forwards(self, a, mask):
        # a的device应该和mu_0一致
        '''
        Performs the forward iteration of causal flexible Kalman filtering, given a batch of trials/segments/time-series

        Parameters: 
        ------------
        - a: torch.Tensor, shape: (num_seq, num_steps, dim_a), Batch of projected manifold latent factors (outputs of encoder; nonlinear manifold embedding step)
        - mask: torch.Tensor, shape: (num_seq, num_steps, 1), Mask input which shows whether 
                                                                     observations at each timestep exists (1) or are missing (0)

        Returns: 
        ------------
        - mu_pred_all: torch.Tensor, shape: (num_steps, num_seq, dim_x), Dynamic latent factor predictions (t+1|t) where first index of the second dimension has x_{1|0}
        - mu_t_all: torch.Tensor, shape: (num_steps, num_seq, dim_x), Dynamic latent factor filtered estimates (t|t) where first index of the second dimension has x_{0|0}
        - Lambda_pred_all: torch.Tensor, shape: (num_steps, num_seq, dim_x, dim_x), Dynamic latent factor estimation error covariance predictions (t+1|t) where first index of the second dimension has P_{1|0}
        - Lambda_t_all: torch.Tensor, shape: (num_steps, num_seq, dim_x, dim_x), Dynamic latent factor estimation error covariance filtered estimates (t|t) where first index of the second dimension has P_{0|0}
        '''
      
        if mask is None:
            mask = torch.ones(a.shape[:-1], dtype = torch.float32, device = self.device)

        num_seq, num_steps, _ = a.shape

        # Make sure that mask is 3D (last axis is 1-dimensional)
        if len(mask.shape) != len(a.shape):
            mask = mask.unsqueeze(dim=-1)  # (num_seq, num_steps, 1)
        
        # To make sure we do not accidentally use the real outputs in the steps with missing values, set them to a dummy value, e.g., 0.
        # The dummy values of observations at masked points are irrelevant because:
        # Kalman disregards the observations by setting Kalman Gain to 0 in K = torch.mul(K, mask[:, t, ...].unsqueeze(dim=1)) @ line 204
        
        a_masked = torch.mul(a, mask) # (num_seq, num_steps, dim_a) x (num_seq, num_steps, 1)
        mu_0 = self.mu_0.unsqueeze(dim = 0).repeat(num_seq, 1) # (num_seq, dim_x)
        Lambda_0 = self.Lambda_0.unsqueeze(dim = 0).repeat(num_seq, 1, 1) # (num_seq, dim_x, dim_x)

        mu_pred = mu_0 # (num_seq, dim_x)
        Lambda_pred = Lambda_0 # (num_seq, dim_x, dim_x)
        # Create empty arrays for filtered and predicted estimates, NOTE: The last time-step of the prediction has T+1|T, which may not be of interest
        mu_pred_all = torch.zeros((num_steps, num_seq, self.dim_x), dtype = torch.float32, device = mu_0.device)
        mu_t_all = torch.zeros((num_steps, num_seq, self.dim_x), dtype = torch.float32, device = mu_0.device)
        # Create empty arrays for filtered and predicted error covariance, NOTE: The last time-step of the prediction has T+1|T, which may not be of interest
        Lambda_pred_all = torch.zeros((num_steps, num_seq, self.dim_x, self.dim_x), dtype = torch.float32, device = mu_0.device)
        Lambda_t_all = torch.zeros((num_steps, num_seq, self.dim_x, self.dim_x), dtype = torch.float32, device = mu_0.device)

        W, R = self._get_covariance_matrices()

        for t in range(num_steps):
            # Tile C matrix for each time segement
            C_t = self.C.repeat(num_seq, 1, 1) # (num_seq, dim_a, dim_x)

            # obtain a residual
            a_pred = (C_t @ mu_pred.unsqueeze(dim=-1)).squeeze(dim=-1) # (num_seq, dim_a)
            r = a_masked[:, t, ...] - a_pred # (num_seq, dim_a)

            # Project system uncertainty into measurement space, get Kalman Gain
            S = C_t @ Lambda_pred @ torch.permute(C_t, (0,2,1)) + R # (num_seq, dim_a, dim_a)
            S_inv = torch.inverse(S.cpu()).to(S.device) # (num_seq, dim_a, dim_a)
            K = Lambda_pred @ torch.permute(C_t, (0, 2, 1)) @ S_inv # (num_seq, dim_x, dim_a)
            K = torch.mul(K, mask[:,t, ...].unsqueeze(dim = 1))   # (num_seq, dim_x, dim_a) x (num_seq, 1,  1)

            mu_t = mu_pred + (K @ r.unsqueeze(dim= - 1)).squeeze(dim = -1) # (num_seq, dim_x)
            I_KC = torch.eye(self.dim_x, dtype=torch.float32, device=mu_0.device) - K @ C_t # (num_seq, dim_x, dim_x)
            Lambda_t = I_KC @ Lambda_pred # (num_seq, dim_x, dim_x)

            # tile A matrix for each time segment
            A_t = self.A.repeat(num_seq, 1, 1) # (num_seq, dim_x, dim_x)
            mu_pred = (A_t @ mu_t.unsqueeze(dim = -1)).squeeze(dim = -1) # (num_seq, dim_x)
            Lambda_pred = A_t @ Lambda_t @ torch.permute(A_t, (0, 2, 1)) + W # (num_seq, dim_x, dim_x) x (num_seq, dim_x, dim_x) x (num_seq, dim_x, dim_x) --> (num_seq, dim_x, dim_x)

            mu_pred_all[t, ...] = mu_pred
            mu_t_all[t, ...] = mu_t
            Lambda_pred_all[t, ...] = Lambda_pred
            Lambda_t_all[t, ...] = Lambda_t

        return mu_pred_all, mu_t_all, Lambda_pred_all, Lambda_t_all
    

    def _get_covariance_matrices(self):
        '''
        Get the process and observation noise covariance matrices from log-diagonals. 

        Returns:
        ------------
        - W: torch.Tensor, shape: (self.dim_x, self.dim_x), Process noise covariance matrix
        - R: torch.Tensor, shape: (self.dim_a, self.dim_a), Observation noise covariance matrix
        '''
        W = torch.diag(torch.exp(self.W_log_diag))
        R = torch.diag(torch.exp(self.R_log_diag))

        return W, R

    def smooth(self, a, mask=None):
        pass

    def compute_backwards(self, mu_pred_all, mu_t_all, Lambda_pre_all, Lambda_t_all):
        pass
