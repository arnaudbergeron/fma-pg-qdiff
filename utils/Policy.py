import torch
import torch.nn as nn

class Policy(nn.Module):
    def __init__(self, env, P, mu, gamma, r, device='cpu', seed=10):
        super(Policy, self).__init__()

        self.env = env
        self.n_obs = self.env.observation_space.n + 1
        generator = torch.Generator(device=device).manual_seed(seed)
        _theta = torch.rand((self.n_obs, self.env.action_space.n), requires_grad=True, device=device, dtype=torch.float64,generator=generator) 
        self.theta = _theta

        self.total_states = self.n_obs * self.env.action_space.n
        self.P = P
        self.mu = mu
        self.gamma = gamma
        self.r = r
        self.start_state = torch.argmax(self.mu)
        self.device = device

        self.qpi = None
        self.vpi = None
        self.dpi = None
        self.dpis = None
        self.jacobian = torch.zeros((self.total_states, self.total_states), device=device, dtype=torch.float64)

    def forward(self):
        self.pi = torch.softmax(self.theta, 1)

    def update_dpi(self):
        p_pi = torch.einsum('xay,xa->xy', self.P, self.pi)
        d_pi = (1 - self.gamma) * torch.linalg.solve((torch.eye(self.n_obs, device=self.device) - self.gamma * p_pi).T,
                            self.mu)
        
        d_pi /= d_pi.sum() # for addressing numerical errors

        self.dpi =  d_pi
    
    def update_vpi(self):
        p_pi = torch.einsum('xay,xa->xy', self.P, self.pi)
        r_pi = torch.einsum('xa,xa->x', self.r, self.pi)
        v_pi = torch.linalg.solve((torch.eye(self.n_obs, device=self.device) - self.gamma * p_pi),
                               r_pi)

        self.vpi =  v_pi
    
    # compute q^pi = r(s, a) + gamma * sum_s' p(s' | s, a) * v^pi(s')
    def update_qpi(self):
        q_pi = self.r + self.gamma * torch.einsum('xay,y->xa', self.P, self.vpi)
        self.qpi = q_pi
    
    def update_all_dpi(self):
        p_pi = torch.einsum('xay,xa->xy', self.P, self.pi)

        dpis = torch.zeros((self.n_obs, self.n_obs), device=self.device)
        for i in range(self.n_obs):
            mu = torch.zeros(self.n_obs, device=self.device)
            mu[i] = 1

            d_pi = (1 - self.gamma) * torch.linalg.solve((torch.eye(self.n_obs, device=self.device) - self.gamma * p_pi).T,
                            mu)
        
            d_pi /= d_pi.sum()
            dpis[i] = d_pi

        self.dpis = dpis
        self.dpi = dpis[self.start_state]

    def update_J(self):
        return torch.sum(self.mu * (torch.sum(self.qpi * self.pi, dim=1)))
    
    def update_all(self):
        self.update_vpi()
        self.update_qpi()
        self.update_dpi()

    def compute_jacobian(self):
        def fw(w):
            return torch.log(w).flatten()[grad_idx]
        
        pi_flat = self.pi.flatten()        
        for _state_action_pair in range(self.total_states):
            grad_idx = _state_action_pair
            self.jacobian[_state_action_pair] = torch.autograd.functional.jacobian(fw, pi_flat)

    def compute_grad_J(self):

        self.compute_jacobian()
        dpi_flat = torch.stack([self.dpi.clone() for k in range(4)], dim=1).flatten()
        pi_flat = self.pi.flatten()
        qpi_flat = self.qpi.flatten()
        grad_J = torch.einsum("s,s,sx,s->x", dpi_flat, qpi_flat, self.jacobian, pi_flat) / (1 - self.gamma)

        return grad_J