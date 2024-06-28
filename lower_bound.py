import torch
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import seaborn as sns

from utils.Policy import Policy
from utils.SetupEnv import setup_env

plt.rcParams['text.usetex'] = True
sns.set_style("whitegrid")

gamma = 0.9
seed = 100001

mu_policy, start_state = setup_env(gamma, 'cpu', seed)
mu_policy.pi = torch.rand((mu_policy.pi.shape[0], mu_policy.pi.shape[1]), device='cpu', dtype=torch.float64)
mu_policy.pi = mu_policy.pi / mu_policy.pi.sum(dim=1)[:,None]
mu_policy()

mu_policy.update_all()
mu_J = mu_policy.update_J()

pi_policy = Policy(env=mu_policy.env, P=mu_policy.P, mu=mu_policy.mu, gamma=mu_policy.gamma, r=mu_policy.r, device='cpu', seed=1)

kl_list = []
softmax_J_diff_list = []
direct_J_diff_list = []

softmax_J_hat_list = []
direct_J_hat_list = []

for i in tqdm(range(1_000)):
    pi_policy.pi = mu_policy.pi.clone()

    # choose a random number of perturbed states
    num_perturbed_states = random.randint(1, mu_policy.pi.shape[0]-1)
    perturbed_states = random.choices(range(0, mu_policy.pi.shape[0]), k=num_perturbed_states)
    perturbed_actions = random.choices(range(0, mu_policy.pi.shape[1]), k=num_perturbed_states)

    pi_policy.pi[perturbed_states, perturbed_actions] += (torch.randn((num_perturbed_states), device='cpu', dtype=torch.float64) * 0.01)
    pi_policy.pi = pi_policy.pi / pi_policy.pi.sum(dim=1)[:,None]

    # Compute the RHS of the inequality
    log_term = torch.log(pi_policy.pi/mu_policy.pi)
    reverse_kl_div = torch.sum(pi_policy.pi* log_term)
    forward_kl_div = torch.sum(-1 * mu_policy.pi * log_term)
    
    pi_policy.update_all()
    pi_J = pi_policy.update_J()

    mu_dpi = mu_policy.dpi
    mu_Q = mu_policy.qpi
    mu_min_Q = torch.min(mu_Q, dim=1).values

    scaled_Q = mu_Q - mu_min_Q[:,None]
    advantage = mu_Q - mu_policy.vpi[:,None]

    softmax_term = torch.log(pi_policy.pi/ mu_policy.pi) * mu_policy.pi
    softmax_new_term_with_kl = torch.sum(mu_dpi*torch.sum(mu_Q * softmax_term, dim=1)) - torch.sum((scaled_Q - advantage) * forward_kl_div)
    softmax_J_hat = pi_J + softmax_new_term_with_kl
    softmax_J_diff =  mu_J - softmax_J_hat

    direct_term = pi_policy.pi - mu_policy.pi
    direct_term_with_kl = torch.sum(mu_dpi*torch.sum(mu_Q * direct_term, dim=1)) - torch.sum((scaled_Q - advantage) * reverse_kl_div)
    direct_J_hat = pi_J + direct_term_with_kl
    direct_J_diff =  mu_J - direct_J_hat

    kl_list.append(reverse_kl_div.sum().item())
    softmax_J_diff_list.append(softmax_J_diff.detach().item())
    direct_J_diff_list.append(direct_J_diff.detach().item())

    softmax_J_hat_list.append(softmax_J_hat.detach().item())
    direct_J_hat_list.append(direct_J_hat.detach().item())


softmax_np_J = np.array(softmax_J_diff_list)
direct_np_J = np.array(direct_J_diff_list)

print(f"N less than Zeros Softmax: {np.sum(softmax_np_J<0)}")
print(f"N less than Zeros Direct: {np.sum(direct_np_J<0)}")

fig, ax = plt.subplots(3,1,)
fig.tight_layout()

ax[0].scatter(kl_list, softmax_J_diff_list)
ax[0].set_xlabel('KL Divergence')
ax[0].set_title(r'$J_\pi - \hat{J}_{softmax}(\pi,\mu)$',)

ax[1].scatter(kl_list, direct_J_diff_list)
ax[1].set_xlabel('KL Divergence')
ax[1].set_title(r'$J_\pi - \hat{J}_{direct}(\pi,\mu) $',)

ax[2].scatter(softmax_J_hat_list, direct_J_hat_list)
ax[2].set_xlabel(r'$\hat{J}_{softmax}(\pi,\mu)$')
ax[2].set_ylabel(r'$\hat{J}_{direct}(\pi,\mu)$')
ax[2].set_title(r'$\hat{J}_{softmax}(\pi,\mu)$ vs $\hat{J}_{direct}(\pi,\mu)$')

plt.show()