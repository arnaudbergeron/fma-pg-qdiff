import gym
import torch

from utils.Policy import Policy

def setup_env(gamma, device, seed):
    game = 'CliffWalking-v0' 
    seed_env = 50

    env = gym.make(game).env
    env.action_space.seed(seed_env)
    start_state, info = env.reset(seed=seed_env)

    # Starting state distribution
    mu = torch.zeros(env.observation_space.n+1, device=device, dtype=torch.float64)
    mu[start_state] = 1

    ## Create the reward table and state transition matrix
    r = torch.zeros((env.observation_space.n+1, env.action_space.n), device=device, dtype=torch.float64)
    P = torch.zeros((env.observation_space.n+1, env.action_space.n, env.observation_space.n+1), device=device, dtype=torch.float64) #from, action, target
    for state, s_rew in env.P.items():
        for action, re in s_rew.items():
            P[state, action, re[0][1]] = 1
            if re[0][3]:
                r[state, action] = 0 #goal state
                end_state = re[0][1]
            else:
                if re[0][2] == -1:
                    r[state, action] = -1 #walking
                else:
                    r[state, action] = -100 #cliff

    
    P[end_state, :, :] = 0
    P[end_state, :, end_state+1] = 1
    P[end_state+1, :, end_state+1] = 1
    # r[:,:] = -1
    # r[start_state+1:end_state, :] = -100
    # r[end_state, :] = 10
    r[end_state+1, :] = 0

    # Initialize the policy
    pi = Policy(env, P, mu, gamma, r, device, seed)
    pi()

    return pi, start_state