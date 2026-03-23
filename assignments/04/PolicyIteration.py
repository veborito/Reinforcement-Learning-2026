import numpy as np

def policy_evaluation(mdp, gamma, policy):
    # build the matrix of next states
    P = np.zeros([mdp.n_states, mdp.n_states])
    R = np.zeros(mdp.n_states)
    ## Fill in P, R
    for s in range(mdp.n_states):
        # policy[s] gives me the action for state s
        P[s,:] = mdp.get_transition_probabilities(s, policy[s])
        R[s] = mdp.get_reward(s, policy[s])
        
    ## A @ y equivalent to np.matmul(A, Y)
    ## If y is a vector in $R^n$, then it becomes a [n x 1] matrix
    ## So if A is $R^{n \times n}$ then $A r \in R^n$.
    ## Use np.eye() for identity matrix
    ## Use np.linalg.inv() for matrix inversion
    ## V = (I - gP)^{-1} r
    V = np.linalg.inv(np.eye() - gamma * P) @ R
    


def policy_evaluation_dp(mdp, gamma, policy, depth):
    # build the matrix of next states
    V = np.zeros([mdp.n_states])
    Q = np.zeros([mdp.n_states, mdp.n_actions])
    ## to fill in
    for i in range(depth):
        V_old = V.copy()
        for s in range(mdp.n_states):
            P = mdp.get_transition_probabilities(s, policy[s])
            V[s] = mdp.get_reward(s, policy[s]) + gamma * np.dot(P, V_old)
    return V

## Define algorithm
def policy_iteration(mdp, n_iterations, gamma, depth):
    policy = np.zeros([mdp.n_states], int)
    V = np.zeros([mdp.n_states])
    Q = np.zeros([mdp.n_states, mdp.n_actions])
    old_policy = policy.copy()
    # Evaluation Step
    V_old = np.ones([mdp.n_states])
    for iteration in range(n_iterations):
        # Evaluate policy
        if (depth==0): 
            V = policy_evaluation(mdp, gamma, policy)
        else:
            V = policy_evaluation_dp(mdp, gamma, policy, depth)
        # Improve policy
        for s in range(mdp.n_states):
            for a in range(mdp.n_actions):
                P = mdp.get_transition_probabilities(s, a)
                Q[s,a] = mdp.get_reward(s, a) + gamma * np.dot(P, V)
            policy[s] = np.argmax(Q[s,:])
            
        if (sum(abs(V - V_old)) < 0.001):
            break
        else:
            V_old = V.copy()
            old_policy = policy.copy()
    return policy, V, Q




    