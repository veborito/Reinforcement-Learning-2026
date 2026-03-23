import numpy as np

## This a discrete MDP with a finite number of states and actions
class DiscreteMDP:
    ## initalise a random MDP with
    ## n_states: the number of states
    ## n_actions: the number of actions
    ## Optional arguments:
    ## P: the state-action-state transition matrix so that P[s,a,s_next] is the probability of s_next given the current state-action pair (s,a)
    ## R: The state-action reward matrix so that R[s,a] is the reward for taking action a in state s.
    def __init__(self, n_states, n_actions, P = None, R = None):
        self.n_states = n_states # the number of states of the MDP
        self.n_actions = n_actions # the number of actions of the MDP
        if (P is None):
            self.P = np.zeros([n_states, n_actions, n_states]) # the transition probability matrix of the MDP so that P[s,a,s'] is the probabiltiy of going to s' from (s,a)
            for s in range(self.n_states):
                for a in range(self.n_actions):
                    self.P[s,a] = np.random.dirichlet(np.ones(n_states)) # generalisation of Beta to multiple outcome
        else:
            self.P = P
        if (R is None):
            self.R = np.zeros([n_states, n_actions]) # the expected reward for each action and state
            # generate uniformly random transitions and 0.1 bernoulli rewards
            for s in range(self.n_states):
                for a in range(self.n_actions):
                    self.R[s,a] = np.round(np.random.uniform(), decimals=1)
        else:
            self.R = R
        
        # check transitions
        # for s in range(self.n_states):
        #     for a in range(self.n_actions):
        #         #print(s,a, ":", self.P[s,a,:])
                # assert(abs(np.sum(self.P[s,a,:])-1) <= 1e-3)
                # assert((P[s,a,:] <= 1).all())
                # assert((P[s,a,:] >= 0).all())
                
    # get the probability of next state j given current state s, action a, i.e. P(j|s,a)
    def get_transition_probability(self, state, action, next_state):
        return self.P[state, action, next_state]
    
    # get the vector of probabilities over next states P( . | s,a)
    def get_transition_probabilities(self, state, action):
        return self.P[state, action]
    
    # Get the reward for the current state action.
    # It can also be interpreted as the expected reward for the state and action.
    def get_reward(self, state, action):
        return self.R[state, action]

class ChainMDP(DiscreteMDP):
    """
    Problem where we need to take the same action n_states-1 time in a row to get a highly rewarding state
    The optimal policy greatly depends on the discount factor we choose.
    """

    def __init__(self, n_states=20):
        assert  n_states > 1

        n_actions = 2
        super().__init__(n_states=n_states, n_actions=n_actions)

        self.R[:] = 0.
        self.P[:] = 0.

        self.R[:, 1] = -1 / (n_states-1)
        self.R[n_states-1, 1] = 1.
        self.R[:, 0] = 1/n_states

        for i in range(self.n_states-1):
            if i > 0:
                self.P[i, 0, i-1] = 1.
            else:
                self.P[i, 0, i] = 1.

            self.P[i, 1, i+1] = 1.

        self.P[self.n_states-1, :, self.n_states-1] = 1.


class SecretaryMDP(DiscreteMDP):
    """
    Implementation of the basic secretary problem.
    """

    def __init__(self, n_candidates=10):

        super().__init__(n_states=2*n_candidates+1, n_actions=2)

        self.P[:] = 0.
        self.R[:] = 0.

        self.QUIT = 1
        self.CONTINUE = 0
        self.end_state = self.n_states-1

        # If we quit, we go to the end state
        self.P[:, self.QUIT, self.end_state] = 1.

        # If we don't quit, move to the next candidate
        best_sofar_idx = np.arange(0, self.n_states-1, 2) # pair idxs are states where we have the best candidate seen so far.
        print(best_sofar_idx)
        regular_candidate_idx = np.arange(1, self.n_states-1, 2) # odd idxs are for states where we have no the best candidate
        print(regular_candidate_idx)

        for i, idx in enumerate(best_sofar_idx):
            if i == n_candidates-1:
                # Last candidate
                self.P[idx, self.CONTINUE, self.end_state] = 1
            else:
                # Go from a best so far individual to an even better individual
                self.P[idx, self.CONTINUE, idx+2] = 1 / (2 + i)
                # Go from a best so far individual to a regular individual
                self.P[idx, self.CONTINUE, idx+3] = (1 + i) / (2 + i)

        for i, idx in enumerate(regular_candidate_idx):
            if i == n_candidates-1:
                # Last candidate
                self.P[idx, self.CONTINUE, self.end_state] = 1
            else:
                # Go from a regular individual to best so far individual
                self.P[idx, self.CONTINUE, idx + 3] = 1 / (2 + i)
                # Go from a regular to a regular
                self.P[idx, self.CONTINUE, idx + 2] = (1 + i) / (2 + i)

        # We are only rewarded for getting a best so far individual
        self.R[best_sofar_idx, self.QUIT] = [n_seen / n_candidates for n_seen in range(n_candidates)]




if __name__ == '__main__':
    # Unit test

    # p = ChainMDP()
    #print(p.P, p.R)

    p = SecretaryMDP()
    print(p.P)
    print(p.R)
