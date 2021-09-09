import numpy as np
from typing import List
from maMDP.environments import Environment
from joblib import Parallel, delayed

# INVERSE BINOMIAL SAMPLING #

def fit_algorithm(env:Environment, agent_name:str, n_planning_steps:int) -> int:
    """
    Predicts the next move for a given agent in a given environment, and returns
    the predicted next state.

    Args:
        env (Environment): The environment to be used
        agent_name (str): The agent moving
        n_planning_steps (int): Number of steps for the agent to plan ahead

    Returns:
        int: Predicted next state
    """

    env.reset()

    env.fit(agent_name, n_planning_steps)
    predicted_state = env.step(agent_name, n_planning_steps)

    env.reset() 

    return predicted_state

def IBS_ll(stopping_point:int, env:Environment, agent_name:str, observed:List[int], max_planning_steps:int=None) -> float:
    """
    Calculates trial-wise log-likelihood for one iteration of IBS. Intended to be used to facilitate
    parallelising the IBS algorithm at the repetition level

    Args:
        stopping_point (int): Maximum number of samples to be drawn from the model.
        env (Environment): Environment
        agent_name (str): Name of agent moving
        observed (List[int]): Sequence of observed moves (states)
        max_planning_steps (int): Maximum number of steps to plan ahead. This is decremented by 1 on each trial.

    Returns:
        float: Log-likelihood
    """

    n_trials = len(observed)

    if max_planning_steps is None:
        max_planning_steps = n_trials

    rep_K = np.zeros(n_trials)
    rep_ll = np.zeros(n_trials)

    predictions = np.zeros((n_trials, stopping_point)) - 999

    hit = np.zeros(n_trials, dtype=bool)

    # Try K until stopping point
    for k in range(stopping_point):

        # print('K = {0}'.format(k))

        for n in range(n_trials):

            if not hit[n]:
            
                predicted_state = fit_algorithm(env[n], agent_name, max_planning_steps-n)
                predictions[n, k] = predicted_state

                # print('Trial = {0}, K = {1}, state = {2}'.format(n, k, predicted_state))

                # If it's a hit, stop this trial from being fit again
                if predicted_state == observed[n]:
                    hit[n] = True

                # Otherwise increment K
                else:
                    rep_K[n] += 1

    # Get LL for each trial on this repeat
    for n in range(n_trials):
        rep_ll[n] = -np.sum(1./np.arange(1, rep_K[n]-1))

    print(rep_K)

    return rep_ll

class IBSEstimator():

    def __init__(self, n_jobs:int=1, n_repeats:int=1, stopping_point:int=6, max_planning_steps=None):

        self.n_jobs = n_jobs
        self.n_repeats = n_repeats
        self.stopping_point = stopping_point
        self.max_planning_steps = max_planning_steps
    
    def fit(self, env:Environment, agent_name:str, observed:List[int]):
        """
        Calculates the IBS log-likelihood of the model.

        Args:
            env (Environment): Environment in which the agent is acting. Can be a list of environments, one per move.
            agent_name (str): Name of the agent to move.
            observed (List[int]): List of observed moves (represented by states visited)
        """

        n_trials = len(observed)

        # If env isn't a list, turn it into one
        if not isinstance(env, list):
            env = [env] * n_trials

        if not len(env) == n_trials:
            raise AttributeError("Number of environments should match number of trials")

        # If not doing this in parallel, just do this with a for loop
        if self.n_jobs == 1:
            
            ll = []
            # prediction_list = []

            for _ in range(self.n_repeats):
                # est_ll, predicted_states = IBS_ll(self.stopping_point, env, agent_name, observed)
                # ll.append(est_ll)
                # prediction_list.append(predicted_states)
                ll.append(IBS_ll(self.stopping_point, env, agent_name, observed, self.max_planning_steps))

        # Otherwise use joblib
        else:
            ll = Parallel(n_jobs=self.n_jobs)(delayed(IBS_ll)(self.stopping_point, env, agent_name, observed, self.max_planning_steps) 
                                              for _ in range(self.n_repeats))

        self.ll = np.stack(ll)

        # Get average LL across repeats for each trial
        self.average_ll = self.ll.mean(axis=0)
        # Get overall summed LL
        self.ll_sum = self.average_ll.sum()
        
        # self.predicted_states = np.stack(prediction_list)



            


