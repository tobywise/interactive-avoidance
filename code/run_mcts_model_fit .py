import itertools
import sys
sys.path.insert(0, '../server/src')
from mdp import HexGridMDP, ValueIteration, HexEnvironment, Agent
from games import PredatorGame
from solvers import MCTS, solve_all_value_iteration, MaxCausalEntIRL, get_actions_states, solve_value_iteration
from tqdm import tqdm
import numpy as np
from numpy.random import RandomState
import argparse
import pandas as pd
import joblib
from copy import deepcopy
import json
import argparse
import os

rng = RandomState(123)

class CategoricalDist():

    def __init__(self, ps):
        self.ps = ps

    def pmf(self, x):
        if len(x) != self.ps.shape[0]:
            raise AttributeError('X is not the right shape')

        x[np.isnan(x)] = self.ps.shape[1]

        return np.log(np.sum(np.pad(self.ps, [(0, 0), (0, 1)])[np.arange(self.ps.shape[0]), x.astype(int)]))

    def pmf_individual(self, x):
        if len(x) != self.ps.shape[0]:
            raise AttributeError('X is not the right shape')

        x[np.isnan(x)] = self.ps.shape[1]

        return np.log(np.pad(self.ps, [(0, 0), (0, 1)])[np.arange(self.ps.shape[0]), x.astype(int)])

def softmax(v, temp=1):

    # print('vv', v)
    # print('ee', np.exp(v))
    # print('oo', np.sum(np.exp(v), axis=1))
    prob = np.exp(v / temp) / np.nansum(np.exp(v / temp), axis=1)[:, None]

    return prob

def normalise_over_actions(x):

    x[x == -np.inf] = np.nan
    return x / np.nansum(x, axis=1)[:, None]

def scale_over_actions(x):
    x[x == -np.inf] = np.nan
    return (x-np.nanmin(x))/(np.nanmax(x)-np.nanmin(x))

# FUNCTIONS TO CREATE FEATURES AND REWARDS
def get_random_move(start, sas):
    next_state = rng.choice(np.where(sas[start, ...])[1])
    return next_state
    
def create_feature(covered_cells, entropy, mdp):

    random_walk_lengths = np.round(np.maximum(1, rng.randn(100) + entropy))

    covered_states = np.zeros(mdp.n_states)

    for walk in random_walk_lengths:
        if not covered_states.sum() >= covered_cells:
            start = rng.randint(mdp.n_states)
            while covered_states[start] != 0:
                start = rng.randint(mdp.n_states)
            covered_states[start] = 1
            for i in range(int(walk - 1)):
                next_state = get_random_move(start, mdp.sas)
                covered_states[next_state] = 1
                if covered_states.sum() >= covered_cells:
                    break

    feature_map = np.zeros(mdp.size)
    for n, i in enumerate(covered_states):
        feature_map[np.unravel_index(n, mdp.size)] = i

    return covered_states, feature_map

def create_overlapping_reward(overlap, n_reward, feature):

    reward_idx = rng.choice(np.where(feature)[0], size=int(n_reward * overlap), replace=False)
    reward_idx = np.hstack([reward_idx, rng.choice(np.where(feature == 0)[0], size=int(n_reward * (1-overlap)), replace=False)])

    reward_array = np.zeros(210)
    reward_array[reward_idx] = 1

    reward_map = np.zeros(len(feature))
    for n, i in enumerate(reward_array):
        reward_map[np.unravel_index(n, len(feature))] = i

    return reward_array, reward_map

def get_value_iteration_actions(agent, mdp, start_state):

    _, qvalues = solve_value_iteration(mdp.sas.shape[0], mdp.sas.shape[1], np.array(agent.reward_function), mdp.features, 500, 0.9, mdp.sas, 1e-4)  # 500 iterations is enough

    actions_states, actions, states = get_actions_states(mdp.sas, start_state)

    return actions, qvalues[start_state, actions], states
    

def mcts_model_comparison(environment, agents, opponent_policy, predator_reward_function, prey_positions, predator_positions,
                          predator_moves='(2, 2)', caught_cost=100000, reward_value=200, n_mcts=10000, n_turns=20):

    # Copy so we don't risk changing things outside the function
    env = deepcopy(environment)
    copied_agents = [deepcopy(agent) for agent in agents]
    
    # Set up MCTS
    mcts = MCTS(env.mdp, copied_agents)

    game_trajectory = {'agent': [env.agents[1].position], 'opponent': [env.agents[0].position]}
    reward_gained = 0
    caught = 0

    predator_min_moves, predator_max_moves = tuple([int(i) for i in predator_moves[1:-1].split(',')])
    predator_max_moves += 1

    # Calculate predator Q values for all possible prey positions
    opponent_q_values = solve_all_value_iteration(env.mdp.sas, np.array(predator_reward_function), env.mdp.features, -1, discount=0.90, tol=1e-8, max_iter=500)

    opponent_values = []

    all_action_values = []
    all_actions = []
    all_states = []

    env.mdp.features[2, :] *= reward_value

    for move in range(len(prey_positions[:])  - 1):  # Use number of prey positions rather than moves to account for being caught
        mcts.reset()

        # Put the prey in the correct place
        env.agents[1].position = prey_positions[move]
        # Solve for prey using MCTS
        if opponent_policy == 'known':
            actions, action_values, states = mcts.fit(env.agents[1].position, env.agents[0].position, n_iter=n_mcts, C=2, min_opponent_moves=predator_min_moves, max_opponent_moves=predator_max_moves, 
                                                    n_steps=n_turns*3, opponent_policy_method='precalculated', caught_cost=-caught_cost, opponent_q_values=env.agents[0].solver.q_values_)
        elif opponent_policy in ['solve', 'solve_IRL']:
            actions, action_values, states = mcts.fit(env.agents[1].position, env.agents[0].position, n_iter=n_mcts, C=2, min_opponent_moves=predator_min_moves, max_opponent_moves=predator_max_moves, 
                                                    n_steps=n_turns*3, opponent_policy_method='solve', caught_cost=-caught_cost, opponent_q_values=opponent_q_values)
        elif opponent_policy == 'unknown':
            actions, action_values, states = mcts.fit(env.agents[1].position, env.agents[0].position, n_iter=n_mcts, C=2, min_opponent_moves=predator_min_moves, max_opponent_moves=predator_max_moves, 
                                                    n_steps=n_turns*3, opponent_policy_method=None, caught_cost=-caught_cost)
        elif opponent_policy == 'value_iteration':
            env.agents[1].reward_function = [0, 0, 1, -1, 0]
            actions, action_values, states = get_value_iteration_actions(env.agents[1], env.mdp, env.agents[1].position)
        elif opponent_policy == 'value_iteration_escape':
            env.agents[1].reward_function = [0, 0, 0, -1, 0]
            actions, action_values, states = get_value_iteration_actions(env.agents[1], env.mdp, env.agents[1].position)

            
        # Move predator
        env.move_agent(0, predator_positions[move][-1])

        # Remove reward
        if env.mdp.features[2, prey_positions[move]] > 0:
            env.mdp.features[2, prey_positions[move]] = 0

        subject_chosen_action = np.where(states == prey_positions[move + 1])
        action_values_filled = np.ones(6) * -np.inf # Include actions that can't be taken to ensure shapes match
        action_values_filled[actions] = action_values

        all_action_values.append(action_values_filled)
        all_actions.append(subject_chosen_action)
        # all_states.append(states)


    # TODO NEED TO FIND A WAY TO COMPARE ACROSS MODELS ON DIFFERENT SCALES
    # e,g, VI gives small q values, MCTS might not (although normalising according to visitation should deal with this to some extent?)

    all_action_values = np.stack(all_action_values)
    all_action_values = np.stack(all_action_values) * 0.001  # Big number cause problems
    all_actions = np.stack(all_actions).squeeze()

    dist = CategoricalDist(normalise_over_actions(softmax(all_action_values, .01)))
    
    ll = dist.pmf(all_actions)
    ll_ind = list(dist.pmf_individual(all_actions))

    return ll, ll_ind

def get_predator_move_positions(predator_positions, n_moves):
    predator_move_positions = []
    predator_move_positions.append([predator_positions[0]])
    for i in range(1, len(predator_positions), n_moves):
        predator_move_positions.append(predator_positions[i:i+n_moves])
    return predator_move_positions

