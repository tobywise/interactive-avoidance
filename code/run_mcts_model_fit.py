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



# TODO is the "sampling" implemented at the MCTS level or the softmax level?
# It's implemented at the level of the whole model. Could try using a model with softmax versus a deterministic one


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


def IBS_MCTS_sample_comparison(trial_n:int, observed_action:int, planning_method:str, env:HexEnvironment, 
                               n_iter:int, C:float, min_opponent_moves:int, max_opponent_moves:int, 
                               n_steps:int, caught_cost:float, opponent_q_values:np.ndarray or None) -> bool:
    """[summary]
    Samples from the MCTS planning model and compares the simulated action to the subject's observed action. Returns
    true if they match, false if not.
    Args:
        trial_n (int): Trial number
        observed_action (int): Observed action by the subject
        planning_method (str): Method to use for planning. If 'known', q values are 
        env (HexEnvironment): [description]
        n_iter (int): [description]
        C (float): [description]
        min_opponent_moves (int): [description]
        max_opponent_moves (int): [description]
        n_steps (int): [description]
        caught_cost (float): [description]
        opponent_q_values (np.ndarray or None): [description]

    Returns:
        bool: [description]
    """

    mcts = MCTS(env.mdp, env.agents)

    if planning_method == 'known':
        actions, action_values, _ = mcts.fit(env.agents[1].position, env.agents[0].position, n_iter=n_iter, C=2, 
                                             min_opponent_moves=min_opponent_moves, max_opponent_moves=max_opponent_moves, 
                                             n_steps=n_steps, opponent_policy_method=planning_method, caught_cost=-caught_cost,
                                             opponent_q_values=env.agents[0].solver.q_values_)
    elif planning_method in ['solve', 'solve_IRL']:
        actions, action_values, _ = mcts.fit(env.agents[1].position, env.agents[0].position, n_iter=n_iter, C=2, 
                                             min_opponent_moves=min_opponent_moves, max_opponent_moves=max_opponent_moves, 
                                             n_steps=n_steps, opponent_policy_method=planning_method, caught_cost=-caught_cost,
                                             opponent_q_values=opponent_q_values)
    elif planning_method == 'unknown':
        actions, action_values, _ = mcts.fit(env.agents[1].position, env.agents[0].position, n_iter=n_iter, C=2,
                                             min_opponent_moves=min_opponent_moves, max_opponent_moves=max_opponent_moves, 
                                             n_steps=n_steps, opponent_policy_method=None, caught_cost=-caught_cost)

    action_values_filled = np.ones(6) * -np.inf # Include actions that can't be taken to ensure shapes match
    action_values_filled[actions] = action_values
    simulated_action = np.argmax(action_values_filled)
    

def VI_sample_comparison(observed_action, planning_method, env):

    if planning_method == 'value_iteration':
        env.agents[1].reward_function = [0, 0, 1, -1, 0]
        actions, action_values, states = get_value_iteration_actions(env.agents[1], env.mdp, env.agents[1].position)
    elif planning_method == 'value_iteration_escape':
        env.agents[1].reward_function = [0, 0, 0, -1, 0]
        actions, action_values, states = get_value_iteration_actions(env.agents[1], env.mdp, env.agents[1].position)
    

def mcts_model_comparison_IBS(environment, agents, opponent_policy, predator_reward_function, prey_positions, predator_positions,
                          predator_moves='(2, 2)', caught_cost=100000, reward_value=200, n_mcts=10000, n_turns=20, ibs_max_iter=20):

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

    all_trial_ll = []



    for sim_actions in range(ibs_max_iter):



        # USING IBS FOR LL ESTIMATION <<< THIS COULD BE PARALLELISED
        # TODO try using rows first implementation, as in paper supplement
        # TODO use chance as lower bound
        for move in range(len(prey_positions[:])  - 1):  # Use number of prey positions rather than moves to account for being caught
            mcts.reset()

            # Put the prey in the correct place
            env.agents[1].position = prey_positions[move]

            # Get subject's choice
            # TODO find states variable
            subject_chosen_action = np.where(states == prey_positions[move + 1])

            simulated_action = None
            k = 0

            while simulated_action != subject_chosen_action:
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
                
                action_values_filled = np.ones(6) * -np.inf # Include actions that can't be taken to ensure shapes match
                action_values_filled[actions] = action_values
                simulated_action = np.argmax(action_values_filled)

                k += 1

        # Get IBS log likelihood for this trial
        trial_ll = -np.sum(1./np.arange(1, k-1))
        all_trial_ll.append(trial_ll)

        # Move predator
        env.move_agent(0, predator_positions[move][-1])

        # Remove reward
        if env.mdp.features[2, prey_positions[move]] > 0:
            env.mdp.features[2, prey_positions[move]] = 0

        
        action_values_filled = np.ones(6) * -np.inf # Include actions that can't be taken to ensure shapes match
        action_values_filled[actions] = action_values

        all_action_values.append(action_values_filled)
        all_actions.append(subject_chosen_action)
        # all_states.append(states)

    ll = np.sum(all_trial_ll)

    # TODO NEED TO FIND A WAY TO COMPARE ACROSS MODELS ON DIFFERENT SCALES - !IBS DEALS WITH THIS!
    # e,g, VI gives small q values, MCTS might not (although normalising according to visitation should deal with this to some extent?)

    # all_action_values = np.stack(all_action_values)
    # all_action_values = np.stack(all_action_values) * 0.001  # Big number cause problems
    # all_actions = np.stack(all_actions).squeeze()
    # print('AA', all_action_values)
    # print(normalise_over_actions(softmax(all_action_values, .01)))
    # dist = CategoricalDist(normalise_over_actions(softmax(all_action_values, .01)))
    
    # ll = dist.pmf(all_actions)
    # ll_ind = list(dist.pmf_individual(all_actions))
    # print('ll', ll_ind)
    return ll, all_trial_ll

def get_predator_move_positions(predator_positions, n_moves):
    predator_move_positions = []
    predator_move_positions.append([predator_positions[0]])
    for i in range(1, len(predator_positions), n_moves):
        predator_move_positions.append(predator_positions[i:i+n_moves])
    return predator_move_positions


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('task_type', type=str)
    parser.add_argument('game_ref', type=str)
    args = parser.parse_args()

    # RUN FITTING #
    if not args.task_type in ['planning', 'combined']:
        raise ValueError("{0} is an invalid task type, should be either 'planning' or 'combined'".format(args.task_type))

    # Load in game info
    games = joblib.load('../data/{1}/{0}/game_info'.format(args.game_ref, (args.task_type)))

    # Load in behaviour
    with open('../data/{1}/{0}/behaviour.json'.format(args.game_ref, args.task_type), 'r') as f:
        data_dict = json.load(f)

    # Each run of this script does one subject
    # runID = 0
    runID = int(os.environ['SLURM_ARRAY_TASK_ID'])

    # subjectID = data_dict['subjectIDs'][runID-1]
    subjectIDs = data_dict['subjectIDs']
    fitting_methods = ['value_iteration', 'value_iteration_escape', 'unknown', 'solve', 'solve_IRL']
    subs_methods = list(itertools.product(subjectIDs, fitting_methods))
    subjectID, fit_policy = subs_methods[runID]
    # subjectID = '5d82d28977546f00166ae702'
    # fit_policy = fitting_methods[-1]

    # subjectID = 'TESTTEST'

    print("Fitting subject {0}, method = {1}".format(subjectID, fit_policy))

    envs = []
    policies = []
    lls = []
    move_lls = []
    move_numbers = []
    
    # Set up IRL for combined task
    if 'IRL' in fit_policy:
        IRL = MaxCausalEntIRL(discount=0.9, learning_rate=0.3, max_iter_irl=20, method='numba')
        IRL.theta = np.ones(games.environments[0].mdp.n_features) * 0.5  # Starting point

        # DEPENDENT ON FEATURES - THIS SETS REWARD AND PREDATOR ITSELF TO ZERO
        IRL.theta[[2, 3]] = 0
        estimated_thetas = np.zeros((games.n_envs, games.environments[0].mdp.n_features))

    for env in tqdm(range(games.n_envs)):

        predator_positions = get_predator_move_positions(data_dict['behaviour'][subjectID]['behaviourData']['env_{0}'.format(env+1)]['predator'], games.predator_moves)
        prey_positions = data_dict['behaviour'][subjectID]['behaviourData']['env_{0}'.format(env+1)]['prey'] # TODO figure out why there are sometimes extra positions

        # Run fitting
        if fit_policy == 'solve_IRL':
            predator_reward_function = IRL.theta
            predator_reward_function[[2, 3]] = 0
        else:
            predator_reward_function = games.environments[env].agents[0].reward_function


        # predator_reward_function = [1, 0.3, 0, 0, 0] #######################
        ll, ll_ind = mcts_model_comparison(games.environments[env], games.environments[env].agents, fit_policy, 
                                                    predator_reward_function, prey_positions, predator_positions, reward_value=games.reward_amount,
                                                    caught_cost=games.caught_cost)
        
        # Row for each move in each environment
        lls += [ll] * len(ll_ind)
        move_lls += ll_ind
        envs += [env] * len(ll_ind)
        policies += [fit_policy] * len(ll_ind)
        move_numbers += range(len(ll_ind))

        # Run IRL based on predator's moves
        if 'IRL' in fit_policy:
            traj = [data_dict['behaviour'][subjectID]['behaviourData']['env_{0}'.format(env+1)]['predator']]

            if env == 0:
                IRL.fit(games.environments[env].mdp, traj, reset=True, ignore_features=[2, 3])
            else:
                IRL.fit(games.environments[env].mdp, traj, reset=False, ignore_features=[2, 3])

            estimated_thetas[env, :] = IRL.theta

    # Get output
    out_df = pd.DataFrame({'environment': envs, 'policy': policies, 'll': lls, 'move_ll': move_lls, 'move_number': move_numbers})
            
    # Add estimated reward function to output if using IRL
    if 'IRL' in fit_policy:
        estimated_thetas_df = pd.DataFrame(estimated_thetas, columns=['feature_{0}'.format(i) for i in range(estimated_thetas.shape[-1])])
        estimated_thetas_df = pd.concat([out_df[out_df['move_number'] == 0][['environment', 'policy', 'll']].reset_index(), estimated_thetas_df], axis=1)
        print(estimated_thetas_df)
        estimated_thetas_df.to_csv('../data/{2}/{0}/estimated_thetas__subject-{1}__method-{3}.csv'.format(args.game_ref, subjectID, args.task_type, fit_policy), index=False)
    
    # Save data
    out_df.to_csv('../data/{2}/{0}/model_fit_results__subject-{1}__method-{3}.csv'.format(args.game_ref, subjectID, args.task_type, fit_policy), index=False)

