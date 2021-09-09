import sys
sys.path.insert(0, '../server/src')
from mdp import HexGridMDP, ValueIteration, HexEnvironment, Agent
from solvers import MaxCausalEntIRL, SimpleIRL, MCTS, solve_all_value_iteration
from tqdm import tqdm
import numpy as np
from numpy.random import RandomState
import argparse
import pandas as pd
import numba.tests.npyufunc.test_ufuncbuilding
import uuid
import os
import joblib
from copy import deepcopy

# np.random.seed(123)

# FUNCTIONS TO CREATE FEATURES AND REWARDS
def get_random_move(start, sas):
    next_state = np.random.choice(np.where(sas[start, ...])[1])
    return next_state
    
def create_feature(covered_cells, entropy, mdp):

    random_walk_lengths = np.round(np.maximum(1, np.random.randn(100) + entropy))

    covered_states = np.zeros(mdp.n_states)

    for walk in random_walk_lengths:
        if not covered_states.sum() >= covered_cells:
            start = np.random.randint(mdp.n_states)
            while covered_states[start] != 0:
                start = np.random.randint(mdp.n_states)
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

    reward_idx = np.random.choice(np.where(feature)[0], size=int(n_reward * overlap), replace=False)
    reward_idx = np.hstack([reward_idx, np.random.choice(np.where(feature == 0)[0], size=int(n_reward * (1-overlap)), replace=False)])

    reward_array = np.zeros(210)
    reward_array[reward_idx] = 1

    reward_map = np.zeros(len(feature))
    for n, i in enumerate(reward_array):
        reward_map[np.unravel_index(n, len(feature))] = i

    return reward_array, reward_map


if __name__ == "__main__":

    parameters = pd.read_csv('../data/parameter_df.csv')
    row = int(os.environ['SLURM_ARRAY_TASK_ID'])
    # row = 20
    print(row)

    # row = 20
    parameters = parameters.iloc[row, :]

    # Create unique ID for this run
    UID = uuid.uuid4()

    # Extract reward function
    predator_reward_function = [float(i) for i in parameters.predator_reward_function.split(',')]

    print(predator_reward_function)

    run_rewards = []
    run_caught = []

    opponent_trajectories = []
    agent_trajectories = []
    feature_maps = []

    for run in range(parameters.n_starting_positions):

        # CREATE ENVIRONMENT
        rng = RandomState(123)
        features = np.zeros((3, 21 * 10))

        if parameters.predator_start_location == 'left':
            prey_pos, predator_pos = np.random.choice(range(50), size=2, replace=False)
        elif parameters.predator_start_location == 'right':
            prey_pos = np.random.randint(0, 50)
            predator_pos = np.random.randint(160, 210)
        elif parameters.predator_start_location == 'center':
            prey_pos = np.random.randint(0, 50)
            predator_pos = np.random.randint(100, 140)

        testAgent1 = Agent('Predator_1', predator_reward_function, position=predator_pos, solver_kwargs={'discount': 0.9, 'tol': 1e-4})
        testAgent2 = Agent('Prey_1', [0, 0, 1, 0, 0], position=prey_pos, solver_kwargs={'discount': 0.9, 'tol': 1e-4})

        testMDP = HexGridMDP(features, (21, 10), self_transitions=False)

        tree_states, tree_array = create_feature(parameters.n_trees, parameters.entropy, testMDP)
        red_states, red_array = create_overlapping_reward(0, parameters.n_red, tree_states)
        reward_states, reward_array = create_overlapping_reward(parameters.overlap, parameters.n_rewards, ((tree_states + red_states) > 0).astype(float))
        testMDP.features[0, :] = tree_states
        testMDP.features[1, :] = red_states
        testMDP.features[2, :] = reward_states

        feature_maps.append(testMDP.features)
        opponent_trajectory_dict = {}
        agent_trajectory_dict = {}

        testEnvironment = HexEnvironment(testMDP, [testAgent1, testAgent2])

        for opponent_policy in ['solve', 'unknown']:

            # Set up MCTS
            mcts = MCTS(testMDP, [testAgent2, testAgent2])

            # Run
            env = deepcopy(testEnvironment)

            game_trajectory = {'agent': [env.agents[1].position], 'opponent': [env.agents[0].position]}
            reward_gained = 0
            caught = 0
            
            predator_min_moves, predator_max_moves = tuple([int(i) for i in parameters.predator_moves[1:-1].split(',')])
            predator_max_moves += 1

            # Calculate predator Q values for all possible prey positions
            opponent_q_values = solve_all_value_iteration(testMDP.sas, np.array(predator_reward_function), testMDP.features, -1, discount=0.9, tol=1e-8, max_iter=500)

            for move in tqdm(range(parameters.n_turns)):
                mcts.reset()

                # Solve for prey using MCTS
                if opponent_policy == 'solve':
                    actions, action_values, states = mcts.fit(env.agents[1].position, env.agents[0].position, n_iter=parameters.n_mcts, C=2, min_opponent_moves=predator_min_moves, max_opponent_moves=predator_max_moves, 
                                                            n_steps=parameters.n_turns*3, opponent_policy_method='solve', caught_cost=parameters.caught_cost, opponent_q_values=opponent_q_values)
                elif opponent_policy == 'unknown':
                    actions, action_values, states = mcts.fit(env.agents[1].position, env.agents[0].position, n_iter=parameters.n_mcts, C=2, min_opponent_moves=predator_min_moves, max_opponent_moves=predator_max_moves, 
                                                            n_steps=parameters.n_turns*3, opponent_policy_method=None, caught_cost=parameters.caught_cost)
                    
                # Get best next state
                next_state = states[np.argmax(action_values)]
                game_trajectory['agent'].append(next_state)

                # Move prey
                env.move_agent(1, next_state)

                # Solve MDP for predator - done every move to allow for dependence on prey position
                env.fit_agent(0, method='numba', show_progress=False)
                # Move predator
                game_trajectory['opponent'] += env.agents[0].generate_trajectory(n_steps=np.random.randint(predator_min_moves, predator_max_moves), start_state=game_trajectory['opponent'][-1])[1:]
                env.move_agent(0, game_trajectory['opponent'][-1])

                # If caught
                if next_state in game_trajectory['opponent']:
                    print("CAUGHT")
                    caught += 1
                    reward_gained += parameters.caught_cost
                    break   

                # Remove reward
                if env.mdp.features[2, next_state] == 1:
                    reward_gained += 1
                    env.mdp.features[2, next_state] = 0

            print(reward_gained)
            run_rewards.append(reward_gained)
            run_caught.append(caught)

            opponent_trajectory_dict[opponent_policy] = game_trajectory['agent']
            agent_trajectory_dict[opponent_policy] = game_trajectory['opponent']

        agent_trajectories.append(agent_trajectory_dict)
        opponent_trajectories.append(opponent_trajectory_dict)

    # Save output
    if not os.path.exists('../data/mcts_simulations/v{0}'.format(parameters.version)):
        os.makedirs('../data/mcts_simulations/v{0}'.format(parameters.version))
    if not os.path.exists('../data/mcts_simulations/v{0}/game_info'.format(parameters.version)):
        os.makedirs('../data/mcts_simulations/v{0}/game_info'.format(parameters.version))
  
    # Results of simulation
    out_df = pd.DataFrame({
        'entropy': [parameters.entropy] * parameters.n_starting_positions * 2,
        'overlap': [parameters.overlap] * parameters.n_starting_positions * 2,
        'entropy': [parameters.entropy] * parameters.n_starting_positions * 2,
        'n_trees': [parameters.n_trees] * parameters.n_starting_positions * 2,
        'n_turns': [parameters.n_turns] * parameters.n_starting_positions * 2,
        'opponent_policy': ['solve', 'unknown'] * parameters.n_starting_positions,
        'n_mcts': [parameters.n_mcts] * parameters.n_starting_positions * 2,
        'n_rewards': [parameters.n_rewards] * parameters.n_starting_positions * 2,
        'predator_reward_function': [parameters.predator_reward_function] * parameters.n_starting_positions * 2,
        'reward_gained': run_rewards,
        'caught': run_caught,
        'runID': [UID] * parameters.n_starting_positions * 2,
        'caught_cost': [parameters.caught_cost] * parameters.n_starting_positions * 2,
        'predator_start_location': [parameters.predator_start_location] * parameters.n_starting_positions * 2,
        'predator_min_moves': [predator_min_moves] * parameters.n_starting_positions * 2,
        'predator_max_moves': [predator_max_moves - 1] * parameters.n_starting_positions * 2
    })

    out_df.to_csv('../data/mcts_simulations/v{0}/run-{1}.csv'.format(parameters.version, UID))

    # Environment details and movement
    game_info = {'features': feature_maps,
                'agent_trajectories': agent_trajectories,
                'opponent_trajectories': opponent_trajectories}

    joblib.dump(game_info, '../data/mcts_simulations/v{0}/game_info/game_info_run-{1}'.format(parameters.version, UID))
