import pandas as pd
import argparse
import os
from maMDP.algorithms.base import Null
from maMDP.algorithms.mcts import MCTS
from maMDP.algorithms.action_selection import SoftmaxActionSelector
from maMDP.env_io import hex_environment_from_dict
import json
import time
from datetime import datetime
import numpy as np

# import sys
# sys.path.insert(0, '../code')
from ibs import IBSEstimator

if __name__ == "__main__":

    # Get command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('params_file', type=str)
    args = parser.parse_args()


    # Use SLURM job ID to determine which subject to fit
    try:
        runID = int(os.environ['SLURM_ARRAY_TASK_ID']) - 1
    except:
        runID = 29

    # Get fitting parameters
    # Get subject IDs and parameter values
    # run_params = pd.read_csv('data/planning_model_fit_runs.csv')
    run_params = pd.read_csv(args.params_file)
    run_params = run_params.iloc[runID].to_dict()

    print(run_params)

    # Determine action selection
    if run_params['softmax_temperature'] > 0:
        action_selection = 'softmax'
    else:
        action_selection = 'max'

    # Get game info
    with open('data/game_info/experiment-{0}/condition_{1}.json'.format(run_params['experiment'], run_params['condition']), 'r') as f:
        game_info = json.load(f)
    
    envs = [hex_environment_from_dict(env, ['Dirt', 'Trees', 'Reward']) for env in game_info['environments']]

    # Get response data
    response_data = pd.read_csv('data/response_data/experiment-{0}/condition-{1}/exp-{0}_cond-{1}_response_data.csv'.format(run_params['experiment'], run_params['condition']))

    # Select subject
    subject = run_params['subjectID']

    # Get response data for this subject
    subject_response_data = response_data[response_data['subjectID'] == subject]

    prey_df = subject_response_data[subject_response_data['agent'] == 'prey']
    predator_df = subject_response_data[subject_response_data['agent'] == 'prey']

    # Loop through games and extract environment & move info
    movement_arrays = []
    state_history = {}
    environments = {}

    # Which environments to use
    if not 'start_env' in run_params:
        start_env = 0
    else:
        start_env = run_params['start_env']
    
    if not 'end_env' in run_params:
        end_env = len(subject_response_data['env'].unique()) + 1
    else:
        end_env = run_params['end_env']

    # Get settings
    with open('settings/ibs_fitting_settings.json', 'r') as f:
        fit_settings = json.load(f) 

    for env in sorted(subject_response_data['env'].unique())[start_env:end_env]:
        
        env_string = 'Env_{0}'.format(env)
        state_history[env_string] = {}
        environments[env_string] = {}
    
        sub_prey_df = prey_df[prey_df['env'] == env].sort_values(['trial', 'response_number'])
        sub_predator_df = predator_df[predator_df['env'] == env].sort_values(['trial', 'response_number'])

        # Check that everything is in order
        assert np.all(sub_prey_df.trial.diff()[1:] <= 1), 'Trials are not in order or there are trials missing, jumps of more than 1'
        assert np.all(sub_prey_df.trial.diff()[1:] >= 0), 'Trials are not in order or there are trials missing, jumps of less than 0'

        sub_prey_states = [envs[env].agents['Prey_1'].position] + sub_prey_df['cellID'].tolist()  

        # Predator states = the state the predator ended up at (i.e. the final response number)
        sub_predator_states = [envs[env].agents['Predator_1'].position] + sub_predator_df.loc[sub_predator_df['response_number'] == sub_predator_df['response_number'].max(), 'cellID'].tolist()
        sub_predator_states = sub_predator_states[:-1]  # Remove final state as this is after the prey has made its final move

        # Repeat predator moves for each prey turn
        n_prey_moves = sub_prey_df['response_number'].max() + 1
        sub_predator_states = np.repeat(sub_predator_states, n_prey_moves).tolist()

        # Number of prey states should be 1 less than number of predator states
        # Each prey state has a corresponding predator state, plus there is the prey's starting state at index 0
        # The starting state is only used to set up the environment, and is not selected later when fitting
        if not len(sub_predator_states) == len(sub_prey_states) - 1:
            print('Unequal number of predator & prey states, env {0}, subject {1}'.format(env, subject))
        
        else:
            state_history[env_string] = {}
            state_history[env_string]['prey'] = sub_prey_states
            state_history[env_string]['predator'] = sub_predator_states

            environments[env_string] = []
            
            for n in range(run_params['n_moves']): 

                game_info['environments'][env]['agents'][0]['Predator_1']['position'] = sub_predator_states[n]
                game_info['environments'][env]['agents'][1]['Prey_1']['position'] = sub_prey_states[n]

                try:
                    subject_env = hex_environment_from_dict(game_info['environments'][env], ['Dirt', 'Trees', 'Reward'])
                    
                    # Set up model

                    # Full interactive MCTS, accounting for predator's reward weights
                    if run_params['model'] == 'MCTS_interactive':
                        subject_env.agents['Prey_1'].algorithm = MCTS(interactive=True, 
                                                                        n_iter=fit_settings['MCTS_iterations'], 
                                                                        caching=True,
                                                                        opponent_action_selection=action_selection,
                                                                        caught_cost=-10,
                                                                        reset_cache=True,
                                                                        softmax_temperature=run_params['softmax_temperature'],
                                                                        n_steps=10,
                                                                        C=run_params['C'])

                    # Interactive MCTS, but assuming predator's actions are random
                    elif run_params['model'] == 'MCTS_interactive_random':
                        subject_env.agents['Prey_1'].algorithm = MCTS(interactive=True, 
                                                                        n_iter=fit_settings['MCTS_iterations'], 
                                                                        caching=True,
                                                                        opponent_action_selection=action_selection,
                                                                        caught_cost=-10,
                                                                        reset_cache=True,
                                                                        softmax_temperature=run_params['softmax_temperature'],
                                                                        opponent_policy_method='random',
                                                                        n_steps=10,
                                                                        C=run_params['C'])

                    # Non-interactive MCTS, ignoring the predator
                    elif run_params['model'] == 'MCTS':
                        subject_env.agents['Prey_1'].algorithm = MCTS(interactive=False, 
                                                                        n_iter=fit_settings['MCTS_iterations'], 
                                                                        caching=True,
                                                                        opponent_action_selection=action_selection,
                                                                        caught_cost=-10,
                                                                        reset_cache=True,
                                                                        softmax_temperature=run_params['softmax_temperature'],
                                                                        n_steps=10,
                                                                        C=run_params['C'])

                    # Null model, random action selection
                    elif run_params['model'] == 'null_model':
                        subject_env.agents['Prey_1'].algorithm = Null()

                    else:
                        raise ValueError("{0} is an invalid model string".format(run_params['model']))
                    
                    subject_env.agents['Prey_1'].action_selector = SoftmaxActionSelector()
                    environments[env_string].append(subject_env)

                # Skip if subject got caught before 3 moves
                except IndexError:
                    pass


    # Inverse binomial sampling

    # Dictionary to store output
    fitting_output = {
        'environment': [],
        'll': [],
        'subjectID': subject,
        'n_repeats': fit_settings['n_repeats'],
        'stopping_point': fit_settings['stopping_point'],
        'start_time': datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    }

    # Fit for each environment
    start_time = time.time()

    for env in sorted(subject_response_data['env'].unique())[start_env:end_env]:

        print('Fitting environment {0}'.format(env))

        # Set up IBS
        n_jobs = len(os.sched_getaffinity(0))
        ibs = IBSEstimator(n_jobs, fit_settings['n_repeats'], fit_settings['stopping_point'], run_params['max_planning_steps'])
        ibs.fit(environments['Env_{0}'.format(env)][:run_params['n_moves']], 'Prey_1', state_history['Env_{0}'.format(env)]['prey'][1:1 + run_params['n_moves']])

        fitting_output['environment'].append(env)
        fitting_output['ll'].append(ibs.ll.mean())

    fitting_output['time_taken'] = time.time() - start_time
    fitting_output['softmax_temperature'] = run_params['softmax_temperature']
    fitting_output['model'] = run_params['model']
    fitting_output['C'] = run_params['C']

    # Convert to dataframe
    fitting_output = pd.DataFrame(fitting_output)

    # Set up output path
    if 'environments' in run_params:
        output_path = 'data/IBS_fits/experiment_{0}/condition_{1}/environments_{2}'.format(run_params['experiment'], run_params['condition'], run_params['environments'])
    else:
        output_path = 'data/IBS_fits/experiment_{0}/condition_{1}'.format(run_params['experiment'], run_params['condition'])
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Save to CSV
    fitting_output.to_csv(os.path.join(output_path, 'jobID-{2}_subject-{0}_model-{1}.csv'.format(subject, run_params['model'], runID)), index=False)

    print("FINISHED")



