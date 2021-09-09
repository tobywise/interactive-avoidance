"""
Runs model fitting for subjects' predictions about the predator's actions.
"""

from maMDP.mdp import HexGridMDP
from maMDP.algorithms.policy_learning import *
from tqdm import tqdm
import pandas as pd
import numpy as np
from typing import Dict
# import sys
# sys.path.append(".") 
from action_prediction import *


def BIC(n_params:int, log_lik:float, n_obs:int):

    return n_params * np.log(n_obs) - 2 * log_lik

def fit_output_to_dataframe(accuracy_dict:Dict, log_lik_dict:Dict, bic_dict:Dict,alpha_values_dict:Dict, 
                            w_values_dict:Dict, subjectID:str, condition:str):

    out = {'env': [], 'model': [], 'accuracy': [], 'log_lik': [], 'BIC': [], 'alpha_values': [], 'w_values': []}

    for env in accuracy_dict.keys():
        for model in accuracy_dict[env].keys():
            out['env'].append(env)
            out['model'].append(model)
            out['accuracy'].append(accuracy_dict[env][model])
            out['log_lik'].append(log_lik_dict[env][model])
            out['BIC'].append(bic_dict[env][model])
            out['alpha_values'].append(alpha_values_dict[env][model])
            out['w_values'].append(w_values_dict[env][model])

    out['subjectID'] = subjectID
    out['condition'] = condition

    out = pd.DataFrame(out)

    return out

def fit_subject_predictions(predator_moves:pd.DataFrame, prey_moves:pd.DataFrame, predictions:pd.DataFrame, env_info:Dict) -> pd.DataFrame:
    """
    Fits a series of models to subjects' predictions about the predator's movements.

    Args:
        predator_moves (pd.DataFrame): A dataframe representing the states visited by the predator
        prey_moves (pd.DataFrame): A dataframe representing the states visited by the prey (i.e. the subject's moves)
        predictions (pd.DataFrame): Subject's predictions about the predator's moves
        env_info (Dict): Information about each environment

    Returns:
        pd.DataFrame: A dataframe of model fitting outputs
    """
    

    # Make sure everything is in the right order
    predator_moves = predator_moves.sort_values(['subjectID', 'exp', 'condition', 'env', 'trial', 'response_number']).reset_index(drop=True)
    prey_moves = prey_moves.sort_values(['subjectID', 'exp', 'condition', 'env', 'trial', 'response_number']).reset_index(drop=True)
    predictions = predictions.sort_values(['subjectID', 'exp', 'condition', 'env', 'trial', 'response_number']).reset_index(drop=True)

    # Check if any data is missing
    if not (predator_moves['trial'].diff() > 1).any() or not len(predator_moves['condition'].unique()) == 1:

        # Dictionaries to store outputs
        accuracy = {}
        log_lik = {}
        bic = {}
        alpha_values = {}
        w_values = {}
        model_predictions = {}
        
        # Get condition that this subject is in
        condition = predator_moves['condition'].tolist()[0]

        # Get subject ID
        subject = predator_moves['subjectID'].tolist()[0]

        # Set up models
        # TODO deal with need to refit
        # TODO need to estimate parameters across all envs

        # Policy repitition
        policy_repetition_model = TDGeneralPolicyLearner(learning_rate=1, decay=0)

        # Policy learning
        policy_learning_model = TDGeneralPolicyLearner(learning_rate=res.x[0])

        # Policy learning & generalisation
        policy_learning_generalisation_model = TDGeneralPolicyLearner(learning_rate=res.x[0], kernel=True)

        # Model-based goal inference
        goal_inference_model = VIPolicyLearner(VI, env_info[condition][env].agents['Predator_1'].reward_function)

        # Combined goal inference and policy repitition
        model1 = VIPolicyLearner(ValueIteration(), env_info[condition][env].agents['Predator_1'].reward_function)
        combined_GI_rep_model = CombinedPolicyLearner(VIPolicyLearner(ValueIteration(), 
                                                                      env_info[condition][env].agents['Predator_1'].reward_function), 
                                                      TDGeneralPolicyLearner(learning_rate=1, decay=0), 
                                                      W=res.x[0], scale=True)

        # Combined goal inference and policy learning
        combined_GI_learn_model = CombinedPolicyLearner(VIPolicyLearner(ValueIteration(), 
                                                                        env_info[condition][env].agents['Predator_1'].reward_function), 
                                                        TDGeneralPolicyLearner(learning_rate=res.x[1]), 
                                                        W=res.x[0], scale=True)

        # Combined goal inference and policy learning with generalisation
        model2 = TDGeneralPolicyLearner(learning_rate=res.x[1], kernel=True)
        combined_GI_learn_gen_model = CombinedPolicyLearner(VIPolicyLearner(ValueIteration(), 
                                                                            env_info[condition][env].agents['Predator_1'].reward_function), 
                                                            TDGeneralPolicyLearner(learning_rate=res.x[1], kernel=True), 
                                                            W=res.x[0], scale=True),

        # Loop through environments
        for env in predator_moves['env'].unique():
            
            # Get data for this environment
            env_predator_df = predator_moves[predator_moves['env'] == env]
            env_prediction_df = predictions[predictions['env'] == env]
            env_prey_df = prey_moves[prey_moves['env'] == env]
            
            # Get trajectories
            predator_trajectory = [env_info[condition][env].agents['Predator_1'].position] + env_predator_df['cellID'].tolist()
            predicted_trajectory = env_prediction_df['cellID'].tolist()

            # Get MDP representing this environment
            mdp = env_info[condition][env].mdp
            
            # Ensure that we have the right number of predictions
            if len(predator_trajectory) - 1 == len(predicted_trajectory) == 20:
                
                # In condition 3, the predator values the prey so we need to update the MDP with the prey's position
                # at every step. Otherwise, the prey's position makes no difference in terms of solving the MDP
                if condition == 3:
                    
                    # Get prey moves
                    prey_moves_array = env_prey_df['cellID'].values[1:]
                    
                    mdps = []

                    for i in range(len(prey_moves_array)):
                        trial_env = env_info[condition][env]
                        trial_env.set_agent_position('Prey_1', prey_moves_array[i])
                        new_mdp = HexGridMDP(trial_env.mdp.features.copy(), trial_env.mdp.walls, shape=(21, 10))
                        mdps += [new_mdp, new_mdp]  # same mdp for 2 moves
                else:
                    mdps = mdp
                
                # GET ACTIONS FROM PREDICTED STATES
                predicted_actions = []

                for i in np.arange(0, len(predicted_trajectory), 2):
                    t = [predator_trajectory[i], *predicted_trajectory[i: i+2]]
                    predicted_actions += mdp._trajectory_to_state_action(t)[:, 1].astype(int).tolist()
    
                # Dictionary to store outputs
                accuracy[env] = {}
                log_lik[env] = {}
                bic[env] = {}
                alpha_values[env] = {}
                w_values[env] = {}
                model_predictions[env] = {}
                
                # POLICY REPETITION
                Q, Q_estimates, predictions_TD = action_prediction(predator_trajectory, mdps, TDGeneralPolicyLearner(learning_rate=1, decay=0), action_selector=MaxActionSelector(seed=123))
                accuracy[env]['repetition'] = np.equal(predicted_trajectory, predictions_TD).sum() / len(predictions_TD)
                model_predictions[env]['repetition'] = predictions_TD
                log_lik[env]['repetition'] = prediction_likelihood(Q_estimates, predicted_actions)
                bic[env]['repetition'] = BIC(0, log_lik[env]['repetition'], len(predicted_trajectory))
                alpha_values[env]['repetition'] = np.nan
                w_values[env]['repetition'] = np.nan
                
                # POLICY LEARNING
                res = minimize(fit_policy_learning, [0.5], args=[predator_trajectory, mdps, predicted_actions, False], bounds=[(0.001, 0.999)])
                Q, Q_estimates, predictions_TD = action_prediction(predator_trajectory, mdp, TDGeneralPolicyLearner(learning_rate=res.x[0]), action_selector=MaxActionSelector(seed=123))
                accuracy[env]['policy_learning'] = np.equal(predicted_trajectory, predictions_TD).sum() / len(predictions_TD)
                model_predictions[env]['policy_learning'] = predictions_TD
                log_lik[env]['policy_learning'] = prediction_likelihood(Q_estimates, predicted_actions)
                bic[env]['policy_learning'] = BIC(1, log_lik[env]['policy_learning'], len(predicted_trajectory))
                alpha_values[env]['policy_learning'] = res.x[0]
                w_values[env]['policy_learning'] = np.nan
                
                # POLICY LEARNING + generalisation
                res = minimize(fit_policy_learning, [0.5], args=[predator_trajectory, mdps, predicted_actions, True], bounds=[(0.001, 0.999)])
                Q, Q_estimates, predictions_TD = action_prediction(predator_trajectory, mdp, TDGeneralPolicyLearner(learning_rate=res.x[0], kernel=True), action_selector=MaxActionSelector(seed=123))
                accuracy[env]['policy_generalisation'] = np.equal(predicted_trajectory, predictions_TD).sum() / len(predictions_TD)
                model_predictions[env]['policy_generalisation'] = predictions_TD
                log_lik[env]['policy_generalisation'] = prediction_likelihood(Q_estimates, predicted_actions)
                bic[env]['policy_generalisation'] = BIC(1, log_lik[env]['policy_generalisation'], len(predicted_trajectory))
                alpha_values[env]['policy_generalisation'] = res.x[0]
                w_values[env]['policy_generalisation'] = np.nan
                
                # FULL VI
                VI = ValueIteration()
                Q, Q_estimates, predictions_VI = action_prediction(predator_trajectory, mdps, 
                                                                VIPolicyLearner(VI, env_info[condition][env].agents['Predator_1'].reward_function), 
                                                                action_selector=MaxActionSelector(seed=123))
                accuracy[env]['goal_inference'] = np.equal(predicted_trajectory, predictions_VI).sum() / len(predictions_VI)
                model_predictions[env]['goal_inference'] = predictions_VI
                log_lik[env]['goal_inference'] = prediction_likelihood(np.stack(Q_estimates), predicted_actions)
                bic[env]['goal_inference'] = BIC(0, log_lik[env]['goal_inference'], len(predicted_trajectory))
                alpha_values[env]['goal_inference'] = np.nan
                w_values[env]['goal_inference'] = np.nan
                
                # COMBINATION
                VI = ValueIteration()
                model1 = VIPolicyLearner(VI, env_info[condition][env].agents['Predator_1'].reward_function)
                model2 = TDGeneralPolicyLearner()

                res = minimize(fit_combined_model, [0.5], args=[predator_trajectory, mdps, predicted_actions, model1, 1, 0], bounds=[(0, 1)], options=dict(eps=1e-2))
                
                Q, Q_estimates, predictions_combined = action_prediction(predator_trajectory, mdps, 
                                                                CombinedPolicyLearner(model1, TDGeneralPolicyLearner(learning_rate=1, decay=0), W=res.x[0], scale=True), 
                                                                action_selector=MaxActionSelector(seed=123))
                accuracy[env]['combined_repetition'] = np.equal(predicted_trajectory, predictions_combined).sum() / len(predictions_combined)
                model_predictions[env]['combined_repetition'] = predictions_combined
                log_lik[env]['combined_repetition'] = prediction_likelihood(np.stack(Q_estimates), predicted_actions)
                bic[env]['combined_repetition'] = BIC(1, log_lik[env]['combined_repetition'], len(predicted_trajectory))
                alpha_values[env]['combined_repetition'] = np.nan
                w_values[env]['combined_repetition'] = res.x[0]
                
                
                # COMBINATION, LEARNING
                res = minimize(fit_combined_model_learning_rate, [0.5, 0.5], args=[predator_trajectory, mdps, 
                                                                                   predicted_actions, model1, False], bounds=[(0, 1), (0.001, 1)], options=dict(eps=1e-2))
                
                model2 = TDGeneralPolicyLearner(learning_rate=res.x[1])
                Q, Q_estimates, predictions_combined = action_prediction(predator_trajectory, mdps, 
                                                                CombinedPolicyLearner(model1, model2, W=res.x[0], scale=True), 
                                                                action_selector=MaxActionSelector(seed=123))
                accuracy[env]['combined_learning'] = np.equal(predicted_trajectory, predictions_combined).sum() / len(predictions_combined)
                model_predictions[env]['combined_learning'] = predictions_combined
                log_lik[env]['combined_learning'] = prediction_likelihood(np.stack(Q_estimates), predicted_actions)
                bic[env]['combined_learning'] = BIC(2, log_lik[env]['combined_learning'], len(predicted_trajectory))
                alpha_values[env]['combined_learning'] = res.x[1]
                w_values[env]['combined_learning'] = res.x[0]
                
                # COMBINATION, LEARNING + GENERALISATION
                res = minimize(fit_combined_model_learning_rate, [0.5, 0.5], args=[predator_trajectory, mdps, 
                                                                                   predicted_actions, model1, True], bounds=[(0, 1), (0.001, 1)], options=dict(eps=1e-2))
                model2 = TDGeneralPolicyLearner(learning_rate=res.x[1], kernel=True)
                Q, Q_estimates, predictions_combined = action_prediction(predator_trajectory, mdps, 
                                                                CombinedPolicyLearner(model1, model2, W=res.x[0], scale=True), 
                                                                action_selector=MaxActionSelector(seed=123))
                
                accuracy[env]['combined_generalisation'] = np.equal(predicted_trajectory, predictions_combined).sum() / len(predictions_combined)
                model_predictions[env]['combined_generalisation'] = predictions_combined
                log_lik[env]['combined_generalisation'] = prediction_likelihood(np.stack(Q_estimates), predicted_actions)
                bic[env]['combined_generalisation'] = BIC(2, log_lik[env]['combined_generalisation'], len(predicted_trajectory))
                alpha_values[env]['combined_generalisation'] = res.x[1]
                w_values[env]['combined_generalisation'] = res.x[0]

                model_predictions[env]['subject'] = predicted_trajectory

                assert np.all([0 <= i <= 1 for i in accuracy[env].values()]), 'Accuracies out of range'

            else:
                print('Subject {0}, env {1}, cond {2} has mismatch between predicted and observed moves'.format(subject, env, condition))

        out_df = fit_output_to_dataframe(accuracy, log_lik, bic, alpha_values, w_values, subject, condition)
        
        return out_df, model_predictions

    else:
        print('Missing data, skipping')


def fit_prediction_models(predator_moves:pd.DataFrame, prey_moves:pd.DataFrame, predictions:pd.DataFrame, env_info:Dict):

    subjects = list(predator_moves['subjectID'].unique())

    fit_dfs = []
    model_prediction_list = []

    for sub in tqdm(subjects):
        fit, model_predictions = fit_subject_predictions(predator_moves[predator_moves['subjectID'] == sub],
                                                   prey_moves[prey_moves['subjectID'] == sub],
                                                   predictions[predictions['subjectID'] == sub],
                                                   env_info)
        fit_dfs.append(fit)
        model_prediction_list.append(model_predictions)

    all_subject_fit = pd.concat(fit_dfs)

    return all_subject_fit, model_prediction_list





