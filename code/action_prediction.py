import numpy as np
from maMDP.algorithms.action_selection import MaxActionSelector, ActionSelector
from maMDP.algorithms.dynamic_programming import ValueIteration
from maMDP.mdp import MDP, HexGridMDP
from maMDP.algorithms.policy_learning import BaseGeneralPolicyLearner, TDGeneralPolicyLearner
from numpy.lib.type_check import nan_to_num
from scipy.stats import zscore
from scipy.optimize import minimize
from typing import List, Union, Dict, Tuple
import pandas as pd
from copy import deepcopy

def minmax_scale(X, min_val=0, max_val=1):
    """ Adapted copy of https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html"""

    X[np.isinf(X)] = np.nan
    X_std = (X - np.nanmin(X, axis=1)[:, None]) / ((np.nanmax(X, axis=1)[:, None] - np.nanmin(X, axis=1)[:, None]) + 1e-10)
    X_scaled = X_std * (max_val - min_val) + min_val

    return X_scaled


class VIPolicyLearner(BaseGeneralPolicyLearner):
    """ Used to make VI-based action prediction work more nicely"""

    def __init__(self, VI_instance:ValueIteration, reward_weights:np.ndarray, refit:bool=True):
        """
        Estimates Q values for actions 

        Args:
            VI_instance (ValueIteration): Instantiated instance of the value iteration algorithm.
            reward_weights (np.ndarray): Reward weights used to calculate reward function for VI.
            refit (bool, optional): If true, calling fit() refits the model even if it was already fit. Otherwise, 
            fit() does nothing. Defaults to True.
        """

        self.VI = VI_instance
        self.reward_weights = reward_weights
        self.q_values = None
        self.fit_complete = False
        self.refit = refit
        
    def reset(self):
        self.q_values = None
        self.fit_complete = False

    def fit(self, mdp:Union[MDP, List[MDP]], trajectories:list):
        """Estimates Q values

        Args:
            mdp (Union[MDP, List[MDP]]): MDP in which the agent is acting.
            trajectories (list): List of trajectories. Not used but retained for compatibility.
        """

        if not self.fit_complete or self.refit:
            self.VI.fit(mdp, self.reward_weights, None, None)
            self.q_values = self.VI.q_values

            self.fit_complete = True

    def get_q_values(self, state: int) -> np.ndarray:
        """
        Returns Q values for each action in a given state.

        Args:
            state (int): State to get Q values for

        Returns:
            np.ndarray: Q values for each action in the provided state.
        """

        Q_values = self.q_values[state]
        Q_values[np.isinf(Q_values)] = 0  # Remove inf for invalid actions

        return Q_values


class CombinedPolicyLearner(BaseGeneralPolicyLearner):

    def __init__(self, model1:BaseGeneralPolicyLearner, model2:BaseGeneralPolicyLearner, W:float=0.5, scale:bool=True,#
                refit_model1:bool=True, refit_model2:bool=True):
        """
        Produces a weighted combination of Q value estimates from two models.

        Args:
            model1 (BaseGeneralPolicyLearner): First model.
            model2 (BaseGeneralPolicyLearner): Second model.
            W (float, optional): Weighting parameter, lower values give Model 1 more weight. Defaults to 0.5.
            scale (bool, optional): If true, Q values from each model are minmax scaled to enable comparability between the two models. 
            Defaults to True.
            refit_model1 (bool, optional): If true, Model 1 is re-fit every time fit() is called. Otherwise, the Q values from
            prior fit() calls are reused. Defaults to True.
            refit_model2 (bool, optional): If true, Model 2 is re-fit every time fit() is called. Otherwise, the Q values from
            prior fit() calls are reused. Defaults to True.
        """

        self.model1 = model1
        self.model2 = model2

        self.model1_fit_complete = False
        self.model2_fit_complete = False
        self.fit_complete = False
        
        self.refit_model1 = refit_model1
        self.refit_model2 = refit_model2

        if not 0 <= W <= 1:
            raise ValueError("W must be between 0 and 1 (inclusive)")

        self.W = W
        self.scale = scale

    def reset(self):

        self.model1.reset()
        self.model2.reset()

        self.model1_fit_complete = False
        self.model2_fit_complete = False

        self.fit_complete = False

    def fit(self, mdp:Union[MDP, List[MDP]], trajectories:list):
        """Estimates Q values

        Args:
            mdp (Union[MDP, List[MDP]]): MDP in which the agent is acting.
            trajectories (list): List of trajectories. NOTE: is only implemented for a single pair of states.

        """

        if len(trajectories) > 1:
            raise NotImplementedError()
        if len(trajectories[0]) > 2:
            raise NotImplementedError()

        # Fit both models
        if not self.model1_fit_complete or self.refit_model1:
            self.model1.fit(mdp, trajectories)
            self.model1_fit_complete = True
        if not self.model2_fit_complete or self.refit_model2:
            self.model2.fit(mdp, trajectories)
            self.model2_fit_complete = True

        self.fit_complete = True

    def get_q_values(self, state: int) -> np.ndarray:
        """
        Returns Q values for each action in a given state.

        Args:
            state (int): State to get Q values for

        Returns:
            np.ndarray: Q values for each action in the provided state.
        """

        model1_Q = self.model1.get_q_values(state)
        model2_Q = self.model2.get_q_values(state)

        if self.scale:
            model1_Q_scaled = minmax_scale(model1_Q[None, :]).squeeze()
            model2_Q_scaled = minmax_scale(model2_Q[None, :]).squeeze()
        else:
            model1_Q_scaled = model1_Q
            model2_Q_scaled = model2_Q

        overall_Q = (1-self.W) * np.nan_to_num(model1_Q_scaled) + self.W * np.nan_to_num(model2_Q_scaled)

        return overall_Q


def nan_softmax(x:np.ndarray, return_nans:bool=False) -> np.ndarray:
    """
    Softmax function, ignoring NaN values. Expects a 2D array of Q values at different observations.

    This is important because for some states certain actions are invalid, so we need to ignore them when
    calculating the softmax rather than letting them influence the probabilities of other actions.

    Args:
        x (np.ndarray): Array of action values. Shape = (observations, actions)
        return_nans (bool, optional): If true, NaN values are returned as NaN, 
        otherwise they are replaced with zeros. Defaults to False.

    Returns:
        np.ndarray: Array of probabilities, ignoring NaN values.
    """
    
    if not x.ndim == 2:
        raise AttributeError("Only works on 2D arrays")

    x_ = np.exp(x) / np.nansum(np.exp(x), axis=1)[:, None]
    
    if return_nans:
        return x_
    else:
        x_[np.isnan(x_)] = 0
        return x_

def prediction_likelihood(q:np.ndarray, pred_actions:List[int]) -> float:
    """
    Calculates categorical likelihood.

    Args:
        q (np.ndarray): Array of Q values for each action at each observation, shape (observations, actions).
        pred_actions (List[int]): List of observed actions, one per observation.

    Returns:
        float: Log likelihood of the observed actions given the provided Q values.
    """
    
    assert len(pred_actions) == q.shape[0], 'Different numbers of predicted actions ({0}) and Q values ({1})'.format(len(pred_actions), q.shape[0])

    # Convert predicted actions to int
    pred_actions = np.array(pred_actions).astype(int).tolist()

    # Scale Q values so they're all on the same scale regardless of the model
    q = minmax_scale(q, max_val=5)  # Using 5 (arbitrarily) has same effect as reducing decision noise

    action_p = nan_softmax(q, return_nans=True)

    logp = np.nansum(np.log((action_p[range(len(pred_actions)), pred_actions]) + 1e-8))
    if np.isinf(logp):
        raise ValueError('Inf in logp')
        
    return logp

def fit_policy_learning(alpha:float, args:List) -> float:
    """
    Fits a policy learning model. Intended for use with scipy.optimize.minimize. 

    Args:
        alpha (float): Learning rate.
        args (List): Other arguments. 1: Predator trajectories, 2: MDPs, 
        3: Subject's predicted actions, 4: Whether to use generalisation kernel

    Returns:
        float: Log likelihood
    """

    if np.isnan(alpha):
        alpha = 0.001
        
    predator_t, target_mdp, predicted_a, kernel = args
    
    _, Q_estimates, _ = action_prediction_envs(predator_t, target_mdp, 
                                               TDGeneralPolicyLearner(learning_rate=alpha, kernel=kernel), 
                                               action_selector=MaxActionSelector(seed=123))

    logp = prediction_likelihood(np.stack(Q_estimates).flatten(), np.stack(predicted_a).flatten())

    return -logp

def fit_combined_model(W:float, args:List):
    """
    Fits a combined policy learning/value iteration model without estimating a learning rate for the policy learner. 
    Intended for use with scipy.optimize.minimize. 

    Args:
        W (float): Weighting parameter, higher = higher weighting of policy learning.
        args (List): Other arguments. 1: Predator trajectory, 2: MDP, 
        3: Subject's predicted actions, 4: Value iteration model, 5: Learning rate, 6: Whether to scale Q values

    Returns:
        float: Log likelihood
    """

    # Avoid NaNs causing problems
    if np.isnan(W):
        W = 0.001
        
    predator_t, target_mdp, predicted_a, model1, learning_rate, decay = args
    
    model2 = TDGeneralPolicyLearner(learning_rate=learning_rate, decay=decay)
    
    _, Q_estimates, _ = action_prediction(predator_t, target_mdp, 
                                         CombinedPolicyLearner(model1, model2, W=W), 
                                         action_selector=MaxActionSelector(seed=123))

    logp = prediction_likelihood(Q_estimates, predicted_a)    
    
    return -logp

def fit_combined_model_learning_rate(X: Tuple[float], args:List):
    """
    Fits a combined policy learning/value iteration model, estimating a learning rate for the policy learner. 
    Intended for use with scipy.optimize.minimize. 

    Args:
        X (Tuple): Weighting parameter and learning rate.
        args (List): Other arguments. 1: Predator trajectory, 2: MDP, 
        3: Subject's predicted actions, 4: Value iteration model, 5: Whether to use a generalisation kernel.

    Returns:
        float: Log likelihood
    """

    W, alpha = X
    
    if np.isnan(W):
        W = 0.001
    if np.isnan(alpha):
        W = 0.001
        
    predator_t, target_mdp, predicted_a, model1, kernel = args
    
    model2 = TDGeneralPolicyLearner(learning_rate=alpha, kernel=kernel)
    
    _, Q_estimates, _ = action_prediction(predator_t, target_mdp, 
                                          CombinedPolicyLearner(model1, model2, W=W), 
                                          action_selector=MaxActionSelector(seed=123))
    
    logp = prediction_likelihood(Q_estimates, predicted_a)    
    
    return -logp

def nan_to_zero(x:np.ndarray):
    """Removes nans and infs from an array"""
    x[np.isnan(x)] = 0
    x[np.isinf(x)] = 0
    return x


def action_prediction_envs(trajectories:List[List[int]], mdps:List[MDP], policy_model:BaseGeneralPolicyLearner,
                           n_predictions:int=2, action_selector:ActionSelector=None, step_reset:bool=False,
                           env_reset:bool=False) -> Union[List, List, List]:
    """
    Estimates an agent's Q values for each action in each state of a trajectory of states, and makes predictions about its actions. This 
    is run for a series of environments.

    By default, assumes a situation where a prediction is being made every 2 moves the agent makes. Q values are estimated every move,
    but predictions are based on the estimated Q values from the first state, and the returned Q values are only updated every `n_predictions`.

    Args:
        trajectories (List[List[int]]): List of state trajectories, one per environment.
        mdps (List[MDP]): List of MDPs in which the agent is acting, one per environment. Each entry can also be a list of MDPs, 
        one per step in the trajectory, to allow different features at each step. If a list of MDPs is supplied, the transition 
        function of the first MDP is used for all, the only thing that changes is the features.
        policy_model (BaseGeneralPolicyLearner): Model used to estimate Q values
        n_predictions (int, optional): Number of predictions to make at a time. Defaults to 2.
        action_selector (ActionSelector, optional): Action selection algorithm. Defaults to None.
        step_reset (bool, optional): If true, the policy learning algorithm is reset at each step. Defaults to False.
        env_reset (bool, optional): If true, the policy learning algorithm is reset at each environment. Defaults to False.

    Returns:
        Union[List, List, List]: Returns lists of most recent Q value estimates for each action, Q estimates at every step,
        and predicted actions at every step for each environemnt.
    """

    all_Q = []
    all_Q_estimates = []
    all_predictions = []

    # Loop through environments
    for n, trajectory in enumerate(trajectories):
        
        # Reset before each environment if needed
        if env_reset:
            policy_model.reset()
                
        Q, Q_estimates, predictions = action_prediction(trajectory, mdps[n], 
                                                        policy_model, 
                                                        n_predictions,
                                                        action_selector,
                                                        reset=step_reset)

        all_Q.append(Q)
        all_Q_estimates.append(Q_estimates)
        all_predictions.append(predictions)

    return all_Q, all_Q_estimates, all_predictions


def action_prediction(trajectory:List[int], mdp:MDP, policy_model:BaseGeneralPolicyLearner, n_predictions:int=2, 
                      action_selector:ActionSelector=None, reset:bool=False) -> Union[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimates an agent's Q values for each action in each state of a trajectory of states, and makes predictions about its actions.

    By default, assumes a situation where a prediction is being made every 2 moves the agent makes. Q values are estimated every move,
    but predictions are based on the estimated Q values from the first state, and the returned Q values are only updated every `n_predictions`.

    Args:
        trajectory (List[int]): Trajectory of states
        mdp (MDP): MDP in which the agent is acting. Can also be a list of MDPs, one per step in the trajectory, to allow different 
        features at each step. If a list of MDPs is supplied, the transition function of the first MDP is used for all, the only thing
        that changes is the features.
        policy_model (BaseGeneralPolicyLearner): Model used to estimate Q values
        n_predictions (int, optional): Number of predictions to make at a time. Defaults to 2.
        action_selector (ActionSelector, optional): Action selection algorithm. Defaults to MaxActionSelector().
        reset (bool, optional): If true, the policy learning algorithm is reset at each step. Defaults to False.

    Returns:
        Union[np.ndarray, np.ndarray, np.ndarray]: Returns most recent Q value estimates for each action, Q estimates at every step,
        and predicted actions at every step.
    """
    # TODO determine refitting before calling this function and store information in policy model refit attribute
    if action_selector is None:
        action_selector = MaxActionSelector(seed=123)

    if isinstance(mdp, list):
        if not len(mdp) == len(trajectory) - 1:
            raise AttributeError("Must provide same number of MDPs as steps in the trajectory")
    else:
        mdp = [mdp] * len(trajectory)

    # Convert trajectory to actions
    state = mdp[0]._trajectory_to_state_action(trajectory)[:, 0].astype(int)
    action = mdp[0]._trajectory_to_state_action(trajectory)[:, 1].astype(int)

    # Initial Q values
    Q = np.zeros(mdp[0].n_actions)

    # List to store estimated Q values
    Q_estimates = []

    # Predictions
    predictions = []
    predicted_state = state[0]

    # Fit the model to get starting Q values before any observations (if it hasn't already been fit)
    if not policy_model.fit_complete:
        policy_model.fit(mdp[0], [[]])  # Using an empty trajectory means learning models produce Q values of zero for all actions

    # Loop through agent moves
    for n in range(len(action)):

        # First prediction of each turn
        if n % n_predictions == 0:
            # Observed starting state
            start_state = state[n]

            # Preserve model state from first prediction to allow real model to update on every move without affecting predictions
            temp_policy_learner = deepcopy(policy_model)

        else:
            # Otherwise the next prediction follows from the previous predicted one
            start_state = predicted_state

        # Get Q values for this start state
        trial_Q = temp_policy_learner.get_q_values(start_state)

        # Q values for making predictions
        prediction_Q = trial_Q.copy()

        # Set impossible actions to -inf 
        prediction_Q[[i for i in range(mdp[0].n_actions) if not i in mdp[0].state_valid_actions(start_state)]] = -np.inf

        # Add Q values to the list here - this is done before estimating as we want expected Q prior to the observation
        Q_estimates.append(prediction_Q.copy())

        # Get action
        predicted_action = action_selector.get_pi(prediction_Q[None, :])[0]
        # Get resulting state
        predicted_state = np.argmax(mdp[0].sas[start_state, predicted_action, :])

        predictions.append(predicted_state)

        # Q VALUE ESTIMATION - after observing agent move
        if n < len(state): 
            if reset:
                policy_model.reset()
            policy_model.fit(mdp[n], [state[n:n+2]])
    
    Q_estimates = np.stack(Q_estimates)

    return Q, Q_estimates, predictions

# TODO add tests for these functions. Please. It'll save you pain in the long run.


# Make sure VI-based is using a new MDP each step when it should

def fit_action_prediction_model(model_name:str, predator_trajectory:List[int], mdps:List[MDP], 
                                predicted_actions:List[int], model:str, learning_rate:float=None, kernel:bool=False,
                                predator_reward_function:np.ndarray=None) -> Union[float, Dict]:

    # Output parameters as a dict of names and values
    estimated_parameters = {}

    # If model 2 is None, we're only fitting a single model rather than a combination
    if model == 'policy_learning':
        
        # If a fixed learning rate is provided
        if learning_rate is not None:
            
            # Get Q estimates for each action 
            _, Q_estimates, _ = action_prediction(predator_trajectory, mdps, 
                                                    TDGeneralPolicyLearner(learning_rate=learning_rate), 
                                                    action_selector=MaxActionSelector(seed=123))

        else:
            
            # Estimate learning rate
            res = minimize(fit_policy_learning, [0.5], args=[predator_trajectory, mdps, predicted_actions, kernel], 
                            bounds=[(0.001, 0.999)])

            # Save estimated learning rate
            estimated_parameters['learning_rate'] = res.x[0]

            # Get Q estimates for each action 
            _, Q_estimates, _ = action_prediction(predator_trajectory, mdps, 
                                                    TDGeneralPolicyLearner(learning_rate=res.x[0]), 
                                                    action_selector=MaxActionSelector(seed=123))

    elif model == 'value_iteration':

        if predator_reward_function is None:
            raise ValueError("Must provide a predator reward function if using value iteration")

        # Get Q estimates for each action 
        _, Q_estimates, _ = action_prediction(predator_trajectory, mdps, 
                                                VIPolicyLearner(ValueIteration(), predator_reward_function), 
                                                action_selector=MaxActionSelector(seed=123))

    elif model == 'combined':

        # Set up the VI model used in the combined model
        VI_model = VIPolicyLearner(ValueIteration(), predator_reward_function)

        if learning_rate is not None:

            # Estimate W parameter
            res = minimize(fit_combined_model, [0.5], args=[predator_trajectory, mdps, predicted_actions, 
                           VI_model, learning_rate], 
                           bounds=[(0, 1)], options=dict(eps=1e-2))

            # Save estimated learning rate
            estimated_parameters['W'] = res.x[0]

            # Get Q estimates for each action 
            _, Q_estimates, _ = action_prediction(predator_trajectory, mdps, 
                                                 CombinedPolicyLearner(VI_model, 
                                                 TDGeneralPolicyLearner(learning_rate=learning_rate), W=res.x[0], scale=True), 
                                                 action_selector=MaxActionSelector(seed=123))

        else:
            
            # Estimate W and learning rate
            res = minimize(fit_combined_model_learning_rate, [0.5, 0.5], args=[predator_trajectory, mdps, 
                            predicted_actions, VI_model, kernel], bounds=[(0, 1), (0.001, 1)], options=dict(eps=1e-2))

            # Save estimated learning rate
            estimated_parameters['W'] = res.x[0]
            estimated_parameters['learning_rate'] = res.x[1]

            # Get Q estimates for each action 
            _, Q_estimates, _ = action_prediction(predator_trajectory, mdps, 
                                                 CombinedPolicyLearner(VI_model, TDGeneralPolicyLearner(learning_rate=res.x[1]), 
                                                 W=res.x[0], scale=True), 
                                                 action_selector=MaxActionSelector(seed=123))

    # Get log likelihood of model
    log_likelihood = prediction_likelihood(Q_estimates, predicted_actions)

    # Dictionary of outputs
    out_dict = {'model_name': model_name,
                'log_likelihood': log_likelihood,
                'estimated_parameters': estimated_parameters,
                'n_parameters': len(estimated_parameters),
                'Q_estimates': Q_estimates}

    return out_dict

def fit_all_action_prediction_models(predator_dfs:pd.DataFrame, prediction_dfs:pd.DataFrame, 
                                     prey_dfs:pd.DataFrame, environments:Dict, prey_value_conditions:List=[3]) -> None:

    # THIS ALL ASSUMES PREDATOR MAKES TWO MOVES

    # Dictionaries for outputs
    accuracy = {}
    log_lik = {}
    w_values = {}

    # Loop over subjects
    for subject in predator_dfs['subjectID'].unique():

        # Get subject-level data
        subject_predator_df = predator_dfs[predator_dfs['subjectID'] == subject]
        subject_prediction_df = prediction_dfs[prediction_dfs['subjectID'] == subject]
        subject_prey_df = prey_dfs[prey_dfs['subjectID'] == subject]

        # Ignore subjects with missing trials and subjects who somehow did multiple conditions
        if not (subject_predator_df['trial'].diff() > 1).any() or not len(subject_predator_df['condition'].unique()) == 1:
            
            accuracy[subject] = {}
            log_lik[subject] = {}
            w_values[subject] = {}
        
            # Loop through environments
            for env in subject_predator_df['env'].unique():
                
                # Get condition for this subject
                condition = subject_predator_df['condition'].tolist()[0]
                
                # Add an entry to the dictionary for this condition if it doesn't already exist
                if not condition in accuracy[subject]:
                    accuracy[subject][condition] = {}
                    log_lik[subject][condition] = {}
                    w_values[subject][condition] = {}
                
                # Get data for this environment
                subject_env_predator_df = subject_predator_df[subject_predator_df['env'] == env]
                subject_env_prediction_df = subject_prediction_df[subject_prediction_df['env'] == env]
                subject_env_prey_df = subject_prey_df[subject_prey_df['env'] == env]
                
                # Add predator starting state and get trajectory
                predator_trajectory = [environments[condition][env].agents['Predator_1'].position] + subject_env_predator_df['cellID'].tolist()
                
                # The trajectory predicted by the subject
                predicted_trajectory = subject_env_prediction_df['cellIDpred'].tolist()

                # Get the MDP for the environment
                mdp = environments[condition][env].mdp
                
                # If the predator trajectory and predicted trajectory have equal numbers of moves
                if len(predator_trajectory) - 1 == len(predicted_trajectory) == 20:
                    
                    # For conditions where the predator values the prey, we need to provide a new MDP at each step
                    # to account for 
                    if condition in prey_value_conditions:
                        
                        # GET PREY MOVES (FOR CONDITION 3)
                        prey_moves = subject_env_prey_df['cellID'].values[1:]  # Ignore starrting state?
                        
                        # Create a list of MDPs
                        mdps = []

                        for i in range(len(prey_moves)):

                            # Get environment and move the prey agent
                            trial_env = environments[condition][env]
                            trial_env.set_agent_position('Prey_1', prey_moves[i])

                            # Create a new MDP to ensure we get a copy not a reference to the original one
                            new_mdp = HexGridMDP(trial_env.mdp.features.copy(), trial_env.mdp.walls, shape=trial_env.mdp.shape)
                            mdps += [new_mdp, new_mdp]  # Same mdp for 2 moves as the predator makes 2 moves at a time
                    else:
                        # Otherwise, the "list" is just a single MDP for the entire task
                        mdps = mdp

                    # Get predicted actions from predicted states
                    predicted_actions = []

                    for i in np.arange(0, len(predicted_trajectory), 2):
                        t = [predator_trajectory[i], *predicted_trajectory[i: i+2]]
                        predicted_actions += mdp._trajectory_to_state_action(t)[:, 1].astype(int).tolist()
        
                    # Set up dictionaries for recording outputs
                    accuracy[subject][condition][env] = {}
                    log_lik[subject][condition][env] = {}