import numpy as np
from maMDP.algorithms.action_selection import MaxActionSelector, ActionSelector
from maMDP.algorithms.dynamic_programming import ValueIteration
from maMDP.mdp import MDP, HexGridMDP
from maMDP.algorithms.policy_learning import (
    BaseGeneralPolicyLearner,
    TDGeneralPolicyLearner,
)
from numpy.lib.type_check import nan_to_num
from scipy.stats import zscore
from scipy.optimize import minimize
from typing import List, Union, Dict, Tuple
import pandas as pd
from copy import copy, deepcopy


def minmax_scale(X, min_val=0, max_val=1):
    """Adapted copy of https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html"""

    X[np.isinf(X)] = np.nan
    X_std = (X - np.nanmin(X, axis=1)[:, None]) / (
        (np.nanmax(X, axis=1)[:, None] - np.nanmin(X, axis=1)[:, None]) + 1e-10
    )
    X_scaled = X_std * (max_val - min_val) + min_val

    return X_scaled


def check_mdp_equal(mdp1, mdp2):
    """Checks whether two MDPs are the same based on features and transition matrix"""

    if not isinstance(mdp1, MDP) or not isinstance(mdp2, MDP):
        return False

    elif (mdp1.features == mdp2.features).all() and (mdp1.sas == mdp2.sas).all():
        return True
    else:
        return False


class VIPolicyLearner(BaseGeneralPolicyLearner):
    """Used to make VI-based action prediction work more nicely"""

    def __init__(
        self,
        VI_instance: ValueIteration,
        reward_weights: np.ndarray,
        refit_on_new_mdp: bool = True,
        caching: bool = False,
    ):
        """
        Estimates Q values for actions

        Args:
            VI_instance (ValueIteration): Instantiated instance of the value iteration algorithm.
            reward_weights (np.ndarray): Reward weights used to calculate reward function for VI.
            refit_on_new_mdp (bool, optional): If true, refits the model whenever fit() is provided with an MDP that differs
            from the previous one.
        """

        self.VI = VI_instance
        self.reward_weights = reward_weights
        self.q_values = None
        self.refit_on_new_mdp = refit_on_new_mdp
        self.caching = caching
        self.previous_mdp = None
        self.previous_mdp_q_values = {}

    def reset(self):
        self.q_values = None

    def fit(self, mdp: MDP, trajectories: list):
        """Estimates Q values

        Args:
            mdp (MDP): MDP in which the agent is acting.
            trajectories (list): List of trajectories. Not used but retained for compatibility.
        """

        if not check_mdp_equal(mdp, self.previous_mdp):
            cached_found = False
            # TODO seems like there are things in cache without having added to cache
            if self.caching:
                for m, q in self.previous_mdp_q_values.items():
                    if check_mdp_equal(mdp, m):
                        self.q_values = q
                        cached_found = True
            if not cached_found:
                self.VI.fit(mdp, self.reward_weights, None, None)
                self.q_values = self.VI.q_values
                if self.caching:
                    self.previous_mdp_q_values[mdp] = self.q_values.copy()

            self.previous_mdp = mdp

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

    def copy(self):

        # Copy without previous MDP to avoid picking error
        new_model = VIPolicyLearner(
            deepcopy(self.VI), copy(self.reward_weights), self.refit_on_new_mdp
        )
        new_model.q_values = self.q_values.copy()
        new_model.previous_mdp = None

        return new_model


class CombinedPolicyLearner(BaseGeneralPolicyLearner):
    def __init__(
        self,
        model1: BaseGeneralPolicyLearner,
        model2: BaseGeneralPolicyLearner,
        W: float = 0.5,
        scale: bool = True,
    ):
        """
        Produces a weighted combination of Q value estimates from two models.

        Args:
            model1 (BaseGeneralPolicyLearner): First model.
            model2 (BaseGeneralPolicyLearner): Second model.
            W (float, optional): Weighting parameter, lower values give Model 1 more weight. Defaults to 0.5.
            scale (bool, optional): If true, Q values from each model are minmax scaled to enable comparability between the two models.
            Defaults to True.
        """

        self.model1 = model1
        self.model2 = model2

        if not 0 <= W <= 1:
            raise ValueError("W must be between 0 and 1 (inclusive)")

        self.W = W
        self.scale = scale

    def reset(self):

        self.model1.reset()
        self.model2.reset()

    def fit(self, mdp: Union[MDP, List[MDP]], trajectories: list):
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
        self.model1.fit(mdp, trajectories)
        self.model2.fit(mdp, trajectories)

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

        overall_Q = (1 - self.W) * np.nan_to_num(
            model1_Q_scaled
        ) + self.W * np.nan_to_num(model2_Q_scaled)

        return overall_Q

    def copy(self):

        model1_copy = self.model1.copy()
        model2_copy = self.model2.copy()

        new_model = CombinedPolicyLearner(model1_copy, model2_copy, self.W, self.scale)
        return new_model


def nan_softmax(x: np.ndarray, return_nans: bool = False) -> np.ndarray:
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


def prediction_likelihood(q: np.ndarray, pred_actions: List[int]) -> float:
    """
    Calculates categorical likelihood.

    Args:
        q (np.ndarray): Array of Q values for each action at each observation, shape (observations, actions).
        pred_actions (List[int]): List of observed actions, one per observation.

    Returns:
        float: Log likelihood of the observed actions given the provided Q values.
    """

    assert (
        len(pred_actions) == q.shape[0]
    ), "Different numbers of predicted actions ({0}) and Q values ({1})".format(
        len(pred_actions), q.shape[0]
    )

    # Convert predicted actions to int
    pred_actions = np.array(pred_actions).astype(int).tolist()

    # Scale Q values so they're all on the same scale regardless of the model
    q = minmax_scale(
        q, max_val=5
    )  # Using 5 (arbitrarily) has same effect as reducing decision noise

    action_p = nan_softmax(q, return_nans=True)

    logp = np.nansum(np.log((action_p[range(len(pred_actions)), pred_actions]) + 1e-8))
    if np.isinf(logp):
        raise ValueError("Inf in logp")

    return logp


def fit_policy_learning(X: Tuple[float], *args: List) -> float:
    """
    Fits a policy learning model. Intended for use with scipy optimization functions.

    Args:
        X (Tuple): Learning rate, learning rate decay.
        args (List): Other arguments. 1: Predator trajectories, 2: MDPs,
        3: Subject's predicted actions, 4: Whether to use generalisation kernel, 5: Whether to reset the model
        for each environment

    Returns:
        float: Log likelihood
    """

    alpha, decay = X

    if np.isnan(alpha):
        alpha = 0.001

    if np.isnan(decay):
        decay = 0.001

    predator_t, target_mdp, predicted_a, kernel, env_reset = args

    _, Q_estimates, _ = action_prediction_envs(
        predator_t,
        target_mdp,
        TDGeneralPolicyLearner(learning_rate=alpha, kernel=kernel, decay=decay),
        action_selector=MaxActionSelector(seed=123),
        env_reset=env_reset,
    )

    logp = prediction_likelihood(np.vstack(Q_estimates), np.hstack(predicted_a))

    return -logp


def fit_combined_model(W: float, *args: List):
    """
    Fits a combined policy learning/value iteration model without estimating a learning rate for the policy learner.
    Intended for use with scipy optimization functions.

    Args:
        W (float): Weighting parameter, higher = higher weighting of policy learning.
        args (List): Other arguments. 1: Predator trajectory, 2: MDP,
        3: Subject's predicted actions, 4: Value iteration model, 5: Learning rate, 6: Whether
        to reset the model for each environment

    Returns:
        float: Log likelihood
    """

    # Avoid NaNs causing problems
    if np.isnan(W):
        W = 0.001

    predator_t, target_mdp, predicted_a, model1, learning_rate, decay, env_reset = args

    model2 = TDGeneralPolicyLearner(learning_rate=learning_rate, decay=decay)

    _, Q_estimates, _ = action_prediction_envs(
        predator_t,
        target_mdp,
        CombinedPolicyLearner(model1, model2, W=W),
        action_selector=MaxActionSelector(seed=123),
        env_reset=env_reset,
    )

    logp = prediction_likelihood(np.vstack(Q_estimates), np.hstack(predicted_a))

    return -logp


def fit_combined_model_learning_rate(X: Tuple[float], *args: List):
    """
    Fits a combined policy learning/value iteration model, estimating a learning rate for the policy learner.
    Intended for use with scipy optimization functions.

    Args:
        X (Tuple): Weighting parameter, learning rate, learning rate decay.
        args (List): Other arguments. 1: Predator trajectory, 2: MDP,
        3: Subject's predicted actions, 4: Value iteration model, 5: Whether to use a generalisation kernel, 6: Whether
        to reset the model for each environment

    Returns:
        float: Log likelihood
    """

    W, alpha, decay = X

    if np.isnan(W):
        W = 0.001
    if np.isnan(alpha):
        W = 0.001
    if np.isnan(decay):
        decay = 0.001

    predator_t, target_mdp, predicted_a, model1, kernel, env_reset = args

    model2 = TDGeneralPolicyLearner(learning_rate=alpha, kernel=kernel, decay=decay)

    _, Q_estimates, _ = action_prediction_envs(
        predator_t,
        target_mdp,
        CombinedPolicyLearner(model1, model2, W=W),
        action_selector=MaxActionSelector(seed=123),
        env_reset=env_reset,
    )

    logp = prediction_likelihood(np.vstack(Q_estimates), np.hstack(predicted_a))

    return -logp


"""
Learning models - not reset each environment (but could be), not reset for different MDPs
Goal inference model - reset for each environment, reset for different MDPs

Goal inference / learning models - goal inference reset for each environment, learning not (but could)
                                   goal inference reset for different MDPs, learning not


"""


def nan_to_zero(x: np.ndarray):
    """Removes nans and infs from an array"""
    x[np.isnan(x)] = 0
    x[np.isinf(x)] = 0
    return x


def action_prediction_envs(
    trajectories: List[List[int]],
    mdps: List[MDP],
    policy_model: BaseGeneralPolicyLearner,
    n_predictions: int = 2,
    action_selector: ActionSelector = None,
    step_reset: bool = False,
    env_reset: bool = False,
) -> Union[List, List, List]:
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

        Q, Q_estimates, predictions = action_prediction(
            trajectory,
            mdps[n],
            policy_model,
            n_predictions,
            action_selector,
            reset=step_reset,
        )

        all_Q.append(Q)
        all_Q_estimates.append(Q_estimates)
        all_predictions.append(predictions)

    return all_Q, all_Q_estimates, all_predictions


def action_prediction(
    trajectory: List[int],
    mdp: MDP,
    policy_model: BaseGeneralPolicyLearner,
    n_predictions: int = 2,
    action_selector: ActionSelector = None,
    reset: bool = False,
) -> Union[np.ndarray, np.ndarray, np.ndarray]:
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

    if action_selector is None:
        action_selector = MaxActionSelector(seed=123)

    if isinstance(mdp, list):
        if not len(mdp) == len(trajectory) - 1:
            raise AttributeError(
                "Must provide same number of MDPs as steps in the trajectory"
            )
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

    # Fit the model to get starting Q values before any observations
    # Using an empty trajectory means learning models produce Q values of zero for all actions if they've not been fit already
    policy_model.fit(mdp[0], [[]])

    # Loop through agent moves
    for n in range(len(action)):

        # First prediction of each turn
        if n % n_predictions == 0:
            # Observed starting state
            start_state = state[n]

            # Preserve model state from first prediction to allow real model to update on every move without affecting predictions
            try:
                temp_policy_learner = policy_model.copy()
            except Exception as e:
                print(policy_model.model1.previous_mdp, policy_model.model1.q_values)
                raise e

        else:
            # Otherwise the next prediction follows from the previous predicted one
            start_state = predicted_state

        # Get Q values for this start state
        trial_Q = temp_policy_learner.get_q_values(start_state)

        # Q values for making predictions
        prediction_Q = trial_Q.copy()

        # Set impossible actions to -inf
        prediction_Q[
            [
                i
                for i in range(mdp[0].n_actions)
                if not i in mdp[0].state_valid_actions(start_state)
            ]
        ] = -np.inf

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
            policy_model.fit(mdp[n], [state[n : n + 2]])

    Q_estimates = np.stack(Q_estimates)

    return Q, Q_estimates, predictions


# TODO add tests for these functions. Please. It'll save you pain in the long run.
