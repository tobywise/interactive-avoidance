"""
Runs model fitting for subjects' predictions about the predator's actions.
"""

from maMDP.mdp import HexGridMDP
from maMDP.algorithms.policy_learning import *
from numpy.lib.function_base import diff
import pandas as pd
import numpy as np
from typing import Dict
from action_prediction import *
from tqdm import tqdm
from scipy.optimize import differential_evolution

def BIC(n_params: int, log_lik: float, n_obs: int):

    return n_params * np.log(n_obs) - 2 * log_lik


def fit_output_to_dataframe(
    accuracy_dict: Dict,
    log_lik_dict: Dict,
    bic_dict: Dict,
    alpha_values_dict: Dict,
    decay_values_dict: Dict,
    w_values_dict: Dict,
    subjectID: str,
    condition: str,
):

    out = {
        "model": [],
        "accuracy": [],
        "log_lik": [],
        "BIC": [],
        "alpha_values": [],
        "decay_values": [],
        "w_values": [],
    }

    for model in accuracy_dict.keys():
        out["model"].append(model)
        out["accuracy"].append(accuracy_dict[model])
        out["log_lik"].append(log_lik_dict[model])
        out["BIC"].append(bic_dict[model])
        out["alpha_values"].append(alpha_values_dict[model])
        out["decay_values"].append(decay_values_dict[model])
        out["w_values"].append(w_values_dict[model])

    out["subjectID"] = subjectID
    out["condition"] = condition

    out = pd.DataFrame(out)

    return out


def prediction_accuracy(
    observed_predictions: np.ndarray, expected_predictions: np.ndarray
):
    """Computes the accuracy of predictions"""

    assert len(observed_predictions) == len(
        expected_predictions
    ), "Observed and expected predictions are not the same length"

    return np.equal(observed_predictions, expected_predictions).sum() / len(
        observed_predictions
    )


def get_model_fit(
    observed_predictions: List[np.ndarray],
    observed_action_predictions: List[np.ndarray],
    expected_predictions: List[np.ndarray],
    Q_estimates: List[np.ndarray],
    n_params: int = 0,
) -> Union[float, float, float]:
    """
    Calculates three model fit metrics: Prediction accuracy, log likelihood, and BIC

    Args:
        observed_predictions (List[np.ndarray]): Subject's predictions
        observed_action_predictions (List[np.ndarray]): Subjects' predicted actions
        expected_predictions (List[np.ndarray]): Model predictions
        Q_estimates (List[np.ndarray]): Estimated Q values for each action
        n_params (int, optional): Number of model parameters, used for BIC calculation. Defaults to 0.
    Returns:
        Union[float, float, float]: Returns accuracy, log likelihood, and BIC
    """

    # Stack arrays
    observed_predictions = np.hstack(observed_predictions)
    expected_predictions = np.hstack(expected_predictions)
    observed_action_predictions = np.hstack(observed_action_predictions)
    Q_estimates = np.vstack(Q_estimates)

    # Calculate
    accuracy = prediction_accuracy(observed_predictions, expected_predictions)
    log_lik = prediction_likelihood(Q_estimates, observed_action_predictions)
    bic = BIC(n_params, log_lik, len(observed_predictions))

    return accuracy, log_lik, bic


def fill_missing_predictions(pred, n_expected=20):

    new_pred = []

    for i in pred:
        if not len(i) == n_expected:
            new_pred.append(i + [-999] * (n_expected - len(i)))
        else:
            new_pred.append(i)

    return new_pred


def fit_models(
    trajectories: List[List[int]],
    mdps: List[MDP],
    predicted_states: List[List[int]],
    predicted_actions: List[List[int]],
    agent_reward_function: np.ndarray,
) -> Union[Dict, Dict, Dict, Dict, Dict, Dict]:
    """
    Fits 7 models to predictions:
    1. Policy repetition
    2. Policy learning
    3. Policy learning with generalisation
    4. Goal inference
    5. Goal inference and policy repetition
    6. Goal inference and policy learning
    7. Goal inference and policy learning with generalisation

    Args:
        trajectories (List[List[int]]): Observed agent trajectories for each environment
        mdps (List[MDP]): MDPs, one (or one list) for each environment
        predicted_states (List[List[int]]): Observed predictions for agent's moves (states)
        predicted_actions (List[List[int]]): Observed predictions for agent's moves (actions)
        agent_reward_function (np.ndarray): Agent's reward function. Assumes the agent has the same reward function across all MDPs.

    Returns:
        Union[Dict, Dict, Dict, Dict, Dict, Dict]: Returns model accuracy, log likelihood, and BIC, along with its predictions and
        fitted parameter values for learning rate and weighting parameters
    """

    # Dictionary to store outputs
    accuracy = {}
    log_lik = {}
    bic = {}
    alpha_values = {}
    decay_values = {}
    w_values = {}
    model_predictions = {}

    #####################
    # POLICY REPETITION #
    #####################
    _, Q_estimates, predictions = action_prediction_envs(
        trajectories,
        mdps,
        TDGeneralPolicyLearner(learning_rate=1, decay=0),
        action_selector=MaxActionSelector(seed=123),
    )

    accuracy["repetition"], log_lik["repetition"], bic["repetition"] = get_model_fit(
        predicted_states, predicted_actions, predictions, Q_estimates
    )
    alpha_values["repetition"], decay_values["repetition"], w_values["repetition"] = (
        np.nan,
        np.nan,
        np.nan,
    )  # No parameters in this model
    model_predictions["repetition"] = np.array(
        fill_missing_predictions(predictions)
    ).copy()

    ###################
    # POLICY LEARNING #
    ###################

    # Estimate learning rate
    res = differential_evolution(
        fit_policy_learning,
        seed=123,
        args=(trajectories, mdps, predicted_actions, False, False),
        bounds=[(0.001, 0.999), (0.001, 0.999)],
    )

    # Simulate with estimated learning rate
    _, Q_estimates, predictions = action_prediction_envs(
        trajectories,
        mdps,
        TDGeneralPolicyLearner(learning_rate=res.x[0], decay=res.x[1]),
        action_selector=MaxActionSelector(seed=123),
    )

    (
        accuracy["policy_learning"],
        log_lik["policy_learning"],
        bic["policy_learning"],
    ) = get_model_fit(predicted_states, predicted_actions, predictions, Q_estimates, 1)

    alpha_values["policy_learning"], decay_values['policy_learning'], w_values["policy_learning"] = (res.x[0], res.x[1], np.nan)
    model_predictions["policy_learning"] = np.array(
        fill_missing_predictions(predictions)
    ).copy()

    #########################
    # POLICY GENERALISATION #
    #########################

    # Estimate learning rate
    res = differential_evolution(
        fit_policy_learning,
        seed=123,
        args=[trajectories, mdps, predicted_actions, True, False],
        bounds=[(0.001, 0.999), (0.001, 0.999)],
    )

    _, Q_estimates, predictions = action_prediction_envs(
        trajectories,
        mdps,
        TDGeneralPolicyLearner(learning_rate=res.x[0], decay=res.x[1], kernel=True),
        action_selector=MaxActionSelector(seed=123),
    )

    (
        accuracy["policy_generalisation"],
        log_lik["policy_generalisation"],
        bic["policy_generalisation"],
    ) = get_model_fit(predicted_states, predicted_actions, predictions, Q_estimates, 1)

    alpha_values["policy_generalisation"], decay_values['policy_generalisation'], w_values["policy_generalisation"] = (
        res.x[0],
        res.x[1],
        np.nan,
    )
    model_predictions["policy_generalisation"] = np.array(
        fill_missing_predictions(predictions)
    ).copy()

    ##################
    # GOAL INFERENCE #
    ##################
    _, Q_estimates, predictions = action_prediction_envs(
        trajectories,
        mdps,
        VIPolicyLearner(ValueIteration(), agent_reward_function),
        action_selector=MaxActionSelector(seed=123),
    )
    (
        accuracy["goal_inference"],
        log_lik["goal_inference"],
        bic["goal_inference"],
    ) = get_model_fit(predicted_states, predicted_actions, predictions, Q_estimates)
    alpha_values["goal_inference"], decay_values['goal_inference'], w_values["goal_inference"] = (np.nan, np.nan, np.nan)
    model_predictions["goal_inference"] = np.array(
        fill_missing_predictions(predictions)
    ).copy()

    ######################################
    # GOAL INFERENCE + POLICY REPETITION #
    ######################################

    # Fit VI here so it doesn't get refit constantly during parameter estimation
    VI_model = VIPolicyLearner(ValueIteration(), agent_reward_function, caching=True)
    if not isinstance(mdps[0], list):
        VI_model.fit(mdps[0], [])
    else:
        VI_model.fit(mdps[0][0], [])

    # Estimate weighting parameter
    res = differential_evolution(
        fit_combined_model,
        seed=123,
        args=[trajectories, mdps, predicted_actions, VI_model, 1, 0, False],
        bounds=[(0, 1)]
    )

    _, Q_estimates, predictions = action_prediction_envs(
        trajectories,
        mdps,
        CombinedPolicyLearner(
            VIPolicyLearner(ValueIteration(), agent_reward_function),
            TDGeneralPolicyLearner(learning_rate=1, decay=0),
            W=res.x[0],
            scale=True,
        ),
        action_selector=MaxActionSelector(seed=123),
    )
    (
        accuracy["combined_repetition"],
        log_lik["combined_repetition"],
        bic["combined_repetition"],
    ) = get_model_fit(predicted_states, predicted_actions, predictions, Q_estimates, 1)
    alpha_values["combined_repetition"], decay_values['combined_repetition'], w_values["combined_repetition"] = (
        np.nan,
        np.nan,
        res.x[0],
    )
    model_predictions["combined_repetition"] = np.array(
        fill_missing_predictions(predictions)
    ).copy()

    #############################
    # GOAL INFERENCE + LEARNING #
    #############################

    res = differential_evolution(
        fit_combined_model_learning_rate,
        seed=123,
        args=[trajectories, mdps, predicted_actions, VI_model, False, False],
        bounds=[(0, 1), (0.001, 0.999), (0.001, 0.999)]
    )

    _, Q_estimates, predictions = action_prediction_envs(
        trajectories,
        mdps,
        CombinedPolicyLearner(
            VIPolicyLearner(ValueIteration(), agent_reward_function),
            TDGeneralPolicyLearner(learning_rate=res.x[1], decay=res.x[2]),
            W=res.x[0],
            scale=True,
        ),
        action_selector=MaxActionSelector(seed=123),
    )
    (
        accuracy["combined_learning"],
        log_lik["combined_learning"],
        bic["combined_learning"],
    ) = get_model_fit(predicted_states, predicted_actions, predictions, Q_estimates, 2)
    alpha_values["combined_learning"], decay_values['combined_learning'], w_values["combined_learning"] = (
        res.x[1],
        res.x[2],
        res.x[0],
    )
    model_predictions["combined_learning"] = np.array(
        fill_missing_predictions(predictions)
    ).copy()

    ###################################
    # GOAL INFERENCE + GENERALISATION #
    ###################################

    res = differential_evolution(
        fit_combined_model_learning_rate,
        seed=123,
        args=[trajectories, mdps, predicted_actions, VI_model, True, False],
        bounds=[(0, 1), (0.001, 0.999), (0.001, 0.999)]
    )

    _, Q_estimates, predictions = action_prediction_envs(
        trajectories,
        mdps,
        CombinedPolicyLearner(
            VIPolicyLearner(ValueIteration(), agent_reward_function),
            TDGeneralPolicyLearner(learning_rate=res.x[1], decay=res.x[2], kernel=True),
            W=res.x[0],
            scale=True,
        ),
        action_selector=MaxActionSelector(seed=123),
    )
    (
        accuracy["combined_generalisation"],
        log_lik["combined_generalisation"],
        bic["combined_generalisation"],
    ) = get_model_fit(predicted_states, predicted_actions, predictions, Q_estimates, 2)
    alpha_values["combined_generalisation"], decay_values['combined_generalisation'], w_values["combined_generalisation"] = (
        res.x[1],
        res.x[2],
        res.x[0],
    )
    model_predictions["combined_generalisation"] = np.array(
        fill_missing_predictions(predictions)
    ).copy()

    return accuracy, log_lik, bic, model_predictions, alpha_values, decay_values, w_values


def fit_subject_predictions(
    predator_moves: pd.DataFrame,
    prey_moves: pd.DataFrame,
    predictions: pd.DataFrame,
    env_info: Dict,
) -> pd.DataFrame:
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
    predator_moves = predator_moves.sort_values(
        ["subjectID", "exp", "condition", "env", "trial", "response_number"]
    ).reset_index(drop=True)
    prey_moves = prey_moves.sort_values(
        ["subjectID", "exp", "condition", "env", "trial", "response_number"]
    ).reset_index(drop=True)
    predictions = predictions.sort_values(
        ["subjectID", "exp", "condition", "env", "trial", "response_number"]
    ).reset_index(drop=True)

    # Check if any data is missing
    try:
        if (
            not (predator_moves["trial"].diff() > 1).any()
            or not len(predator_moves["condition"].unique()) == 1
        ):

            # Dictionaries to store outputs
            accuracy = {}
            log_lik = {}
            bic = {}
            alpha_values = {}
            w_values = {}
            model_predictions = {}

            # Get condition that this subject is in
            condition = predator_moves["condition"].tolist()[0]

            # Get subject ID
            subject = predator_moves["subjectID"].tolist()[0]

            # Information for model fitting
            env_predator_trajectories = []
            env_mdps = []
            env_predictions = []
            env_action_predictions = []
            valid = True

            # Loop through environments
            for env in predator_moves["env"].unique():

                # Get data for this environment
                env_predator_df = predator_moves[predator_moves["env"] == env]
                env_prediction_df = predictions[predictions["env"] == env]
                env_prey_df = prey_moves[prey_moves["env"] == env]

                # Remove missing data due to getting caught
                env_predator_df = env_predator_df[env_predator_df["cellID"] != -999]
                env_prediction_df = env_prediction_df[
                    env_prediction_df["cellID"] != -999
                ]
                env_prey_df = env_prey_df[env_prey_df["cellID"] != -999]

                # Get trajectories
                predator_trajectory = [
                    env_info[condition][env].agents["Predator_1"].position
                ] + env_predator_df["cellID"].tolist()
                predicted_trajectory = env_prediction_df["cellID"].tolist()

                # Get MDP representing this environment
                mdp = env_info[condition][env].mdp

                # Ensure that we have the right number of predictions
                if len(predator_trajectory) - 1 == len(predicted_trajectory):

                    # In condition 3, the predator values the prey so we need to update the MDP with the prey's position
                    # at every step. Otherwise, the prey's position makes no difference in terms of solving the MDP
                    if condition == 3:

                        # Get prey moves
                        prey_moves_array = env_prey_df["cellID"].values[1:]

                        mdps = []

                        for i in range(len(prey_moves_array)):
                            trial_env = env_info[condition][env]
                            trial_env.set_agent_position("Prey_1", prey_moves_array[i])
                            new_mdp = HexGridMDP(
                                trial_env.mdp.features.copy(),
                                trial_env.mdp.walls,
                                shape=(21, 10),
                            )
                            mdps += [new_mdp, new_mdp]  # same mdp for 2 moves
                    else:
                        mdps = mdp

                    # GET ACTIONS FROM PREDICTED STATES
                    predicted_actions = []

                    for i in np.arange(0, len(predicted_trajectory), 2):
                        t = [predator_trajectory[i], *predicted_trajectory[i : i + 2]]
                        predicted_actions += (
                            mdp._trajectory_to_state_action(t)[:, 1]
                            .astype(int)
                            .tolist()
                        )

                    # Add info for this environment to list
                    env_predator_trajectories.append(predator_trajectory)
                    env_mdps.append(mdps)
                    env_predictions.append(predicted_trajectory)
                    env_action_predictions.append(predicted_actions)

                # Otherwise skip this subject
                else:
                    print(len(predator_trajectory), len(predicted_trajectory))
                    raise ValueError(
                        "Subject {0}, env {1}, cond {2} has mismatch between predicted and observed moves".format(
                            subject, env, condition
                        )
                    )

            if valid:
                (
                    accuracy,
                    log_lik,
                    bic,
                    model_predictions,
                    decay_values,
                    alpha_values,
                    w_values,
                ) = fit_models(
                    env_predator_trajectories,
                    env_mdps,
                    env_predictions,
                    env_action_predictions,
                    env_info[condition][env].agents["Predator_1"].reward_function,
                )

                out_df = fit_output_to_dataframe(
                    accuracy, log_lik, bic, alpha_values, decay_values, w_values, subject, condition
                )

                return out_df, model_predictions

        else:
            print("Missing data, skipping")
    except:
        raise ValueError("PROBLEM WITH DATA")


def fit_prediction_models(
    predator_moves: pd.DataFrame,
    prey_moves: pd.DataFrame,
    predictions: pd.DataFrame,
    env_info: Dict,
    n_jobs=1,
) -> Union[pd.DataFrame, List]:
    """
    Fits 7 different prediction models to subjects' data:

    1. Policy repetition
    2. Policy learning
    3. Policy learning with generalisation
    4. Goal inference
    5. Goal inference and policy repetition
    6. Goal inference and policy learning
    7. Goal inference and policy learning with generalisation

    Args:
        predator_moves (pd.DataFrame): Dataframe containing moves made by the predator
        prey_moves (pd.DataFrame): Dataframes containing moves made my prey
        predictions (pd.DataFrame): Dataframe containing predictions made by subject
        env_info (Dict): Environment information
        n_jobs (int, optional): Number of jobs for parallel processing. Defaults to 1.

    Returns:
        Union[pd.Dataframe, List]: Returns fit statistics and predictions made by the models
    """

    subjects = list(predator_moves["subjectID"].unique())

    fit_dfs = []
    model_prediction_list = []

    if n_jobs == 1:

        for sub in tqdm(subjects):
            fit, model_predictions = fit_subject_predictions(
                predator_moves[predator_moves["subjectID"] == sub],
                prey_moves[prey_moves["subjectID"] == sub],
                predictions[predictions["subjectID"] == sub],
                env_info,
            )

            fit_dfs.append(fit)
            model_prediction_list.append(model_predictions)

    else:

        from joblib import Parallel, delayed

        class ProgressParallel(Parallel):
            """https://stackoverflow.com/a/61027781"""

            def __call__(self, *args, **kwargs):
                with tqdm() as self._pbar:
                    return Parallel.__call__(self, *args, **kwargs)

            def print_progress(self):
                self._pbar.total = self.n_dispatched_tasks
                self._pbar.n = self.n_completed_tasks
                self._pbar.refresh()

        print("Fitting in parallel, {0} jobs".format(n_jobs))
        fits = ProgressParallel(n_jobs=n_jobs)(
            delayed(fit_subject_predictions)(
                predator_moves[predator_moves["subjectID"] == sub],
                prey_moves[prey_moves["subjectID"] == sub],
                predictions[predictions["subjectID"] == sub],
                env_info,
            )
            for sub in subjects
        )

        fit_dfs = [i[0] for i in fits]

    all_subject_fit = pd.concat(fit_dfs)

    return all_subject_fit, model_prediction_list
