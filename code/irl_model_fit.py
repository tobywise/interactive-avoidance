"""
Runs model fitting for subjects' ratings of the predator's preferences.
"""

from maMDP.mdp import HexGridMDP
from maMDP.algorithms.policy_learning import *
from maMDP.algorithms.irl import (
    MaxCausalEntIRL,
    HyptestIRL,
    MFFeatureMatching,
    SimpleActionIRL,
    SimpleActionFeatureDiffIRL,
)
from tqdm import tqdm
import pandas as pd
import numpy as np
from typing import Dict
from sklearn.preprocessing import maxabs_scale
from action_prediction import *
from hypothesis_testing_irl import fit_hyp_test_irl_model
from numba import njit
import os


@njit
def self_transitions(sas):
    new_sas = sas.copy()
    for s in range(new_sas.shape[0]):
        for a in range(new_sas.shape[1]):
            if np.max(new_sas[s, a, :]) == 0:
                new_sas[s, a, s] = 1

    return new_sas


def get_actions(sas, trajectory):
    actions = []
    for t in range(len(trajectory[:-1])):
        actions.append(np.argwhere(sas[trajectory[t], :, trajectory[t + 1]])[0][0])
    return actions


def fit_subject_ratings(
    model_id: int,
    predator_moves: pd.DataFrame,
    prey_moves: pd.DataFrame,
    ratings: np.ndarray,
    env_info: Dict,
    max_ent_learning_rate: float = 0.8,
    max_ent_learning_rate_decay: float = 0.001
) -> pd.DataFrame:

    # Make sure everything is in the right order
    predator_moves = predator_moves.sort_values(
        ["subjectID", "exp", "condition", "env", "trial", "response_number"]
    ).reset_index(drop=True)
    prey_moves = prey_moves.sort_values(
        ["subjectID", "exp", "condition", "env", "trial", "response_number"]
    ).reset_index(drop=True)

    # Check if any data is missing
    if (
        not (predator_moves["trial"].diff() > 1).any()
        or not len(predator_moves["condition"].unique()) == 1
    ):

        # Get condition that this subject is in
        condition = predator_moves["condition"].tolist()[0]

        # Get subject ID
        subject = predator_moves["subjectID"].tolist()[0]

        # Predator movement data and MDP info needs to be extracted in two different ways:
        # 1. One MDP and trajectory per environment, for MaxEnt which cannot use different MDPs (representing the changing prey feature)
        #    for each step.
        # 2. One MDP and trajectory per move, combining environments, for MF and HypTest models which can use a different MDP
        #    for each step.
        #

        # These list represent one-step data for MF and HypTest models - each entry corresponds to one predator turn
        # (each turn can include multiple predator moves, e.g. if the predator makes 3 moves per turn and has 2 turns, the
        # list of MDPs will have 2 entries while the list of moves will have 2 entries, each of which has 4 entries including the start state)
        one_step_mdps = []
        one_step_predator_trajectories = []
        one_step_predator_actions = []

        # These lists represent environment-level data for MaxEnt - each entry corresponds to one environment
        env_mdps = []
        env_predator_trajectories = []

        # Loop through environments
        for env in predator_moves["env"].unique():

            # Get data for this environment
            env_predator_df = predator_moves[predator_moves["env"] == env]
            env_prey_df = prey_moves[prey_moves["env"] == env]

            # Remove trials after the prey got caught
            env_prey_df = env_prey_df[env_prey_df["cellID"] != -999]
            env_predator_df = env_predator_df[env_predator_df["cellID"] != -999]

            # Get trajectories
            predator_trajectory = [
                env_info[condition][env].agents["Predator_1"].position
            ] + env_predator_df["cellID"].tolist()

            predator_trajectory = [i for i in predator_trajectory if not i == -999]
            # Get MDP representing this environment
            mdp = env_info[condition][env].mdp

            # Ensure that we have the right number of moves
            if len(predator_trajectory):

                # Add environment-level data
                env_mdps.append(mdp)
                env_predator_trajectories.append(env_prey_df["cellID"].values)

                # Add one-step data

                # Get prey moves
                prey_moves_array = env_prey_df["cellID"].values[1:]

                # Figure out how many moves the predator makes per turn
                n_predator_moves = env_predator_df["response_number"].max() + 1

                # Get MDPs - one per turn
                mdps = []

                for i in range(len(prey_moves_array)):
                    trial_env = env_info[condition][env]
                    trial_env.set_agent_position("Prey_1", prey_moves_array[i])
                    new_mdp = HexGridMDP(
                        trial_env.mdp.features.copy(),
                        trial_env.mdp.walls,
                        shape=(21, 10),
                    )
                    mdps.append(new_mdp)

                # Get predator moves
                predator_moves_array = env_predator_df["cellID"].values

                predator_moves_list = []
                predator_actions_list = []

                for i in range(len(prey_moves_array)):
                    predator_moves_list.append(
                        list(
                            predator_moves_array[
                                i * n_predator_moves : i * n_predator_moves
                                + n_predator_moves
                                + 1
                            ]
                        )
                    )
                    predator_actions_list.append(
                        get_actions(
                            mdp.sas,
                            list(
                                predator_moves_array[
                                    i * n_predator_moves : i * n_predator_moves
                                    + n_predator_moves
                                    + 1
                                ]
                            ),
                        )
                    )

                # Add to lists
                one_step_mdps += mdps
                one_step_predator_trajectories += predator_moves_list
                one_step_predator_actions += predator_actions_list

            else:
                print(
                    "Subject {0}, env {1}, cond {2} has mismatch between predicted and observed predator moves".format(
                        subject, env, condition
                    )
                )

        # Dictionary to store outputs
        theta = {}

        # Only run these models once as they have no hyperparameters
        if model_id == 0:

            # Model-free feature count matching
            mf_irl = MFFeatureMatching()
            theta['MF'] = mf_irl.fit(one_step_mdps, one_step_predator_trajectories, [2, 3])
            print('MF done')

            # Directionality
            action_irl = SimpleActionIRL()
            theta['MF_direction'] = action_irl.fit(one_step_mdps, one_step_predator_trajectories)
            print('MF direction done')

            # Directionality reiative to other directions
            relative_action_irl = SimpleActionFeatureDiffIRL()
            theta['MF_relative_direction'] = relative_action_irl.fit(one_step_mdps, one_step_predator_trajectories)
            print('MF relative direction done')

            # Hypothesis testing
            # This approach works for Condition 3 as it is based on single steps, rather than entire trajectories,
            # and does not depend upon feature counts across an entire trajectory.

            theta["HypTest"] = fit_hyp_test_irl_model(
                one_step_mdps, one_step_predator_actions, one_step_predator_trajectories
            )
            print("HypTest Done")


        # Run MaxEnt on every iteration

        # Make walls etc transition to the same state, necessary for soft VI
        for m in env_mdps:
            m.sas = self_transitions(m.sas)

        for m in one_step_mdps:
            m.sas = self_transitions(m.sas)

        # MaxEnt
        if condition != 'C':
            # MaxEnt doesn't work for Condition 3 as features are unstable due to the prey moving
            # This is because it works at the level of the trajectory - comparing the feature counts from an observed trajectory to
            # those expected if the agent were solving the MDP according to the currently inferred reward function.
            # In principle, it would be possible to use one-step "trajectories" to partially get around this problem - however
            # this doesn't give the algorithm much to work with in terms of feature counts, as only one state will be encountered each time. This also
            # would still assume that the agent is covering a stable MDP (in terms of its features) during the calculation of expected feature counts.
            maxent = MaxCausalEntIRL(max_iter_irl=1000, learning_rate=max_ent_learning_rate, decay=max_ent_learning_rate_decay, tol=1e-6)
            for n, m in enumerate(env_mdps[:]):
                maxent.fit(m, [env_predator_trajectories[n]], [2, 3], True)

            theta['MaxEnt'] = maxent.theta

        else: # Cannot do condition 3
            theta['MaxEnt'] = np.ones(mdps[0].n_features) * np.nan

        print('MaxEnt done')


        # Turn output into a dataframe
        out_df = {
            "model": [],
            "f1": [],  # feature 1
            "f2": [],  # feature 2
            "f3": [],  # feature 3
            # 'r2': [],  # fit
            "f1_true": [],  # subject's ratings for feature 1
            "f2_true": [],  # subject's ratings for feature 2
            "f3_true": [],  # subject's ratings for feature 3
        }

        for model in theta.keys():
            out_df["model"].append(model)
            out_df["f1"].append(theta[model][0])
            out_df["f2"].append(theta[model][1])
            out_df["f3"].append(theta[model][4])
            out_df["f1_true"].append(ratings[0])
            out_df["f2_true"].append(ratings[1])
            out_df["f3_true"].append(ratings[2])

        out_df = pd.DataFrame(out_df)

        out_df["max_ent_learning_rate"] = max_ent_learning_rate
        out_df["max_ent_learning_rate_decay"] = max_ent_learning_rate_decay

        out_df["subjectID"] = subject
        out_df["condition"] = condition

        return out_df

    else:
        print("Missing data, skipping")


def fit_IRL_models(
    predator_moves: pd.DataFrame,
    prey_moves: pd.DataFrame,
    ratings: pd.DataFrame,
    env_info: Dict,
):

    subjects = list(predator_moves["subjectID"].unique())

    fit_dfs = []

    for sub in tqdm(subjects):
        sub_ratings = (
            ratings[
                (ratings["subjectID"] == sub) & (ratings["env"] == ratings["env"].max())
            ]
            .sort_values(by="feature_index")["rating"]
            .values
        )
        if not len(sub_ratings) == 3:
            raise AttributeError(
                "Expected rating for 3 features, got {0}".format(len(sub_ratings))
            )
        fit = fit_subject_ratings(
            predator_moves[predator_moves["subjectID"] == sub],
            prey_moves[prey_moves["subjectID"] == sub],
            sub_ratings,
            env_info,
        )
        fit_dfs.append(fit)

    all_subject_fit = pd.concat(fit_dfs)

    return all_subject_fit


