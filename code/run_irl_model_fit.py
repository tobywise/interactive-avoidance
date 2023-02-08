import pandas as pd
from irl_model_fit import fit_subject_ratings
from maMDP.env_io import *
from maMDP.env_io import hex_environment_from_dict
import os
from itertools import product
import argparse

if __name__ == "__main__":

    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env", help="Environment to run up to. By default, runs on all", default=0, type=int
    )

    args = parser.parse_args()

    # Get slurm run ID
    try:
        runID = int(os.environ["SLURM_ARRAY_TASK_ID"])
    except:
        runID = 47

    # Load in data
    rating_dfs = []
    rt_dfs = []
    confidence_dfs = []
    prediction_dfs = []
    response_dfs = []

    output_dir = "data/"
    experiment = "experiment-1"

    data_dfs = {
        "rating_data": rating_dfs,
        "rt_data": rt_dfs,
        "confidence_data": confidence_dfs,
        "prediction_data": prediction_dfs,
        "response_data": response_dfs,
    }

    for data_type, df_list in data_dfs.items():

        data_type_dir = os.path.join(output_dir, data_type, experiment)

        conditions = os.listdir(data_type_dir)

        for c in conditions:
            for i in os.listdir(os.path.join(data_type_dir, c)):
                if ".csv" in i:
                    df = pd.read_csv(os.path.join(data_type_dir, c, i))
                    df_list.append(df)

    rating_df = pd.concat(rating_dfs)
    rt_df = pd.concat(rt_dfs)
    confidence_df = pd.concat(confidence_dfs)
    prediction_df = pd.concat(prediction_dfs)
    response_df = pd.concat(response_dfs)

    # Select environments to run up to
    if int(args.env) > 0:
        rating_df = rating_df[rating_df["env"] <= int(args.env)]
        rt_df = rt_df[rt_df["env"] <= int(args.env)]
        confidence_df = confidence_df[confidence_df["env"] <= int(args.env)]
        prediction_df = prediction_df[prediction_df["env"] <= int(args.env)]
        response_df = response_df[response_df["env"] <= int(args.env)]

    prediction_df = prediction_df.sort_values(
        ["subjectID", "exp", "condition", "env", "trial", "response_number"]
    ).reset_index(drop=True)
    response_df = response_df.sort_values(
        ["agent", "subjectID", "exp", "condition", "env", "trial", "response_number"]
    ).reset_index(drop=True)

    # Get environment information
    envs = {}
    for cond in ["A", "B", "C"]:
        with open(
            "data/game_info/experiment-1/condition_{0}.json".format(cond), "r"
        ) as f:
            game_info = json.load(f)
        envs[cond] = [
            hex_environment_from_dict(env, ["Dirt", "Trees", "Reward"])
            for env in game_info["environments"]
        ]

    # Get data for IRL model fitting
    predator_df = response_df[response_df["agent"] == "predator"]
    prey_df = response_df[response_df["agent"] == "prey"]
    rating_df["feature_index"] = rating_df["feature"].replace(
        {"red": 1, "trees": 0, "prey": 2}
    )
    rating_df = rating_df[rating_df["env"] == 3]

    # Get subject IDs and parameter values
    run_params = pd.read_csv("data/IRL_runs.csv")

    # Candidate learning rates and decays
    learning_rates = np.linspace(0.1, 0.5, 5)
    learning_rate_decays = np.array([0.0001, 0.001, 0.01, 0.1, 0])
    learning_rate_params = product(learning_rates, learning_rate_decays)

    run_params = run_params.iloc[runID - 1].to_dict()
    sub = run_params["subjectID"]

    print('Fitting subject {0}'.format(sub))

    # Where to save output
    irl_fit_output_dir = "data/irl_fits/{0}".format(experiment)

    # Add in environment number if specified
    if int(args.env) > 0:
        irl_fit_output_dir = os.path.join(irl_fit_output_dir, "env_{0}".format(args.env))

    if not os.path.exists(irl_fit_output_dir):
        os.makedirs(irl_fit_output_dir)

    # Get ratings for this subject
    sub_ratings = (
        rating_df[
            (rating_df["subjectID"] == sub)
            & (rating_df["env"] == rating_df["env"].max())
        ]
        .sort_values(by="feature_index")["rating"]
        .values
    )
    if not len(sub_ratings) == 3:
        raise AttributeError(
            "Expected rating for 3 features, got {0}".format(len(sub_ratings))
        )

    n_jobs = len(os.sched_getaffinity(0))

    # Fit models
    if n_jobs > 1:
        from joblib import Parallel, delayed

        fits = Parallel(n_jobs=n_jobs)(
            delayed(fit_subject_ratings)(
                modelID,
                predator_df[predator_df["subjectID"] == sub],
                prey_df[prey_df["subjectID"] == sub],
                sub_ratings,
                envs,
                max_ent_learning_rate=lr,
                max_ent_learning_rate_decay=lr_decay,
            )
            for modelID, (lr, lr_decay) in enumerate(learning_rate_params)
        )

        # Save output
        for modelID, fit in enumerate(fits):
            fit.to_csv(
                os.path.join(
                    irl_fit_output_dir,
                    "job-{2}_subject-{0}_fit-{1}_IRL_fit.csv".format(
                        sub, modelID, runID
                    ),
                ),
                index=None,
            )

    else:
        for modelID, (lr, lr_decay) in enumerate(learning_rate_params):
            fit = fit_subject_ratings(
                modelID,
                predator_df[predator_df["subjectID"] == sub],
                prey_df[prey_df["subjectID"] == sub],
                sub_ratings,
                envs,
                max_ent_learning_rate=lr,
                max_ent_learning_rate_decay=lr_decay,
            )

            # Save output
            fit.to_csv(
                os.path.join(
                    irl_fit_output_dir,
                    "job-{2}_subject-{0}_fit-{1}_IRL_fit.csv".format(
                        sub, modelID, runID
                    ),
                ),
                index=None,
            )

    print("MODEL FIT FINISHED")
