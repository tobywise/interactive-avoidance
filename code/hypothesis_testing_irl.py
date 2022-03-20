import jax
import jax.numpy as jnp
from jax import jit
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from typing import List, Union


def get_sa_sr(sas: np.ndarray, gamma=0.95) -> np.ndarray:
    """
    Derives the successor representation for state-action pairs.

    Args:
        sas (np.ndarray): State - action - state transitions.
        gamma (float, optional): Discount factor. Defaults to 0.95.

    Returns:
        np.ndarray: Successor representation of shape [n states * n actions, n states * n actions]
    """

    n_states = sas.shape[0]
    n_actions = sas.shape[1]

    sathing = np.zeros((n_states * n_actions, n_states * n_actions))

    for s in range(n_states):
        for a in range(n_actions):
            next_state = np.argwhere(sas[s, a, :])
            if len(next_state):
                next_state = next_state[0][0]
                for a_ in range(n_actions):
                    sathing[(s * n_actions) + a, (next_state * n_actions) + a_] = (
                        1 / n_actions
                    )

    sr = np.linalg.inv(np.eye(sathing.shape[0]) - gamma * sathing)

    return sr


def get_state_q_values(
    state: int,
    theta: np.ndarray,
    features: np.ndarray,
    sr: np.ndarray,
    n_states: int,
    n_actions: int,
) -> jnp.ndarray:
    """
    Gets Q values for a single state, given the successor representation and reward
    weights.

    Args:
        state (int): State for which to return Q values
        theta (np.ndarray): Reward weights, one entry per feature
        features (np.ndarray): Features
        sr (np.ndarray): Successor representation. Each entry must represent a state-action
        pair rather than a state.
        n_states (int): Number of states
        n_actions (int): Number of actions

    Returns:
        np.ndarray: Returns Q values for every action from the given state.
    """

    assert (
        features.shape[-1] == n_states
    ), "Mismatch between features shape and number of states"

    reward = jnp.dot(theta, features)
    qs = jnp.dot(
        sr, reward.repeat(n_actions)
    )  # Repeat so we have the same reward for each action in a state
    qs = qs.reshape((n_states, n_actions))

    return qs[state, :]


# Vmap this function across observations (i.e. moves)
get_state_q_values_vmap = jax.vmap(
    get_state_q_values, in_axes=(0, None, 0, None, None, None)
)


@jit
def softmax_function(qs: np.ndarray, temperature=1):
    """Softmax function"""
    out = (jnp.exp((qs - qs.max(axis=0)) / temperature)) / (
        jnp.sum(jnp.exp((qs - qs.max(axis=0)) / temperature), axis=0)
    )
    return out


def hyp_test_irl_model(
    observed_states: np.array,
    features: np.array,
    observed_actions: np.array,
    sr: np.ndarray,
    n_states: int,
    n_actions: int,
):
    """
    Hypothesis testing IRL model.

    Args:
        states (np.array): Sequence of observed states
        features (np.array): Observed features, of shape [observations, features, states]
        observed_actions (np.array): Sequence of observed actions
        sr (np.ndarray): Successor representation, with each entry representing a state-action pair
        n_states (int): Number of states
        n_actions (int): Number of actions
    """

    # Prior on reward weights
    theta_raw = numpyro.sample("theta_raw", dist.Beta(jnp.ones(3), jnp.ones(3)))
    theta = numpyro.deterministic("theta", (theta_raw * 2) - 1)

    # Get Q values
    qs = get_state_q_values_vmap(
        observed_states, theta, features, sr, n_states, n_actions
    )

    # Convert to choice probabilities
    ps = softmax_function(qs.T, 0.083)

    # Observed
    numpyro.sample("obs", dist.Categorical(ps.T), obs=observed_actions)


def get_features_states_actions(
    one_step_mdps: List,
    one_step_predator_actions: List,
    one_step_predator_trajectories: List,
) -> Union[np.ndarray, np.ndarray, np.ndarray]:

    feature_arrays = []

    for i in range(len(one_step_mdps)):
        feature_arrays += [one_step_mdps[i].features] * len(
            one_step_predator_actions[i]
        )

    feature_arrays = np.stack(feature_arrays)

    states = []
    for i in one_step_predator_trajectories:
        states.append(i[: len(i) - 1])

    states = np.hstack(states)

    # states = np.array(
    #     [i[: len(i) - 1] for i in one_step_predator_trajectories]
    # ).flatten()
    actions = np.hstack(one_step_predator_actions)

    return feature_arrays, actions, states


def fit_hyp_test_irl_model(
    one_step_mdps: List,
    one_step_predator_actions: List,
    one_step_predator_trajectories: List,
    n_samples: int = 2000,
    gamma: float = 0.99,
) -> np.ndarray:
    """
    Fits the hypothesis testing IRL model using MCMC sampling.

    Args:
        states (np.array): Sequence of observed states
        features (np.array): Observed features, of shape [observations, features, states]
        observed_actions (np.array): Sequence of observed actions
        n_samples (int, optional): Number of samples. Defaults to 2000.
        gamma (float, optional): Discount. Defaults to .95.

    Returns:
        np.ndarray: Returns the mean of the posterior distribution over reward weights.
    """

    # Extract features, agent states, agent actions
    features, observed_actions, observed_states = get_features_states_actions(
        one_step_mdps, one_step_predator_actions, one_step_predator_trajectories
    )

    # Calculate successor representation
    sr = get_sa_sr(one_step_mdps[0].sas, gamma=gamma)

    n_states = one_step_mdps[0].n_states
    n_actions = one_step_mdps[0].n_actions

    nuts_kernel = NUTS(hyp_test_irl_model, forward_mode_differentiation=True)
    mcmc = MCMC(nuts_kernel, num_samples=n_samples, num_warmup=1000)

    rng_key = jax.random.PRNGKey(123)

    # Select features
    features = features[:, [0, 1, 4], :]

    mcmc.run(
        rng_key, observed_states, features, observed_actions, sr, n_states, n_actions
    )

    posterior_samples = mcmc.get_samples()

    # Get mean of posterior
    mean_theta = posterior_samples["theta"].mean(axis=0)
    theta_out = np.zeros(5)
    theta_out[:2] = mean_theta[:2]
    theta_out[4] = mean_theta[2]

    return theta_out
