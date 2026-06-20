"""Minimal PyTorch components for the Flow Matching research material."""

from .data import eight_gaussian_centers, sample_eight_gaussians, sample_standard_normal
from .diffusion_basics import ddim_sample, epsilon_prediction_loss, q_sample, trig_alpha_sigma
from .evaluation import endpoint_error, gaussian_kernel_mmd, nearest_mode_coverage, pairwise_squared_distances
from .image_models import TinyDiTVelocity, TinyUNetVelocity, patchify, unpatchify
from .couplings import greedy_minibatch_coupling, pairwise_squared_cost, random_coupling
from .losses import conditional_flow_matching_loss, rectified_flow_loss
from .models import MLPVelocity, RectifiedFlowMLP
from .paths import ConditionalPath, LinearPath, TrigGaussianPath
from .reflow import make_reflow_pairs, straightness_ratio, trajectory_path_length
from .solvers import euler_solve, heun_solve, nfe_per_step, rk4_solve, solver_nfe, steps_from_nfe_budget
from .time_samplers import (
    TIME_SAMPLERS,
    TimeSampler,
    get_time_sampler,
    sample_center_time,
    sample_endpoint_time,
    sample_uniform_time,
)

__all__ = [
    "LinearPath",
    "ConditionalPath",
    "TrigGaussianPath",
    "TIME_SAMPLERS",
    "TimeSampler",
    "TinyDiTVelocity",
    "TinyUNetVelocity",
    "MLPVelocity",
    "RectifiedFlowMLP",
    "conditional_flow_matching_loss",
    "ddim_sample",
    "rectified_flow_loss",
    "epsilon_prediction_loss",
    "endpoint_error",
    "eight_gaussian_centers",
    "euler_solve",
    "gaussian_kernel_mmd",
    "get_time_sampler",
    "greedy_minibatch_coupling",
    "heun_solve",
    "make_reflow_pairs",
    "nfe_per_step",
    "nearest_mode_coverage",
    "pairwise_squared_cost",
    "pairwise_squared_distances",
    "patchify",
    "rk4_solve",
    "q_sample",
    "random_coupling",
    "sample_eight_gaussians",
    "sample_center_time",
    "sample_endpoint_time",
    "sample_standard_normal",
    "sample_uniform_time",
    "solver_nfe",
    "steps_from_nfe_budget",
    "straightness_ratio",
    "trajectory_path_length",
    "trig_alpha_sigma",
    "unpatchify",
]
