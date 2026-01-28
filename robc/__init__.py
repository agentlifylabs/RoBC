"""
RoBC - Routing on Bayesian Clustering

An online learning LLM router that uses Thompson Sampling and semantic clustering
to adaptively route requests to the best model based on learned quality posteriors.
"""

from robc.controller import Controller
from robc.cluster import ClusterManager
from robc.posterior import GaussianPosterior, PosteriorManager
from robc.thompson_sampling import ThompsonSampler

__version__ = "0.1.0"
__all__ = [
    "Controller",
    "ClusterManager",
    "GaussianPosterior",
    "PosteriorManager",
    "ThompsonSampler",
]
