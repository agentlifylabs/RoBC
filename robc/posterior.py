"""
Bayesian posterior management for RoBC.

Maintains Gaussian posteriors over model quality for each (model, cluster) pair,
enabling online learning from observed outcomes.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class GaussianPosterior:
    """
    Gaussian posterior distribution over model quality.
    
    Represents our belief about a model's quality for a specific cluster,
    updated via Bayesian inference as we observe outcomes.
    """
    
    mean: float = 0.5
    variance: float = 0.25
    observations: int = 0
    
    @property
    def std(self) -> float:
        """Standard deviation."""
        return np.sqrt(self.variance)
    
    def sample(self) -> float:
        """Sample from the posterior distribution."""
        return float(np.random.normal(self.mean, self.std))
    
    def update(self, outcome: float, observation_variance: float = 0.01) -> "GaussianPosterior":
        """
        Bayesian update with a new observation.
        
        Args:
            outcome: Observed quality score (0-1)
            observation_variance: Noise in the observation
            
        Returns:
            Updated posterior (new instance)
        """
        prior_precision = 1.0 / self.variance
        obs_precision = 1.0 / observation_variance
        
        new_precision = prior_precision + obs_precision
        new_variance = 1.0 / new_precision
        
        new_mean = new_variance * (
            self.mean * prior_precision + outcome * obs_precision
        )
        
        return GaussianPosterior(
            mean=new_mean,
            variance=new_variance,
            observations=self.observations + 1,
        )
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "mean": self.mean,
            "variance": self.variance,
            "observations": self.observations,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "GaussianPosterior":
        """Deserialize from dictionary."""
        return cls(
            mean=data.get("mean", 0.5),
            variance=data.get("variance", 0.25),
            observations=data.get("observations", 0),
        )


class PosteriorManager:
    """
    Manages posteriors for all (model, cluster) pairs.
    
    This is the core learning component of RoBC - it maintains and updates
    our beliefs about model quality across different semantic clusters.
    """
    
    def __init__(
        self,
        models: List[str],
        n_clusters: int,
        prior_mean: float = 0.5,
        prior_variance: float = 0.25,
        observation_variance: float = 0.01,
    ):
        self.models = list(models)
        self.n_clusters = n_clusters
        self.prior_mean = prior_mean
        self.prior_variance = prior_variance
        self.observation_variance = observation_variance
        
        self._posteriors: Dict[str, Dict[int, GaussianPosterior]] = {}
        self._initialize_posteriors()
    
    def _initialize_posteriors(self):
        """Initialize all posteriors with the prior."""
        for model in self.models:
            self._posteriors[model] = {}
            for cluster_id in range(self.n_clusters):
                self._posteriors[model][cluster_id] = GaussianPosterior(
                    mean=self.prior_mean,
                    variance=self.prior_variance,
                    observations=0,
                )
    
    def get_posterior(self, model: str, cluster_id: int) -> GaussianPosterior:
        """Get the posterior for a (model, cluster) pair."""
        if model not in self._posteriors:
            self._posteriors[model] = {}
        
        if cluster_id not in self._posteriors[model]:
            self._posteriors[model][cluster_id] = GaussianPosterior(
                mean=self.prior_mean,
                variance=self.prior_variance,
                observations=0,
            )
        
        return self._posteriors[model][cluster_id]
    
    def update(self, model: str, cluster_id: int, outcome: float):
        """Update the posterior for a (model, cluster) pair with an observation."""
        current = self.get_posterior(model, cluster_id)
        updated = current.update(outcome, self.observation_variance)
        self._posteriors[model][cluster_id] = updated
    
    def get_aggregated_posterior(
        self,
        model: str,
        cluster_weights: Dict[int, float],
    ) -> GaussianPosterior:
        """
        Get an aggregated posterior using weighted cluster assignments.
        
        Uses Gaussian Mixture Model approximation to combine posteriors
        from multiple clusters.
        """
        if not cluster_weights:
            return GaussianPosterior(mean=self.prior_mean, variance=self.prior_variance)
        
        posteriors = [
            (weight, self.get_posterior(model, cluster_id))
            for cluster_id, weight in cluster_weights.items()
        ]
        
        weighted_mean = sum(w * p.mean for w, p in posteriors)
        
        weighted_variance = sum(
            w * (p.variance + p.mean ** 2) for w, p in posteriors
        ) - weighted_mean ** 2
        
        weighted_variance = max(weighted_variance, 1e-6)
        
        total_observations = sum(p.observations for _, p in posteriors)
        
        return GaussianPosterior(
            mean=weighted_mean,
            variance=weighted_variance,
            observations=total_observations,
        )
    
    def add_model(self, model: str):
        """Add a new model with uninformative priors."""
        if model not in self.models:
            self.models.append(model)
            self._posteriors[model] = {}
            for cluster_id in range(self.n_clusters):
                self._posteriors[model][cluster_id] = GaussianPosterior(
                    mean=self.prior_mean,
                    variance=self.prior_variance,
                    observations=0,
                )
    
    def to_dict(self) -> Dict:
        """Serialize all posteriors."""
        return {
            model: {
                str(cluster_id): posterior.to_dict()
                for cluster_id, posterior in clusters.items()
            }
            for model, clusters in self._posteriors.items()
        }
    
    @classmethod
    def from_dict(
        cls,
        data: Dict,
        models: List[str],
        n_clusters: int,
        **kwargs,
    ) -> "PosteriorManager":
        """Deserialize from dictionary."""
        manager = cls(models=models, n_clusters=n_clusters, **kwargs)
        
        for model, clusters in data.items():
            if model not in manager._posteriors:
                manager._posteriors[model] = {}
            for cluster_id_str, posterior_data in clusters.items():
                cluster_id = int(cluster_id_str)
                manager._posteriors[model][cluster_id] = GaussianPosterior.from_dict(posterior_data)
        
        return manager
