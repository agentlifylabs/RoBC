"""
Main controller for RoBC routing.

Provides a simple interface for routing requests to LLM models using
Bayesian clustering and Thompson Sampling.
"""

import json
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from robc.cluster import ClusterManager, ClusterConfig, Cluster
from robc.posterior import GaussianPosterior, PosteriorManager
from robc.thompson_sampling import ThompsonSampler, SamplerConfig


class Controller:
    """
    RoBC Controller for LLM routing.
    
    Routes requests to the best model based on learned quality posteriors,
    using Thompson Sampling for exploration-exploitation balance.
    
    Example:
        ```python
        controller = Controller(
            models=["gpt-4o", "gpt-4o-mini", "claude-3.5-sonnet"],
            n_clusters=10,
        )
        
        # Route a request
        model = controller.route(embedding)
        
        # Update with feedback
        controller.update(model, embedding, quality_score=0.85)
        ```
    """
    
    def __init__(
        self,
        models: List[str],
        n_clusters: int = 10,
        cluster_centroids: Optional[List[np.ndarray]] = None,
        posteriors_path: Optional[str] = None,
        cluster_config: Optional[ClusterConfig] = None,
        sampler_config: Optional[SamplerConfig] = None,
        prior_mean: float = 0.5,
        prior_variance: float = 0.25,
        observation_variance: float = 0.01,
    ):
        """
        Initialize the RoBC controller.
        
        Args:
            models: List of model identifiers to route between
            n_clusters: Number of semantic clusters
            cluster_centroids: Pre-computed cluster centroids (optional)
            posteriors_path: Path to saved posteriors (optional)
            cluster_config: Configuration for cluster assignment
            sampler_config: Configuration for Thompson Sampling
            prior_mean: Prior mean for model quality
            prior_variance: Prior variance for model quality
            observation_variance: Observation noise for Bayesian updates
        """
        self.models = list(models)
        self.n_clusters = n_clusters
        
        if cluster_centroids:
            self.cluster_manager = ClusterManager.from_centroids(
                centroids=cluster_centroids,
                config=cluster_config,
            )
            self.n_clusters = len(cluster_centroids)
        else:
            self.cluster_manager = ClusterManager(config=cluster_config)
        
        self.posterior_manager = PosteriorManager(
            models=models,
            n_clusters=self.n_clusters,
            prior_mean=prior_mean,
            prior_variance=prior_variance,
            observation_variance=observation_variance,
        )
        
        if posteriors_path:
            self.load_posteriors(posteriors_path)
        
        self.sampler = ThompsonSampler(config=sampler_config)
        
        self._selection_count = 0
        self._model_selections: Dict[str, int] = {m: 0 for m in models}
    
    def route(
        self,
        embedding: np.ndarray,
        excluded_models: Optional[List[str]] = None,
    ) -> str:
        """
        Route a request to the best model.
        
        Args:
            embedding: The prompt embedding vector
            excluded_models: Models to exclude from consideration
            
        Returns:
            The selected model identifier
        """
        cluster_weights = self.cluster_manager.get_cluster_weights(embedding)
        
        posteriors = {}
        for model in self.models:
            if excluded_models and model in excluded_models:
                continue
            posteriors[model] = self.posterior_manager.get_aggregated_posterior(
                model, cluster_weights
            )
        
        selected = self.sampler.select_model(posteriors, excluded_models)
        
        self._selection_count += 1
        self._model_selections[selected] = self._model_selections.get(selected, 0) + 1
        
        return selected
    
    def route_with_details(
        self,
        embedding: np.ndarray,
        excluded_models: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Route a request and return detailed selection information.
        
        Useful for debugging and understanding routing decisions.
        """
        cluster_weights = self.cluster_manager.get_cluster_weights(embedding)
        
        posteriors = {}
        for model in self.models:
            if excluded_models and model in excluded_models:
                continue
            posteriors[model] = self.posterior_manager.get_aggregated_posterior(
                model, cluster_weights
            )
        
        selected, samples = self.sampler.select_with_scores(posteriors)
        
        self._selection_count += 1
        self._model_selections[selected] = self._model_selections.get(selected, 0) + 1
        
        return {
            "selected_model": selected,
            "cluster_weights": cluster_weights,
            "posteriors": {m: p.to_dict() for m, p in posteriors.items()},
            "samples": samples,
        }
    
    def update(
        self,
        model: str,
        embedding: np.ndarray,
        quality_score: float,
    ):
        """
        Update the posterior with an observed outcome.
        
        Args:
            model: The model that was used
            embedding: The prompt embedding
            quality_score: The observed quality (0-1)
        """
        cluster_weights = self.cluster_manager.get_cluster_weights(embedding)
        
        for cluster_id, weight in cluster_weights.items():
            if weight > 0.1:
                self.posterior_manager.update(model, cluster_id, quality_score)
    
    def add_model(self, model: str):
        """Add a new model with uninformative priors."""
        if model not in self.models:
            self.models.append(model)
            self.posterior_manager.add_model(model)
            self._model_selections[model] = 0
    
    def save_posteriors(self, path: str):
        """Save posteriors to a JSON file."""
        data = {
            "models": self.models,
            "n_clusters": self.n_clusters,
            "posteriors": self.posterior_manager.to_dict(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    
    def load_posteriors(self, path: str):
        """Load posteriors from a JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        
        posteriors_data = data.get("posteriors", {})
        for model, clusters in posteriors_data.items():
            if model not in self.posterior_manager._posteriors:
                self.posterior_manager._posteriors[model] = {}
            for cluster_id_str, posterior_data in clusters.items():
                cluster_id = int(cluster_id_str)
                self.posterior_manager._posteriors[model][cluster_id] = (
                    GaussianPosterior.from_dict(posterior_data)
                )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get routing statistics."""
        return {
            "total_selections": self._selection_count,
            "model_selections": dict(self._model_selections),
            "model_selection_pct": {
                m: count / max(self._selection_count, 1) * 100
                for m, count in self._model_selections.items()
            },
        }
    
    def get_model_posteriors(self, model: str) -> Dict[int, Dict]:
        """Get all posteriors for a specific model."""
        return {
            cluster_id: self.posterior_manager.get_posterior(model, cluster_id).to_dict()
            for cluster_id in range(self.n_clusters)
        }
