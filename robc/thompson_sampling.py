"""
Thompson Sampling for RoBC.

Implements the exploration-exploitation strategy that makes RoBC an online
learning router, balancing trying new options with exploiting known good ones.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from robc.posterior import GaussianPosterior, PosteriorManager


@dataclass
class SamplerConfig:
    """Configuration for Thompson Sampling."""
    
    exploration_bonus: float = 0.02
    min_variance: float = 0.001


class ThompsonSampler:
    """
    Thompson Sampling for model selection.
    
    Samples from each model's quality posterior and selects the model
    with the highest sample, naturally balancing exploration and exploitation.
    """
    
    def __init__(self, config: Optional[SamplerConfig] = None):
        self.config = config or SamplerConfig()
    
    def sample_with_bonus(self, posterior: GaussianPosterior) -> float:
        """
        Sample from posterior with exploration bonus for underexplored options.
        
        The exploration bonus decreases as we gather more observations,
        encouraging exploration of less-tested models.
        """
        n_obs = posterior.observations
        bonus = self.config.exploration_bonus / (1 + np.log1p(n_obs))
        
        effective_variance = max(posterior.variance + bonus, self.config.min_variance)
        
        return float(np.random.normal(posterior.mean, np.sqrt(effective_variance)))
    
    def select_model(
        self,
        posteriors: Dict[str, GaussianPosterior],
        excluded_models: Optional[List[str]] = None,
    ) -> str:
        """
        Select a model using Thompson Sampling.
        
        Args:
            posteriors: Dictionary mapping model names to their posteriors
            excluded_models: Models to exclude from selection
            
        Returns:
            The selected model name
        """
        excluded = set(excluded_models or [])
        available = {m: p for m, p in posteriors.items() if m not in excluded}
        
        if not available:
            raise ValueError("No models available for selection")
        
        samples = {
            model: self.sample_with_bonus(posterior)
            for model, posterior in available.items()
        }
        
        return max(samples, key=samples.get)
    
    def select_with_scores(
        self,
        posteriors: Dict[str, GaussianPosterior],
    ) -> Tuple[str, Dict[str, float]]:
        """
        Select a model and return all sampled scores.
        
        Useful for debugging and understanding selection decisions.
        """
        samples = {
            model: self.sample_with_bonus(posterior)
            for model, posterior in posteriors.items()
        }
        
        selected = max(samples, key=samples.get)
        return selected, samples
    
    def get_ucb_scores(
        self,
        posteriors: Dict[str, GaussianPosterior],
        confidence: float = 2.0,
    ) -> Dict[str, float]:
        """
        Get Upper Confidence Bound scores (alternative to sampling).
        
        UCB is a deterministic alternative that may be preferred in some scenarios.
        """
        return {
            model: posterior.mean + confidence * posterior.std
            for model, posterior in posteriors.items()
        }
