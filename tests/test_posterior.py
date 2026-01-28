"""Tests for posterior management."""

import numpy as np
import pytest
from robc.posterior import GaussianPosterior, PosteriorManager


class TestGaussianPosterior:
    def test_default_values(self):
        posterior = GaussianPosterior()
        
        assert posterior.mean == 0.5
        assert posterior.variance == 0.25
        assert posterior.observations == 0
    
    def test_sample(self):
        np.random.seed(42)
        posterior = GaussianPosterior(mean=0.8, variance=0.01)
        
        samples = [posterior.sample() for _ in range(1000)]
        sample_mean = np.mean(samples)
        
        assert abs(sample_mean - 0.8) < 0.05
    
    def test_update(self):
        posterior = GaussianPosterior(mean=0.5, variance=0.25)
        
        updated = posterior.update(outcome=0.9, observation_variance=0.01)
        
        assert updated.mean > 0.5
        assert updated.variance < 0.25
        assert updated.observations == 1
    
    def test_multiple_updates(self):
        posterior = GaussianPosterior(mean=0.5, variance=0.25)
        
        for _ in range(100):
            posterior = posterior.update(outcome=0.9, observation_variance=0.01)
        
        assert abs(posterior.mean - 0.9) < 0.05
        assert posterior.variance < 0.01
    
    def test_serialization(self):
        posterior = GaussianPosterior(mean=0.7, variance=0.15, observations=5)
        
        data = posterior.to_dict()
        restored = GaussianPosterior.from_dict(data)
        
        assert restored.mean == posterior.mean
        assert restored.variance == posterior.variance
        assert restored.observations == posterior.observations


class TestPosteriorManager:
    def test_initialization(self):
        models = ["model-a", "model-b"]
        manager = PosteriorManager(models=models, n_clusters=5)
        
        assert len(manager.models) == 2
        assert manager.n_clusters == 5
    
    def test_get_posterior(self):
        models = ["model-a"]
        manager = PosteriorManager(models=models, n_clusters=3)
        
        posterior = manager.get_posterior("model-a", 0)
        
        assert posterior.mean == manager.prior_mean
        assert posterior.variance == manager.prior_variance
    
    def test_update(self):
        models = ["model-a"]
        manager = PosteriorManager(models=models, n_clusters=3)
        
        manager.update("model-a", 0, 0.9)
        
        posterior = manager.get_posterior("model-a", 0)
        assert posterior.mean > manager.prior_mean
        assert posterior.observations == 1
    
    def test_aggregated_posterior(self):
        models = ["model-a"]
        manager = PosteriorManager(models=models, n_clusters=3)
        
        manager._posteriors["model-a"][0] = GaussianPosterior(mean=0.9, variance=0.01)
        manager._posteriors["model-a"][1] = GaussianPosterior(mean=0.5, variance=0.01)
        
        cluster_weights = {0: 0.7, 1: 0.3}
        aggregated = manager.get_aggregated_posterior("model-a", cluster_weights)
        
        expected_mean = 0.7 * 0.9 + 0.3 * 0.5
        assert abs(aggregated.mean - expected_mean) < 0.001
    
    def test_add_model(self):
        models = ["model-a"]
        manager = PosteriorManager(models=models, n_clusters=3)
        
        manager.add_model("model-b")
        
        assert "model-b" in manager.models
        posterior = manager.get_posterior("model-b", 0)
        assert posterior.mean == manager.prior_mean
