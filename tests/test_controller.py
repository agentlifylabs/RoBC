"""Tests for the RoBC controller."""

import numpy as np
import pytest
from robc import Controller


class TestController:
    def test_initialization(self):
        controller = Controller(
            models=["model-a", "model-b"],
            n_clusters=5,
        )
        
        assert len(controller.models) == 2
        assert controller.n_clusters == 5
    
    def test_initialization_with_centroids(self):
        centroids = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        ]
        controller = Controller(
            models=["model-a", "model-b"],
            cluster_centroids=centroids,
        )
        
        assert controller.n_clusters == 3
        assert controller.cluster_manager.n_clusters == 3
    
    def test_route_returns_valid_model(self):
        controller = Controller(
            models=["model-a", "model-b", "model-c"],
            n_clusters=3,
        )
        
        embedding = np.random.randn(768)
        selected = controller.route(embedding)
        
        assert selected in controller.models
    
    def test_route_with_excluded_models(self):
        controller = Controller(
            models=["model-a", "model-b", "model-c"],
            n_clusters=3,
        )
        
        embedding = np.random.randn(768)
        
        for _ in range(10):
            selected = controller.route(embedding, excluded_models=["model-a", "model-b"])
            assert selected == "model-c"
    
    def test_route_with_details(self):
        controller = Controller(
            models=["model-a", "model-b"],
            n_clusters=3,
        )
        
        embedding = np.random.randn(768)
        result = controller.route_with_details(embedding)
        
        assert "selected_model" in result
        assert "cluster_weights" in result
        assert "posteriors" in result
        assert "samples" in result
        assert result["selected_model"] in controller.models
    
    def test_update(self):
        centroids = [np.random.randn(768) for _ in range(3)]
        controller = Controller(
            models=["model-a", "model-b"],
            cluster_centroids=centroids,
        )
        
        embedding = centroids[0] + np.random.randn(768) * 0.01
        
        for _ in range(50):
            controller.update("model-a", embedding, quality_score=0.95)
        
        posterior = controller.posterior_manager.get_posterior("model-a", 0)
        assert posterior.mean > 0.8
    
    def test_add_model(self):
        controller = Controller(
            models=["model-a"],
            n_clusters=3,
        )
        
        controller.add_model("model-b")
        
        assert "model-b" in controller.models
        embedding = np.random.randn(768)
        selected = controller.route(embedding)
        assert selected in ["model-a", "model-b"]
    
    def test_stats(self):
        controller = Controller(
            models=["model-a", "model-b"],
            n_clusters=3,
        )
        
        embedding = np.random.randn(768)
        for _ in range(100):
            controller.route(embedding)
        
        stats = controller.get_stats()
        
        assert stats["total_selections"] == 100
        assert sum(stats["model_selections"].values()) == 100
    
    def test_learning_improves_selection(self):
        np.random.seed(42)
        
        centroids = [np.random.randn(768) for _ in range(3)]
        for i, c in enumerate(centroids):
            centroids[i] = c / np.linalg.norm(c)
        
        controller = Controller(
            models=["good-model", "bad-model"],
            cluster_centroids=centroids,
        )
        
        embedding = centroids[0]
        
        for _ in range(100):
            selected = controller.route(embedding)
            if selected == "good-model":
                controller.update("good-model", embedding, quality_score=0.95)
            else:
                controller.update("bad-model", embedding, quality_score=0.3)
        
        selections = {"good-model": 0, "bad-model": 0}
        for _ in range(100):
            selected = controller.route(embedding)
            selections[selected] += 1
        
        assert selections["good-model"] > selections["bad-model"]
