"""Tests for cluster management."""

import numpy as np
import pytest
from robc.cluster import ClusterManager, ClusterConfig, Cluster


class TestCluster:
    def test_similarity(self):
        centroid = np.array([1.0, 0.0, 0.0])
        cluster = Cluster(id=0, centroid=centroid)
        
        same = cluster.similarity(np.array([1.0, 0.0, 0.0]))
        assert abs(same - 1.0) < 0.001
        
        orthogonal = cluster.similarity(np.array([0.0, 1.0, 0.0]))
        assert abs(orthogonal) < 0.001
        
        opposite = cluster.similarity(np.array([-1.0, 0.0, 0.0]))
        assert abs(opposite + 1.0) < 0.001


class TestClusterManager:
    def test_from_centroids(self):
        centroids = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        ]
        manager = ClusterManager.from_centroids(centroids)
        
        assert manager.n_clusters == 3
    
    def test_get_primary_cluster(self):
        centroids = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
        ]
        manager = ClusterManager.from_centroids(centroids)
        
        embedding = np.array([0.9, 0.1, 0.0])
        primary = manager.get_primary_cluster(embedding)
        
        assert primary == 0
    
    def test_high_confidence_skip(self):
        centroids = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
        ]
        config = ClusterConfig(high_confidence_threshold=0.9)
        manager = ClusterManager.from_centroids(centroids, config=config)
        
        embedding = np.array([1.0, 0.0, 0.0])
        weights = manager.get_cluster_weights(embedding)
        
        assert len(weights) == 1
        assert weights[0] == 1.0
    
    def test_knn_weights(self):
        centroids = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.707, 0.707, 0.0]),
        ]
        config = ClusterConfig(
            n_neighbors=2,
            high_confidence_threshold=0.99,
            min_similarity_threshold=0.3,
        )
        manager = ClusterManager.from_centroids(centroids, config=config)
        
        embedding = np.array([0.9, 0.3, 0.0])
        embedding = embedding / np.linalg.norm(embedding)
        weights = manager.get_cluster_weights(embedding)
        
        assert len(weights) == 2
        assert sum(weights.values()) == pytest.approx(1.0)
