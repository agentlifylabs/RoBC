"""
Semantic clustering for RoBC.

Manages cluster assignments using k-nearest neighbors with softmax weighting
for smooth interpolation between clusters.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class ClusterConfig:
    """Configuration for cluster assignment."""
    
    n_neighbors: int = 2
    softmax_temperature: float = 0.2
    min_similarity_threshold: float = 0.5
    high_confidence_threshold: float = 0.9


@dataclass
class Cluster:
    """Represents a semantic cluster."""
    
    id: int
    centroid: np.ndarray
    name: Optional[str] = None
    category: Optional[str] = None
    
    def similarity(self, embedding: np.ndarray) -> float:
        """Compute cosine similarity to the cluster centroid."""
        norm_a = np.linalg.norm(self.centroid)
        norm_b = np.linalg.norm(embedding)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(self.centroid, embedding) / (norm_a * norm_b))


class ClusterManager:
    """
    Manages semantic clustering for contextual routing.
    
    Uses k-nearest neighbors with softmax weighting to assign prompts
    to clusters based on embedding similarity.
    """
    
    def __init__(
        self,
        clusters: Optional[List[Cluster]] = None,
        config: Optional[ClusterConfig] = None,
    ):
        self.clusters = clusters or []
        self.config = config or ClusterConfig()
    
    @classmethod
    def from_centroids(
        cls,
        centroids: List[np.ndarray],
        names: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        config: Optional[ClusterConfig] = None,
    ) -> "ClusterManager":
        """Create a ClusterManager from a list of centroid embeddings."""
        clusters = []
        for i, centroid in enumerate(centroids):
            name = names[i] if names and i < len(names) else None
            category = categories[i] if categories and i < len(categories) else None
            clusters.append(Cluster(id=i, centroid=np.array(centroid), name=name, category=category))
        return cls(clusters=clusters, config=config)
    
    def add_cluster(self, centroid: np.ndarray, name: Optional[str] = None) -> int:
        """Add a new cluster and return its ID."""
        cluster_id = len(self.clusters)
        self.clusters.append(Cluster(id=cluster_id, centroid=np.array(centroid), name=name))
        return cluster_id
    
    def get_cluster_weights(self, embedding: np.ndarray) -> Dict[int, float]:
        """
        Get weighted cluster assignments for an embedding.
        
        Uses k-nearest neighbors with softmax weighting. If the top cluster
        has very high confidence, returns it alone (high confidence skip).
        """
        if not self.clusters:
            return {}
        
        embedding = np.array(embedding)
        
        similarities = [
            (cluster.id, cluster.similarity(embedding))
            for cluster in self.clusters
        ]
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        if not similarities:
            return {}
        
        top_cluster_id, top_similarity = similarities[0]
        
        if top_similarity >= self.config.high_confidence_threshold:
            return {top_cluster_id: 1.0}
        
        k = min(self.config.n_neighbors, len(similarities))
        top_k = similarities[:k]
        
        top_k = [(cid, sim) for cid, sim in top_k if sim >= self.config.min_similarity_threshold]
        
        if not top_k:
            return {top_cluster_id: 1.0}
        
        if len(top_k) == 1:
            return {top_k[0][0]: 1.0}
        
        temp = self.config.softmax_temperature
        max_sim = max(sim for _, sim in top_k)
        exp_sims = [(cid, np.exp((sim - max_sim) / temp)) for cid, sim in top_k]
        total = sum(exp for _, exp in exp_sims)
        
        return {cid: exp / total for cid, exp in exp_sims}
    
    def get_primary_cluster(self, embedding: np.ndarray) -> int:
        """Get the single best cluster for an embedding."""
        weights = self.get_cluster_weights(embedding)
        if not weights:
            return 0
        return max(weights, key=weights.get)
    
    @property
    def n_clusters(self) -> int:
        """Number of clusters."""
        return len(self.clusters)
