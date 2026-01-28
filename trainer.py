#!/usr/bin/env python3
"""
RoBC Trainer

Train RoBC cluster centroids and initialize posteriors from historical data.
Unlike RoRF which trains a static classifier, RoBC training focuses on:
1. Learning good cluster centroids from your data distribution
2. Initializing posteriors with prior knowledge (optional)

The trained artifacts can then be used with the Controller for online learning.
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from sklearn.cluster import KMeans

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False


@dataclass
class TrainingConfig:
    """Configuration for RoBC training."""
    n_clusters: int = 10
    embedding_dim: int = 768
    random_state: int = 42
    prior_mean: float = 0.5
    prior_variance: float = 0.25


def load_embeddings_from_dataset(
    dataset_path: str,
    embedding_column: str = "embedding",
    split: str = "train",
) -> np.ndarray:
    """Load embeddings from a HuggingFace dataset."""
    if not HAS_DATASETS:
        raise ImportError("Please install datasets: pip install datasets")
    
    dataset = load_dataset(dataset_path, split=split)
    embeddings = np.array(dataset[embedding_column])
    return embeddings


def load_embeddings_from_file(path: str) -> np.ndarray:
    """Load embeddings from a numpy or JSON file."""
    path = Path(path)
    
    if path.suffix == ".npy":
        return np.load(path)
    elif path.suffix == ".json":
        with open(path) as f:
            data = json.load(f)
        return np.array(data["embeddings"])
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def train_clusters(
    embeddings: np.ndarray,
    config: TrainingConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Train cluster centroids using K-Means.
    
    Returns:
        centroids: Cluster centroid vectors
        labels: Cluster assignments for each embedding
    """
    print(f"Training {config.n_clusters} clusters on {len(embeddings)} embeddings...")
    
    kmeans = KMeans(
        n_clusters=config.n_clusters,
        random_state=config.random_state,
        n_init=10,
    )
    labels = kmeans.fit_predict(embeddings)
    centroids = kmeans.cluster_centers_
    
    centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
    
    print(f"Cluster sizes: {np.bincount(labels)}")
    
    return centroids, labels


def initialize_posteriors_from_scores(
    models: List[str],
    n_clusters: int,
    scores: Optional[Dict[str, Dict[int, List[float]]]] = None,
    config: Optional[TrainingConfig] = None,
) -> Dict[str, Dict[int, Dict]]:
    """
    Initialize posteriors from historical quality scores.
    
    Args:
        models: List of model identifiers
        n_clusters: Number of clusters
        scores: Optional dict of {model: {cluster_id: [scores]}}
        config: Training configuration
        
    Returns:
        Posterior data structure for loading into Controller
    """
    config = config or TrainingConfig()
    posteriors = {}
    
    for model in models:
        posteriors[model] = {}
        for cluster_id in range(n_clusters):
            if scores and model in scores and cluster_id in scores[model]:
                model_scores = scores[model][cluster_id]
                mean = np.mean(model_scores)
                n = len(model_scores)
                variance = config.prior_variance / (1 + n * config.prior_variance / 0.01)
            else:
                mean = config.prior_mean
                variance = config.prior_variance
                n = 0
            
            posteriors[model][str(cluster_id)] = {
                "mean": float(mean),
                "variance": float(variance),
                "observations": int(n),
            }
    
    return posteriors


def save_training_artifacts(
    output_dir: str,
    centroids: np.ndarray,
    posteriors: Dict,
    models: List[str],
    config: TrainingConfig,
):
    """Save all training artifacts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(output_dir / "centroids.npy", centroids)
    print(f"Saved centroids to {output_dir / 'centroids.npy'}")
    
    posteriors_data = {
        "models": models,
        "n_clusters": config.n_clusters,
        "posteriors": posteriors,
    }
    with open(output_dir / "posteriors.json", "w") as f:
        json.dump(posteriors_data, f, indent=2)
    print(f"Saved posteriors to {output_dir / 'posteriors.json'}")
    
    config_data = {
        "n_clusters": config.n_clusters,
        "embedding_dim": config.embedding_dim,
        "prior_mean": config.prior_mean,
        "prior_variance": config.prior_variance,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_data, f, indent=2)
    print(f"Saved config to {output_dir / 'config.json'}")


def main():
    parser = argparse.ArgumentParser(
        description="Train RoBC cluster centroids and initialize posteriors"
    )
    
    parser.add_argument(
        "--embeddings",
        type=str,
        help="Path to embeddings file (.npy or .json) or HuggingFace dataset",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["openai:gpt-5.2", "google:gemini-2.5-flash", "anthropic:claude-4.5-sonnet"],
        help="List of model identifiers",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=10,
        help="Number of semantic clusters",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./robc_artifacts",
        help="Output directory for trained artifacts",
    )
    parser.add_argument(
        "--prior-mean",
        type=float,
        default=0.5,
        help="Prior mean for model quality",
    )
    parser.add_argument(
        "--prior-variance",
        type=float,
        default=0.25,
        help="Prior variance for model quality",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for reproducibility",
    )
    
    args = parser.parse_args()
    
    config = TrainingConfig(
        n_clusters=args.n_clusters,
        prior_mean=args.prior_mean,
        prior_variance=args.prior_variance,
        random_state=args.random_state,
    )
    
    print("=" * 60)
    print("RoBC Training")
    print("=" * 60)
    print(f"Models: {args.models}")
    print(f"Clusters: {config.n_clusters}")
    print(f"Output: {args.output_dir}")
    
    if args.embeddings:
        if args.embeddings.startswith("hf://") or "/" in args.embeddings:
            embeddings = load_embeddings_from_dataset(
                args.embeddings.replace("hf://", "")
            )
        else:
            embeddings = load_embeddings_from_file(args.embeddings)
        
        config.embedding_dim = embeddings.shape[1]
        centroids, labels = train_clusters(embeddings, config)
    else:
        print("\nNo embeddings provided. Generating random centroids for demonstration...")
        centroids = np.random.randn(config.n_clusters, config.embedding_dim)
        centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
    
    posteriors = initialize_posteriors_from_scores(
        models=args.models,
        n_clusters=config.n_clusters,
        config=config,
    )
    
    save_training_artifacts(
        output_dir=args.output_dir,
        centroids=centroids,
        posteriors=posteriors,
        models=args.models,
        config=config,
    )
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"\nTo use the trained artifacts:")
    print(f"""
import numpy as np
from robc import Controller

centroids = [np.load("{args.output_dir}/centroids.npy")[i] for i in range({config.n_clusters})]

controller = Controller(
    models={args.models},
    cluster_centroids=centroids,
    posteriors_path="{args.output_dir}/posteriors.json",
)
""")


if __name__ == "__main__":
    main()
