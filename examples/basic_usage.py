#!/usr/bin/env python3
"""
Basic usage example for RoBC.

This example demonstrates how to use RoBC for LLM routing with
simulated embeddings and quality feedback.
"""

import numpy as np
from robc import Controller


def simulate_embedding(text: str, dim: int = 768) -> np.ndarray:
    """Simulate an embedding (replace with real embedding in production)."""
    np.random.seed(hash(text) % 2**32)
    embedding = np.random.randn(dim)
    return embedding / np.linalg.norm(embedding)


def simulate_quality(model: str, task_type: str) -> float:
    """Simulate model quality based on task type."""
    quality_matrix = {
        "openai:gpt-5.2": {"reasoning": 0.93, "coding": 0.91, "creative": 0.86, "simple": 0.90},
        "google:gemini-2.5-flash": {"reasoning": 0.72, "coding": 0.76, "creative": 0.74, "simple": 0.84},
        "anthropic:claude-4.5-sonnet": {"reasoning": 0.91, "coding": 0.94, "creative": 0.93, "simple": 0.88},
    }
    base_quality = quality_matrix.get(model, {}).get(task_type, 0.7)
    return np.clip(base_quality + np.random.normal(0, 0.05), 0, 1)


def main():
    print("=" * 60)
    print("RoBC Basic Usage Example")
    print("=" * 60)
    
    models = ["openai:gpt-5.2", "google:gemini-2.5-flash", "anthropic:claude-4.5-sonnet"]
    
    centroids = [simulate_embedding(f"cluster_{i}") for i in range(5)]
    
    controller = Controller(
        models=models,
        cluster_centroids=centroids,
    )
    
    print(f"\nInitialized RoBC with {len(models)} models and {len(centroids)} clusters")
    
    tasks = [
        ("Explain quantum entanglement in simple terms", "reasoning"),
        ("Write a Python function to merge two sorted lists", "coding"),
        ("Write a haiku about artificial intelligence", "creative"),
        ("What is 2 + 2?", "simple"),
        ("Prove the Pythagorean theorem", "reasoning"),
        ("Debug this JavaScript async/await code", "coding"),
    ]
    
    print("\n" + "-" * 60)
    print("Phase 1: Initial routing (uninformative priors)")
    print("-" * 60)
    
    for prompt, task_type in tasks[:3]:
        embedding = simulate_embedding(prompt)
        selected = controller.route(embedding)
        quality = simulate_quality(selected, task_type)
        
        print(f"\nPrompt: {prompt[:50]}...")
        print(f"  Task type: {task_type}")
        print(f"  Selected: {selected}")
        print(f"  Quality: {quality:.3f}")
        
        controller.update(selected, embedding, quality)
    
    print("\n" + "-" * 60)
    print("Phase 2: Learning phase (100 simulated requests)")
    print("-" * 60)
    
    task_types = ["reasoning", "coding", "creative", "simple"]
    
    for i in range(100):
        task_type = np.random.choice(task_types)
        prompt = f"Task {i} of type {task_type}"
        embedding = simulate_embedding(prompt)
        
        selected = controller.route(embedding)
        quality = simulate_quality(selected, task_type)
        controller.update(selected, embedding, quality)
    
    stats = controller.get_stats()
    print(f"\nTotal selections: {stats['total_selections']}")
    print("Model distribution:")
    for model, pct in stats['model_selection_pct'].items():
        print(f"  {model}: {pct:.1f}%")
    
    print("\n" + "-" * 60)
    print("Phase 3: Routing after learning")
    print("-" * 60)
    
    for prompt, task_type in tasks[3:]:
        embedding = simulate_embedding(prompt)
        result = controller.route_with_details(embedding)
        
        print(f"\nPrompt: {prompt[:50]}...")
        print(f"  Task type: {task_type}")
        print(f"  Selected: {result['selected_model']}")
        print(f"  Model scores: ", end="")
        for model, score in sorted(result['samples'].items(), key=lambda x: -x[1]):
            print(f"{model.split('/')[-1]}={score:.3f} ", end="")
        print()
    
    print("\n" + "-" * 60)
    print("Phase 4: Save and reload")
    print("-" * 60)
    
    controller.save_posteriors("/tmp/robc_posteriors.json")
    print("Saved posteriors to /tmp/robc_posteriors.json")
    
    new_controller = Controller(
        models=models,
        cluster_centroids=centroids,
        posteriors_path="/tmp/robc_posteriors.json",
    )
    print("Loaded posteriors into new controller")
    
    embedding = simulate_embedding("Test prompt after reload")
    selected = new_controller.route(embedding)
    print(f"New controller selected: {selected}")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
