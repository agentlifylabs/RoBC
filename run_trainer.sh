#!/bin/bash
# Example training command for RoBC

python trainer.py \
    --models "openai:gpt-5.2" "google:gemini-2.5-flash" "anthropic:claude-4.5-sonnet" \
    --n-clusters 10 \
    --output-dir ./robc_artifacts \
    --prior-mean 0.5 \
    --prior-variance 0.25 \
    --random-state 42

# With embeddings from a file:
# python trainer.py \
#     --embeddings embeddings.npy \
#     --models "openai:gpt-5.2" "google:gemini-2.5-flash" \
#     --n-clusters 20 \
#     --output-dir ./my_robc_model

# With embeddings from HuggingFace:
# python trainer.py \
#     --embeddings "your-org/your-dataset" \
#     --models "meta:llama-4-maverick" "meta:llama-4-scout" \
#     --n-clusters 15 \
#     --output-dir ./llama_router
