# ============================================================
#  CONFIG
# ============================================================

HF_API_KEY = "Enter api key"

# LLM model — runs on HuggingFace cloud via huggingface_hub SDK
LLM_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# Embedding model (runs on your CPU)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# NLI verification model (runs on your CPU)
NLI_MODEL = "cross-encoder/nli-MiniLM2-L6-H768"

# Number of chunks to retrieve per question
TOP_K = 5
