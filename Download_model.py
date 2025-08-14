import os

# Set Hugging Face cache directory to D:/HF_model
os.environ["TRANSFORMERS_CACHE"] = "D:/HF_model"

from transformers import AutoTokenizer, AutoModel

# # === Qwen3 Embedding Model ===
# print("Downloading Qwen3 Embedding Model...")
# embedding_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B", trust_remote_code=True)
# embedding_model = AutoModel.from_pretrained("Qwen/Qwen3-Embedding-0.6B", trust_remote_code=True)

# # Save locally
# embedding_tokenizer.save_pretrained("./qwen-embedding")
# embedding_model.save_pretrained("./qwen-embedding")
# print("Saved Qwen3 Embedding model to ./qwen-embedding")

# === Qwen3 Reranker Model ===
print("Downloading Qwen3 Reranker Model...")
reranker_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Reranker-0.6B", trust_remote_code=True)
reranker_model = AutoModel.from_pretrained("Qwen/Qwen3-Reranker-0.6B", trust_remote_code=True)

# Save locally
reranker_tokenizer.save_pretrained("./qwen-reranker")
reranker_model.save_pretrained("./qwen-reranker")
print("Saved Qwen3 Reranker model to ./qwen-reranker")
