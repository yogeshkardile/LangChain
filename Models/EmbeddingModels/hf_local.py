from langchain_huggingface import HuggingFaceEmbeddings

# Load local embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

# Convert text into embeddings
text = "Machine learning is interesting."
vector = embeddings.embed_query(text)

print(vector)
print("Embedding dimensions:", len(vector))
