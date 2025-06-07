import requests
import numpy as np
import random

API_URL = "http://localhost:8000"

# 1. Construir árvore
requests.post(f"{API_URL}/construir-arvore")

# 2. Gerar embeddings de teste
def generate_embedding():
    return [random.random() for _ in range(128)]

known_faces = {
    "eu": generate_embedding(),
    "pessoa1": generate_embedding(),
    "pessoa2": generate_embedding()
}

# 3. Inserir faces
for name, emb in known_faces.items():
    requests.post(f"{API_URL}/inserir", json={
        "embedding": emb,
        "person_id": name
    })

# 4. Inserir faces aleatórias
for i in range(997):
    requests.post(f"{API_URL}/inserir", json={
        "embedding": generate_embedding(),
        "person_id": f"random_{i}"
    })

# 5. Testar buscas
for name, emb in known_faces.items():
    response = requests.post(f"{API_URL}/buscar", json={
        "embedding": emb,
        "k": 1
    })
    
    print(f"Consulta: {name}")
    print("Resultado:", response.json()["resultados"][0]["person_id"])
    print("Distância:", np.linalg.norm(np.array(emb) - np.array(response.json()["resultados"][0]["embedding"]))
    print()
