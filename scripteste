import requests
import numpy as np
import random

# Configuração da API
API_URL = "http://localhost:8000"

# Funções auxiliares
def generate_random_embedding():
    return [random.random() for _ in range(128)]

def insert_face(embedding, person_id):
    response = requests.post(
        f"{API_URL}/inserir",
        json={"embedding": embedding, "person_id": person_id}
    )
    return response.json()

def search_neighbors(embedding, k=1):
    response = requests.get(
        f"{API_URL}/buscar",
        params={"embedding": str(embedding), "k": k}
    )
    return response.json()

# Construir a árvore
requests.post(f"{API_URL}/construir-arvore")

# Faces conhecidas
known_faces = {
    "voce": generate_random_embedding(),
    "pessoa1": generate_random_embedding(),
    "pessoa2": generate_random_embedding()
}

# Inserir faces
for name, embedding in known_faces.items():
    insert_face(embedding, name)

# Inserir faces aleatórias
for i in range(997):
    insert_face(generate_random_embedding(), f"random_{i}")

# Testar reconhecimento
results = []
for name, embedding in known_faces.items():
    search_result = search_neighbors(embedding)
    neighbor_id = search_result["resultados"][0]["person_id"]
    results.append(neighbor_id == name)

# Resultado final
print("Teste concluído!")
if all(results):
    print("✅ Todas as faces foram reconhecidas corretamente")
else:
    print("❌ Algumas faces não foram reconhecidas:")
    for i, name in enumerate(known_faces):
        print(f"- {name}: {'✅' if results[i] else '❌'}")
