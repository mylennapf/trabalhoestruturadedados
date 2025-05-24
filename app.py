from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from kdtree_wrapper import lib, TReg
import numpy as np

app = FastAPI()

class FaceEmbedding(BaseModel):
    embedding: List[float]  # 128 floats
    person_id: str          # ID da pessoa

class FaceQuery(BaseModel):
    embedding: List[float]  # 128 floats para consulta
    k: int = 1              # Número de vizinhos a retornar

@app.post("/construir-arvore")
def constroi_arvore():
    lib.kdtree_construir()
    return {"mensagem": "Árvore KD inicializada com sucesso."}

@app.post("/inserir")
def inserir(face: FaceEmbedding):
    # Converte o embedding para o formato C
    embedding_array = (ctypes.c_float * 128)(*face.embedding)
    
    # Cria o registro
    novo_registro = TReg()
    novo_registro.embedding = embedding_array
    novo_registro.person_id = face.person_id.encode('utf-8')[:99]
    
    # Insere na árvore
    lib.inserir_ponto(novo_registro)
    
    return {"mensagem": f"Embedding de '{face.person_id}' inserido com sucesso."}

@app.post("/buscar")
def buscar(query: FaceQuery):
    # Prepara a query
    query_embedding = (ctypes.c_float * 128)(*query.embedding)
    query_reg = TReg()
    query_reg.embedding = query_embedding
    
    # Prepara o array de resultados
    resultados = (TReg * query.k)()
    
    # Executa a busca
    arv = lib.get_tree()
    lib.buscar_n_vizinhos_proximos(arv, query_reg, resultados, query.k)
    
    # Converte os resultados para o formato Python
    resultados_python = []
    for i in range(query.k):
        if resultados[i].person_id:  # Verifica se há resultado
            embedding = [float(resultados[i].embedding[j]) for j in range(128)]
            person_id = resultados[i].person_id.decode('utf-8')
            resultados_python.append({
                "embedding": embedding,
                "person_id": person_id
            })
    
    return {"resultados": resultados_python}
