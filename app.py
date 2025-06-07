from fastapi import FastAPI
from pydantic import BaseModel
import ctypes
from typing import List

app = FastAPI()

# Estruturas compatíveis com a biblioteca C
class TReg(ctypes.Structure):
    _fields_ = [("embedding", ctypes.c_float * 128),
                ("person_id", ctypes.c_char * 100)]

class Tarv(ctypes.Structure):
    _fields_ = [("raiz", ctypes.c_void_p),
                ("cmp", ctypes.c_void_p),
                ("dist", ctypes.c_void_p),
                ("k", ctypes.c_int)]

# Carrega a biblioteca compartilhada
lib = ctypes.CDLL('./kdtree.so')

# Configura os tipos de retorno e parâmetros
lib.kdtree_construir.restype = None
lib.inserir_ponto.argtypes = [TReg]
lib.buscar_n_vizinhos_proximos.argtypes = [ctypes.POINTER(Tarv), TReg, ctypes.POINTER(TReg), ctypes.c_int]
lib.buscar_n_vizinhos_proximos.restype = None

# Modelos Pydantic
class FaceEmbedding(BaseModel):
    embedding: List[float]
    person_id: str

class FaceQuery(BaseModel):
    embedding: List[float]
    k: int = 1

@app.post("/construir-arvore")
def construir_arvore():
    lib.kdtree_construir()
    return {"status": "Árvore construída"}

@app.post("/inserir")
def inserir(face: FaceEmbedding):
    registro = TReg()
    registro.embedding = (ctypes.c_float * 128)(*face.embedding)
    registro.person_id = face.person_id.encode('utf-8')
    lib.inserir_ponto(registro)
    return {"status": "Face inserida"}

@app.post("/buscar")
def buscar(query: FaceQuery):
    query_reg = TReg()
    query_reg.embedding = (ctypes.c_float * 128)(*query.embedding)
    
    resultados = (TReg * query.k)()
    arvore = Tarv()
    
    lib.buscar_n_vizinhos_proximos(ctypes.byref(arvore), query_reg, resultados, query.k)
    
    output = []
    for i in range(query.k):
        if resultados[i].person_id:
            output.append({
                "person_id": resultados[i].person_id.decode('utf-8'),
                "embedding": list(resultados[i].embedding)
            })
    
    return {"resultados": output}
