import ctypes
from ctypes import Structure, POINTER, c_float, c_char, c_int, c_double

class TReg(Structure):
    _fields_ = [("embedding", c_float * 128),
                ("person_id", c_char * 100)]

class TNode(Structure):
    pass

TNode._fields_ = [("key", ctypes.c_void_p),
                  ("esq", POINTER(TNode)),
                  ("dir", POINTER(TNode))]

class Tarv(Structure):
    _fields_ = [("k", c_int),
                ("dist", ctypes.CFUNCTYPE(c_double, ctypes.c_void_p, ctypes.c_void_p)),
                ("cmp", ctypes.CFUNCTYPE(c_int, ctypes.c_void_p, ctypes.c_void_p, c_int)),
                ("raiz", POINTER(TNode))]

# Carregar a biblioteca C
lib = ctypes.CDLL("./libkdtree.so")

# Definir as assinaturas das funções
lib.kdtree_construir.argtypes = []
lib.kdtree_construir.restype = None

lib.inserir_ponto.argtypes = [TReg]
lib.inserir_ponto.restype = None

lib.get_tree.restype = POINTER(Tarv)

lib.buscar_n_vizinhos_proximos.argtypes = [POINTER(Tarv), TReg, POINTER(TReg), c_int]
lib.buscar_n_vizinhos_proximos.restype = None
