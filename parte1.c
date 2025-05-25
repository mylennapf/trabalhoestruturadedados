#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <string.h>
#include <assert.h>
#include <math.h>

/* 1º - Estrutura para embeddings faciais */
typedef struct {
    float embedding[128];  // Vetor de 128 floats
    char person_id[100];   // ID com 100 caracteres
} treg;

/* Estrutura do nó da KD-Tree */
typedef struct _node {
    void* key;
    struct _node* esq;
    struct _node* dir;
} tnode;

typedef struct {
    tnode* raiz;
    int (*cmp)(void*, void*, int);
    double (*dist)(void*, void*);
    int k;
} tarv;

/* 2º - Função de distância euclidiana quadrada */
double distancia(void* a, void* b) {
    float* emb_a = ((treg*)a)->embedding;
    float* emb_b = ((treg*)b)->embedding;
    double sum = 0.0;
    for (int i = 0; i < 128; i++) {
        double diff = emb_a[i] - emb_b[i];
        sum += diff * diff;
    }
    return sum;
}

/* Funções auxiliares */
void* aloca_reg(float embedding[128], const char* person_id) {
    treg* reg = malloc(sizeof(treg));
    memcpy(reg->embedding, embedding, sizeof(float) * 128);
    strncpy(reg->person_id, person_id, 99);
    reg->person_id[99] = '\0';
    return reg;
}

int comparador(void* a, void* b, int pos) {
    float* emb_a = ((treg*)a)->embedding;
    float* emb_b = ((treg*)b)->embedding;
    return (emb_a[pos] > emb_b[pos]) - (emb_a[pos] < emb_b[pos]);
}

/* 8º - Implementação do MinHeap integrado */
typedef struct {
    double distance;
    treg* record;
} HeapItem;

typedef struct {
    HeapItem* items;
    int capacity;
    int size;
} MinHeap;

MinHeap* criar_minheap(int capacity) {
    MinHeap* heap = malloc(sizeof(MinHeap));
    heap->items = malloc(sizeof(HeapItem) * capacity);
    heap->capacity = capacity;
    heap->size = 0;
    return heap;
}

void liberar_minheap(MinHeap* heap) {
    free(heap->items);
    free(heap);
}

void heapify_up(MinHeap* heap, int index) {
    while (index > 0) {
        int parent = (index - 1) / 2;
        if (heap->items[index].distance < heap->items[parent].distance) {
            HeapItem temp = heap->items[index];
            heap->items[index] = heap->items[parent];
            heap->items[parent] = temp;
            index = parent;
        } else {
            break;
        }
    }
}

void heapify_down(MinHeap* heap, int index) {
    while (1) {
        int left = 2 * index + 1;
        int right = 2 * index + 2;
        int smallest = index;
        
        if (left < heap->size && heap->items[left].distance < heap->items[smallest].distance) {
            smallest = left;
        }
        if (right < heap->size && heap->items[right].distance < heap->items[smallest].distance) {
            smallest = right;
        }
        
        if (smallest != index) {
            HeapItem temp = heap->items[index];
            heap->items[index] = heap->items[smallest];
            heap->items[smallest] = temp;
            index = smallest;
        } else {
            break;
        }
    }
}

void inserir_heap(MinHeap* heap, double distance, treg* record) {
    if (heap->size < heap->capacity) {
        heap->items[heap->size].distance = distance;
        heap->items[heap->size].record = record;
        heapify_up(heap, heap->size);
        heap->size++;
    } else if (distance < heap->items[0].distance) {
        heap->items[0].distance = distance;
        heap->items[0].record = record;
        heapify_down(heap, 0);
    }
}

/* 3º - Função de busca com MinHeap integrado */
void _kdtree_busca_knn(tarv* arv, tnode* atual, void* key, int profund, MinHeap* heap) {
    if (atual != NULL) {
        double dist_atual = arv->dist(atual->key, key);
        inserir_heap(heap, dist_atual, (treg*)atual->key);
        
        int pos = profund % arv->k;
        int comp = arv->cmp(key, atual->key, pos);
        
        tnode* lado_principal = comp < 0 ? atual->esq : atual->dir;
        tnode* lado_oposto = comp < 0 ? atual->dir : atual->esq;
        
        _kdtree_busca_knn(arv, lado_principal, key, profund + 1, heap);
        
        double axis_diff = ((treg*)key)->embedding[pos] - ((treg*)atual->key)->embedding[pos];
        if (heap->size < heap->capacity || axis_diff * axis_diff < heap->items[0].distance) {
            _kdtree_busca_knn(arv, lado_oposto, key, profund + 1, heap);
        }
    }
}

void kdtree_busca_knn(tarv* arv, void* key, treg* resultados, int k) {
    MinHeap* heap = criar_minheap(k);
    _kdtree_busca_knn(arv, arv->raiz, key, 0, heap);
    
    // Extrai os resultados em ordem crescente
    for (int i = 0; i < heap->size; i++) {
        memcpy(&resultados[i], heap->items[i].record, sizeof(treg));
    }
    
    liberar_minheap(heap);
}

/* Funções da KD-Tree */
void kdtree_constroi(tarv* arv, int (*cmp)(void*, void*, int), double (*dist)(void*, void*), int k) {
    arv->raiz = NULL;
    arv->cmp = cmp;
    arv->dist = dist;
    arv->k = k;
}

void _kdtree_insere(tnode** raiz, void* key, int (*cmp)(void*, void*, int), int profund, int k) {
    if (*raiz == NULL) {
        *raiz = malloc(sizeof(tnode));
        (*raiz)->key = key;
        (*raiz)->esq = NULL;
        (*raiz)->dir = NULL;
    } else {
        int pos = profund % k;
        if (cmp((*raiz)->key, key, pos) < 0) {
            _kdtree_insere(&(*raiz)->dir, key, cmp, profund + 1, k);
        } else {
            _kdtree_insere(&(*raiz)->esq, key, cmp, profund + 1, k);
        }
    }
}

void kdtree_insere(tarv* arv, void* key) {
    _kdtree_insere(&arv->raiz, key, arv->cmp, 0, arv->k);
}

void _kdtree_destroi(tnode* node) {
    if (node != NULL) {
        _kdtree_destroi(node->esq);
        _kdtree_destroi(node->dir);
        free(node->key);
        free(node);
    }
}

void kdtree_destroi(tarv* arv) {
    _kdtree_destroi(arv->raiz);
}

/* 5º - Variável global e funções de interface */
tarv arvore_global;

void inserir_ponto(treg p) {
    treg* novo = aloca_reg(p.embedding, p.person_id);
    kdtree_insere(&arvore_global, novo);
}

void kdtree_construir_global() {
    arvore_global.k = 128;
    arvore_global.dist = distancia;
    arvore_global.cmp = comparador;
    arvore_global.raiz = NULL;
}

void buscar_n_vizinhos_proximos(tarv* arv, treg query, treg* resultados, int n) {
    kdtree_busca_knn(arv, &query, resultados, n);
}

void test_kdtree_facial() {
    kdtree_construir_global();
    
    printf("=== TESTE KD-TREE COM MINHEAP INTEGRADO ===\n\n");
    
    float emb1[128], emb2[128], emb3[128], query_emb[128];
    
    for(int i = 0; i < 128; i++) {
        emb1[i] = 0.1f + i*0.001f;  // Pessoa 1
        emb2[i] = 0.2f + i*0.001f;  // Pessoa 2
        emb3[i] = 0.5f + i*0.002f;  // Pessoa 3
        query_emb[i] = 0.15f + i*0.001f; // Query
    }
    

    treg registros[3];
    
    memcpy(registros[0].embedding, emb1, sizeof(emb1));
    strcpy(registros[0].person_id, "usuario_001");
    
    memcpy(registros[1].embedding, emb2, sizeof(emb2));
    strcpy(registros[1].person_id, "usuario_002");
    
    memcpy(registros[2].embedding, emb3, sizeof(emb3));
    strcpy(registros[2].person_id, "usuario_003");
    
    for(int i = 0; i < 3; i++) {
        inserir_ponto(registros[i]);
    }
    printf("Registros inseridos: 3\n");
    
    treg query;
    memcpy(query.embedding, query_emb, sizeof(query_emb));
    strcpy(query.person_id, "consulta");
    
    int k = 2;
    treg resultados[k];
    buscar_n_vizinhos_proximos(&arvore_global, query, resultados, k);

    printf("\n%d vizinhos mais proximos:\n", k);
    for(int i = 0; i < k; i++) {
        double dist = sqrt(distancia(&query, &resultados[i]));
        printf("%d: %s (distancia: %.4f)\n", i+1, resultados[i].person_id, dist);
    }
    
    kdtree_destroi(&arvore_global);
}

int main() {
    test_kdtree_facial();
    printf("\nTeste concluido com sucesso!\n");
    return 0;
}
