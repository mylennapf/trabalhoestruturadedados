#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <string.h>
#include <assert.h>
#include <math.h>

typedef struct _reg {
    float embedding[128];  // Vetor de 128 floats
    char person_id[100];   // ID da pessoa (100 caracteres)
} treg;

typedef struct {
    double distance;
    treg* record;
} heap_item;

void* aloca_reg(float embedding[128], const char* person_id) {
    treg* reg = malloc(sizeof(treg));
    memcpy(reg->embedding, embedding, sizeof(float) * 128);
    strncpy(reg->person_id, person_id, 99);
    reg->person_id[99] = '\0'; // Garante terminação nula
    return reg;
}

int comparador(void* a, void* b, int pos) {
    float* emb_a = ((treg*)a)->embedding;
    float* emb_b = ((treg*)b)->embedding;
    return (emb_a[pos] > emb_b[pos]) - (emb_a[pos] < emb_b[pos]);
}

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

typedef struct _node {
    void* key;
    struct _node* esq;
    struct _node* dir;
} tnode;

typedef struct _arv {
    tnode* raiz;
    int (*cmp)(void*, void*, int);
    double (*dist)(void*, void*);
    int k;
} tarv;

void kdtree_constroi(tarv* arv, int (*cmp)(void* a, void* b, int), double (*dist)(void*, void*), int k) {
    arv->raiz = NULL;
    arv->cmp = cmp;
    arv->dist = dist;
    arv->k = k;
}

void _kdtree_insere(tnode** raiz, void* key, int (*cmp)(void* a, void* b, int), int profund, int k) {
    if (*raiz == NULL) {
        *raiz = malloc(sizeof(tnode));
        (*raiz)->key = key;
        (*raiz)->esq = NULL;
        (*raiz)->dir = NULL;
    } else {
        int pos = profund % k;
        if (cmp((*(*raiz)).key, key, pos) < 0) {
            _kdtree_insere(&((*(*raiz)).dir), key, cmp, profund + 1, k);
        } else {
            _kdtree_insere(&((*raiz)->esq), key, cmp, profund + 1, k);
        }
    }
}

void kdtree_insere(tarv* arv, void* key) {
    _kdtree_insere(&(arv->raiz), key, arv->cmp, 0, arv->k);
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

void _kdtree_busca_knn(tarv* arv, tnode** atual, void* key, int profund, 
                      heap_item* heap, int* heap_size, int k) {
    if (*atual != NULL) {
        double dist_atual = arv->dist((*atual)->key, key);
        
        // Se o heap não está cheio ou esta distância é menor que a maior no heap
        if (*heap_size < k || dist_atual < heap[0].distance) {
            if (*heap_size == k) {
                // Remove o maior elemento do heap (raiz)
                heap[0] = heap[*heap_size - 1];
                (*heap_size)--;
            }
            
            // Adiciona o novo elemento ao heap
            heap[*heap_size].distance = dist_atual;
            heap[*heap_size].record = (treg*)((*atual)->key);
            (*heap_size)++;
            
            // Reorganiza o heap
            for (int i = (*heap_size - 1) / 2; i >= 0; i--) {
                int largest = i;
                int left = 2 * i + 1;
                int right = 2 * i + 2;
                
                if (left < *heap_size && heap[left].distance > heap[largest].distance)
                    largest = left;
                if (right < *heap_size && heap[right].distance > heap[largest].distance)
                    largest = right;
                
                if (largest != i) {
                    heap_item temp = heap[i];
                    heap[i] = heap[largest];
                    heap[largest] = temp;
                }
            }
        }
        
        int pos = profund % arv->k;
        int comp = arv->cmp(key, (*atual)->key, pos);
        
        tnode** lado_principal = (comp < 0) ? &((*atual)->esq) : &((*atual)->dir);
        tnode** lado_oposto = (comp < 0) ? &((*atual)->dir) : &((*atual)->esq);
        
        _kdtree_busca_knn(arv, lado_principal, key, profund + 1, heap, heap_size, k);
        
        double axis_diff = ((treg*)key)->embedding[pos] - ((treg*)((*atual)->key))->embedding[pos];
        if (*heap_size < k || axis_diff * axis_diff < heap[0].distance) {
            _kdtree_busca_knn(arv, lado_oposto, key, profund + 1, heap, heap_size, k);
        }
    }
}

void kdtree_busca_knn(tarv* arv, void* key, treg* resultados, int k) {
    heap_item* heap = malloc(k * sizeof(heap_item));
    int heap_size = 0;
    
    _kdtree_busca_knn(arv, &(arv->raiz), key, 0, heap, &heap_size, k);
    
    for (int i = heap_size - 1; i > 0; i--) {
        heap_item temp = heap[0];
        heap[0] = heap[i];
        heap[i] = temp;
        
        for (int j = (i - 1) / 2; j >= 0; j--) {
            int largest = j;
            int left = 2 * j + 1;
            int right = 2 * j + 2;
            
            if (left < i && heap[left].distance > heap[largest].distance)
                largest = left;
            if (right < i && heap[right].distance > heap[largest].distance)
                largest = right;
                
            if (largest != j) {
                heap_item temp = heap[j];
                heap[j] = heap[largest];
                heap[largest] = temp;
            }
        }
    }
    
    for (int i = 0; i < heap_size; i++) {
        memcpy(&resultados[i], heap[i].record, sizeof(treg));
    }
    
    free(heap);
}

tarv arvore_global;

tarv* get_tree() {
    return &arvore_global;
}

void inserir_ponto(treg p) {
    treg* novo = malloc(sizeof(treg));
    *novo = p;
    kdtree_insere(&arvore_global, novo);
}

void kdtree_construir() {
    arvore_global.k = 128;  // Agora temos 128 dimensões
    arvore_global.dist = distancia;
    arvore_global.cmp = comparador;
    arvore_global.raiz = NULL;
}

void buscar_n_vizinhos_proximos(tarv* arv, treg query, treg* resultados, int n) {
    kdtree_busca_knn(arv, &query, resultados, n);
}
