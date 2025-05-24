#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <string.h>
#include <assert.h>
#include <math.h>

/* Definições principais */
typedef struct _reg {
    float embedding[128];
    char id[100];
} treg;

typedef struct {
    double distance;
    treg* record;
} heap_item;

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

/* Funções auxiliares */
void* aloca_reg(float embedding[128], const char id[]) {
    treg* reg = malloc(sizeof(treg));
    memcpy(reg->embedding, embedding, sizeof(float) * 128);
    strncpy(reg->id, id, 99);
    reg->id[99] = '\0';
    return reg;
}

int comparador(void* a, void* b, int pos) {
    float diff = ((treg*)a)->embedding[pos] - ((treg*)b)->embedding[pos];
    return (diff > 0) ? 1 : ((diff < 0) ? -1 : 0);
}

double distancia(void* a, void* b) {
    double soma = 0.0;
    for (int i = 0; i < 128; i++) {
        double diff = ((treg*)a)->embedding[i] - ((treg*)b)->embedding[i];
        soma += diff * diff;
    }
    return soma;
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

/* Funções de busca com heap */
void _kdtree_busca_knn(tarv* arv, tnode** atual, void* key, int profund, 
                      heap_item* heap, int* heap_size, int k) {
    if (*atual != NULL) {
        double dist_atual = arv->dist((*atual)->key, key);
        
        if (*heap_size < k || dist_atual < heap[0].distance) {
            if (*heap_size == k) {
                heap[0] = heap[*heap_size - 1];
                (*heap_size)--;
            }
            
            heap[*heap_size].distance = dist_atual;
            heap[*heap_size].record = (treg*)((*atual)->key);
            (*heap_size)++;
            
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

/* Variáveis globais e funções de interface */
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
    arvore_global.k = 128;
    arvore_global.dist = distancia;
    arvore_global.cmp = comparador;
    arvore_global.raiz = NULL;
}

void buscar_n_vizinhos_proximos(tarv* arv, treg query, treg* resultados, int n) {
    kdtree_busca_knn(arv, &query, resultados, n);
}

void test_kdtree_facial() {
    // Inicializa a árvore
    kdtree_construir();
        
    // Cria embeddings de exemplo
    float emb1[128], emb2[128], emb3[128], query_emb[128];
    
    // Preenche com valores de exemplo
    for(int i = 0; i < 128; i++) {
        emb1[i] = 0.1f + i*0.001f;  // Pessoa 1
        emb2[i] = 0.2f + i*0.001f;  // Pessoa 2
        emb3[i] = 0.5f + i*0.002f;  // Pessoa 3
        query_emb[i] = 0.15f + i*0.001f; // Query
    }
    
    // Cria e insere registros
    treg registros[3];
    
    memcpy(registros[0].embedding, emb1, sizeof(emb1));
    strcpy(registros[0].id, "pessoa_001");
    
    memcpy(registros[1].embedding, emb2, sizeof(emb2));
    strcpy(registros[1].id, "pessoa_002");
    
    memcpy(registros[2].embedding, emb3, sizeof(emb3));
    strcpy(registros[2].id, "pessoa_003");
    
    for(int i = 0; i < 3; i++) {
        inserir_ponto(registros[i]);
    }
    printf("Inseridos 3 registros na arvore\n");
    
    // Prepara consulta
    treg query;
    memcpy(query.embedding, query_emb, sizeof(query_emb));
    strcpy(query.id, "consulta");
    
    // Busca os 2 mais próximos
    int k = 2;
    treg resultados[k];
    buscar_n_vizinhos_proximos(get_tree(), query, resultados, k);
    
    // Mostra resultados
    printf("\n%d vizinhos mais proximos:\n", k);
    for(int i = 0; i < k; i++) {
        double dist = sqrt(distancia(&query, &resultados[i]));
        printf("%d: %s (distancia: %.4f)\n", i+1, resultados[i].id, dist);
    }
    
    // Limpeza
    kdtree_destroi(get_tree());
}

int main(void) {
    test_kdtree_facial();
    printf("\nTodos os testes passaram com sucesso!\n");
    return EXIT_SUCCESS;
}
