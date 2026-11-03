#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <omp.h>

// Ajuste das constantes para o dataset Shuttle (9 atributos, 7 classes)
#define NUM_FEATURES 9
#define NUM_CLASSES 7

typedef struct {
    double features[NUM_FEATURES];
    int label;
} Point;

typedef struct {
    int c1;
    int c2;
} ClusterPair;

// Distância Euclidiana entre dois pontos
static inline double euclidean_distance(const Point* a, const Point* b) {
    double sum = 0.0;
    for (int i = 0; i < NUM_FEATURES; ++i) {
        double diff = a->features[i] - b->features[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

// Leitura dinâmica do dataset shuttle.tst
Point* load_full_dataset(const char* filepath, size_t* dataset_size) {
    FILE* file = fopen(filepath, "r");
    if (!file) {
        fprintf(stderr, "Erro ao abrir o arquivo: %s\n", filepath);
        *dataset_size = 0;
        return NULL;
    }

    size_t capacity = 1000;
    size_t count = 0;
    Point* dataset = (Point*)malloc(capacity * sizeof(Point));

    char line[1024];
    while (fgets(line, sizeof(line), file)) {
        // Trata separadores por vírgula convertendo para espaço
        for (int i = 0; line[i] != '\0'; i++) {
            if (line[i] == ',') line[i] = ' ';
        }

        Point p;
        char* ptr = line;
        int n_read = 0;
        int valid = 1;

        // Leitura dos 9 atributos do Shuttle
        for (int i = 0; i < NUM_FEATURES; i++) {
            if (sscanf(ptr, "%lf%n", &p.features[i], &n_read) == 1) {
                ptr += n_read;
            } else {
                valid = 0;
                break;
            }
        }

        // Leitura da classe (1 a 7) e conversão para base 0 (0 a 6)
        int raw_label;
        if (valid && sscanf(ptr, "%d", &raw_label) == 1) {
            p.label = raw_label - 1; // Ajusta de 1..7 para 0..6

            if (count >= capacity) {
                capacity *= 2;
                Point* temp = (Point*)realloc(dataset, capacity * sizeof(Point));
                if (!temp) {
                    fprintf(stderr, "Erro na realocacao de memoria.\n");
                    free(dataset);
                    fclose(file);
                    *dataset_size = 0;
                    return NULL;
                }
                dataset = temp;
            }
            dataset[count++] = p;
        }
    }

    fclose(file);
    *dataset_size = count;
    return dataset;
}

// Cálculo do Índice de Dunn com OpenMP
double compute_dunn_index_openmp(const Point* dataset, size_t dataset_size, int num_classes) {
    // 1. Contagem de pontos por cluster
    int* cluster_sizes = (int*)calloc(num_classes, sizeof(int));
    for (size_t i = 0; i < dataset_size; ++i) {
        if (dataset[i].label >= 0 && dataset[i].label < num_classes) {
            cluster_sizes[dataset[i].label]++;
        }
    }

    // Alocação das listas de índices por classe
    int** clusters = (int**)malloc(num_classes * sizeof(int*));
    for (int c = 0; c < num_classes; ++c) {
        clusters[c] = (int*)malloc(cluster_sizes[c] * sizeof(int));
    }

    int* cluster_counts = (int*)calloc(num_classes, sizeof(int));
    for (size_t i = 0; i < dataset_size; ++i) {
        int lbl = dataset[i].label;
        if (lbl >= 0 && lbl < num_classes) {
            clusters[lbl][cluster_counts[lbl]++] = (int)i;
        }
    }

    // 2. Diâmetro Máximo Intra-cluster
    double max_intra_diameter = 0.0;

    #pragma omp parallel for reduction(max:max_intra_diameter) schedule(dynamic)
    for (int c = 0; c < num_classes; ++c) {
        int n_points = cluster_sizes[c];
        double local_max_dist = 0.0;

        for (int i = 0; i < n_points; ++i) {
            for (int j = i + 1; j < n_points; ++j) {
                double dist = euclidean_distance(&dataset[clusters[c][i]], &dataset[clusters[c][j]]);
                if (dist > local_max_dist) {
                    local_max_dist = dist;
                }
            }
        }
        if (local_max_dist > max_intra_diameter) {
            max_intra_diameter = local_max_dist;
        }
    }

    // 3. Distância Mínima Inter-cluster
    double min_inter_distance = DBL_MAX;

    int total_pairs = (num_classes * (num_classes - 1)) / 2;
    ClusterPair* pairs = (ClusterPair*)malloc(total_pairs * sizeof(ClusterPair));
    int pair_idx = 0;
    for (int c1 = 0; c1 < num_classes; ++c1) {
        for (int c2 = c1 + 1; c2 < num_classes; ++c2) {
            pairs[pair_idx].c1 = c1;
            pairs[pair_idx].c2 = c2;
            pair_idx++;
        }
    }

    #pragma omp parallel for reduction(min:min_inter_distance) schedule(dynamic)
    for (int p = 0; p < total_pairs; ++p) {
        int c1 = pairs[p].c1;
        int c2 = pairs[p].c2;

        int size1 = cluster_sizes[c1];
        int size2 = cluster_sizes[c2];

        // Se uma das classes não tiver elementos, ignora o par
        if (size1 == 0 || size2 == 0) continue;

        double local_min_dist = DBL_MAX;

        for (int i = 0; i < size1; ++i) {
            for (int j = 0; j < size2; ++j) {
                double dist = euclidean_distance(&dataset[clusters[c1][i]], &dataset[clusters[c2][j]]);
                if (dist < local_min_dist) {
                    local_min_dist = dist;
                }
            }
        }

        if (local_min_dist < min_inter_distance) {
            min_inter_distance = local_min_dist;
        }
    }

    // Liberação de memória
    for (int c = 0; c < num_classes; ++c) {
        free(clusters[c]);
    }
    free(clusters);
    free(cluster_sizes);
    free(cluster_counts);
    free(pairs);

    if (max_intra_diameter == 0.0) return 0.0;
    return min_inter_distance / max_intra_diameter;
}

int main(int argc, char* argv[]) {
    const char* filepath = (argc > 1) ? argv[1] : "shuttle.tst";

    printf("Carregando todo o dataset de: %s...\n", filepath);
    size_t dataset_size = 0;
    Point* dataset = load_full_dataset(filepath, &dataset_size);

    if (!dataset || dataset_size == 0) {
        fprintf(stderr, "Nenhum dado foi carregado. Verifique o caminho do arquivo.\n");
        return 1;
    }

    printf("Total de instancias carregadas: %zu\n", dataset_size);

    double start_time = omp_get_wtime();
    double dunn_index = compute_dunn_index_openmp(dataset, dataset_size, NUM_CLASSES);
    double end_time = omp_get_wtime();

    printf("Indice de Dunn: %lf\n", dunn_index);
    printf("Tempo de execucao: %lf segundos.\n", end_time - start_time);

    free(dataset);
    return 0;
}