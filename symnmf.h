#ifndef SYM_NMF_H
#define SYM_NMF_H

/* === constants (shared) === */
#define MAX_ITER 300
#define EPS 1e-4
#define BETA 0.5
#define TINY 1e-12

/* === Matrix type === */
typedef struct {
    int rows;
    int columns;
    double **data;
} Matrix;

/* === memory === */
Matrix* allocate_matrix(int rows, int columns);
void free_matrix(Matrix* mat);

/* === file & input === */
Matrix* load_matrix_from_file(const char* filename);
void print_matrix(const Matrix* mat);

/* === c path === */
Matrix* create_similarity_matrix(const Matrix* mat);
Matrix* create_diagonal_degree_matrix(const Matrix* mat);
Matrix* create_normalized_similarity_matrix(const Matrix* A, const Matrix* D);

/* === symnmf main logic (python passes W and initial H) === */
int update_H_mat(const Matrix* W, const Matrix* H_in, Matrix* H_out);
int symnmf_c(const Matrix* W, const Matrix* H_init, Matrix** H_out);

double compute_frobenius_diff(const Matrix* A, const Matrix* B);
int has_converged(const Matrix* H_old, const Matrix* H_new, double eps);

/* === utilities === */
Matrix* matrix_multiply(const Matrix* A, const Matrix* B);
Matrix* matrix_transpose(const Matrix* A);
Matrix* normalize_rows(Matrix* mat);

#endif /* SYM_NMF_H */