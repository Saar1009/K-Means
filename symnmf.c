/* === imports === */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

/* === constants === */
#define MAX_ITER 300
#define EPS 1e-4
#define BETA 0.5
#define TINY 1e-12


/* === matrix === */
typedef struct {
    int rows;
    int columns;
    double** data;
}Matrix;

/* === functions declarations === */
/* memory */
Matrix* allocate_matrix(int rows, int columns);
void free_matrix(Matrix* mat);

/* file & input */
Matrix* load_matrix_from_file(const char* filename); 
void print_matrix(const Matrix* mat); /* For Debugging */

/* c path */
Matrix* create_similarity_matrix(const Matrix* mat);
Matrix* create_diagonal_degree_matrix(const Matrix* mat);
Matrix* create_normalized_similarity_matrix(const Matrix* A, const Matrix* D);

/* symnmf main logic (python passes W and initial H) */
int update_H_mat(const Matrix* W, const Matrix* H_in, Matrix* H_out);
int symnmf_c(const Matrix* W, const Matrix* H_init, Matrix** H_out);
double compute_frobenius_diff(const  Matrix* A, const Matrix* B);
int has_converged(const Matrix* H_old, const Matrix* H_new, double eps);


/* utilities */
Matrix* matrix_multiply(const Matrix* A, const Matrix* B);
Matrix* matrix_transpose(const Matrix* A);
Matrix* normalize_rows(Matrix* mat);

/* === main === */

int main(int argc, char * argv[])
{
    const char *goal;
    const char *filename;
    Matrix *X; /* input data */
    Matrix *S; /* similarity */
    Matrix *D; /* diagonal degree */
    Matrix *N; /* normalized similarity */
    if (argc != 3) {
        fprintf(stderr, "Invalid Input!\n");
        return 1;
    }
    goal = argv[1];
    filename = argv[2];
    X = load_matrix_from_file(filename);
    if (!X) {
        fprintf(stderr, "An Error Has Occurred\n");
       return 1;
    }

    if (strcmp(goal, "sym") == 0) {
        S = create_similarity_matrix(X);
        if (!S) { free_matrix(X); fprintf(stderr, "An Error Has Occurred\n"); return 1; }
        print_matrix(S);
        free_matrix(S);
        free_matrix(X);
        return 0;
    }
    else if (strcmp(goal, "ddg") == 0) {
        S = create_similarity_matrix(X);
        if (!S) { free_matrix(X); fprintf(stderr, "An Error Has Occurred\n"); return 1; }
        D = create_diagonal_degree_matrix(S);
        if (!D) { free_matrix(S); free_matrix(X); fprintf(stderr, "An Error Has Occurred\n"); return 1; }
        print_matrix(D);
        free_matrix(D);
        free_matrix(S);
        free_matrix(X);
        return 0;
    }
    else if (strcmp(goal, "norm") == 0) {
        S = create_similarity_matrix(X);
        if (!S) { free_matrix(X); fprintf(stderr, "An Error Has Occurred\n"); return 1; }
        D = create_diagonal_degree_matrix(S);
        if (!D) { free_matrix(S); free_matrix(X); fprintf(stderr, "An Error Has Occurred\n"); return 1; }
        N = create_normalized_similarity_matrix(S, D);
        if (!N) { free_matrix(D); free_matrix(S); free_matrix(X); fprintf(stderr, "An Error Has Occurred\n"); return 1; }
        print_matrix(N);
        free_matrix(N);
        free_matrix(D);
        free_matrix(S);
        free_matrix(X);
        return 0;
    }
    else {
        /* Unsupported goal in C phase */
        free_matrix(X);
        fprintf(stderr, "Invalid Input!\n");
        return 1;
    }
}
Matrix * allocate_matrix(int rows, int columns) /* allocates the matrix */
{
    /* declarations */
    int i, j, k;
    /* sanity checks */
    Matrix * mat = malloc(sizeof(Matrix));
    assert(mat != NULL);
    /* allocate_matrix */
    mat -> columns = columns;
    mat -> rows = rows;
    mat -> data = malloc(rows * sizeof(double*));
    if (mat -> data == NULL) {
        free(mat);
        return NULL;
    }
    for (i = 0; i < rows; i++){
        mat -> data[i] = malloc(columns * sizeof(double));
        if (mat->data[i] == NULL) {
            for (j = 0; j < i; j++) {
                free(mat->data[j]);
            }
            free(mat->data);
            free(mat);
            return NULL;
        }
        /* Initialize allocated memory to zero */
        for (k = 0; k < columns; k++) {
            mat->data[i][k] = 0.0;
        }
    } 
    return mat;  
}
void free_matrix(Matrix * mat) /* free the matrix */
{
    /* declarations */
    int i;
    /* sanity checks */
    if (!mat) return;
    /* free_matrix */
    if (mat->data) {
        for (i = 0; i < mat->rows; i++) {
            free(mat->data[i]); /* free(NULL) is safe */
        }
        free(mat->data);
    }
    free(mat);
}

Matrix * load_matrix_from_file(const char* filename) /* loads matrix from the file */
{
    FILE* file;
    char line[1024];
    char* tok;
    char* endp;
    int rows, columns, count, i, j;
    Matrix* mat;
    double v;
    file = fopen(filename, "r");
    if (file == NULL) return NULL;
    /* Pass 1: infer columns, count rows (CSV only) */
    rows = 0; columns = -1;
    while (fgets(line, sizeof(line), file)) {
        count = 0;
        tok = strtok(line, ",\r\n");
        while (tok != NULL) { count++; tok = strtok(NULL, ",\r\n"); }
        if (count == 0) continue; /* skip blank lines */
        if (columns < 0) columns = count;
        else if (count != columns) { fclose(file); fprintf(stderr, "Error reading matrix value at [%d][%d]\n", rows, 0); return NULL; }
        rows++;
    }
    if (rows <= 0 || columns <= 0) { fclose(file); return NULL; }
    rewind(file);
    mat = allocate_matrix(rows, columns);
    if (!mat) { fclose(file); return NULL; }
    /* Pass 2: read values */
    i = 0;
    while (i < rows) {
        if (!fgets(line, sizeof(line), file)) { free_matrix(mat); fclose(file); fprintf(stderr, "Error reading matrix value at [%d][%d]\n", i, 0); return NULL; }
        tok = strtok(line, ",\r\n");
        if (tok == NULL) continue; /* skip blank lines */
        j = 0;
        while (tok != NULL && j < columns) {
            v = strtod(tok, &endp);
            /* no chars consumed OR trailing junk => error */
            if (endp == tok || *endp != '\0') {
                free_matrix(mat);
                fclose(file);
                fprintf(stderr, "Error reading matrix value at [%d][%d]\n", i, j);
                return NULL;
            }
            mat->data[i][j] = v;
            j++;
            tok = strtok(NULL, ",\r\n");
        }
        if (j != columns) { free_matrix(mat); fclose(file); fprintf(stderr, "Error reading matrix value at [%d][%d]\n", i, j); return NULL; }
        i++;
    }
    fclose(file);
    return mat;
}
void print_matrix(const Matrix * mat) /* print matrix (for debugging) */
{
    int i, j;
    if (!mat) return;
    for (i = 0; i < mat->rows; i++) {
        for (j = 0; j < mat->columns; j++) {
            if (j > 0) printf(",");
            /* normalize -0.0000 to 0.0000 */
            if (fabs(mat->data[i][j]) < 0.00005) {
                printf("%.4f", 0.0);
            } else {
                printf("%.4f", mat->data[i][j]);
            }
        }
        printf("\n");
    }
}

Matrix * create_similarity_matrix(const Matrix * mat) /* creates the similarity matrix according to demands */
{
    /* declarations */
    int i, j, d;
    double euclidian_sum = 0;
    double dummy_euclidian_sum = 0;
    /* sanity checks */
    Matrix * sim = allocate_matrix(mat -> rows, mat -> rows);
    if (!sim) return NULL;
    /* create_similarity_matrix */
    for (i = 0; i < sim -> rows; i++)
    {
        for (j = 0; j < sim -> rows; j++)
        {
            euclidian_sum = 0;
            dummy_euclidian_sum = 0;
            if (i == j)/* even though the euclidian distance will be the same, we can reduce time complexity in best case*/
            {
                sim -> data[i][i] = 0; /* i = j*/
            }
            
            else
            {
                for (d = 0; d < mat -> columns; d++)
                {
                    dummy_euclidian_sum = mat -> data[i][d] - mat -> data[j][d];
                    euclidian_sum -= ((dummy_euclidian_sum * dummy_euclidian_sum )/2);
                }
                sim -> data[i][j] = exp(euclidian_sum);
            }
        }
    }
    return sim;
}
Matrix * create_diagonal_degree_matrix(const Matrix * mat) /* sums up each row to the diagonal */
{
    /* declarations */
    int i, j;
    double row_sum = 0;
    /* sanity checks */
    Matrix * dig = allocate_matrix(mat -> rows, mat -> rows);
    if (!dig) return NULL;
    /* create_diagonal_degree_matrix */
    for (i = 0; i < mat -> rows; i++)
    {
        row_sum = 0;
        for (j = 0; j < mat -> columns; j++)
        {
            row_sum = row_sum + mat -> data[i][j];
        }
        dig -> data[i][i] = row_sum;
    }
    
    return dig;
}
Matrix * create_normalized_similarity_matrix(const Matrix * A, const Matrix * D) /* creates the normalized similarity matrix in the most efficient way */
{
    /* declarations*/
    int i, a, b;
    Matrix *norm, *D_modify;
    if (!A || !D) return NULL;
    /* sanity checks */
    if (A->rows != D->rows) return NULL;
    norm = allocate_matrix(A->rows, A->rows);
    if (!norm) return NULL;
    D_modify = allocate_matrix(D->rows, D->rows);
    if (!D_modify) { free_matrix(norm); return NULL; }
    /* create_normalized_similarity_matrix */
    for (i = 0; i < A -> rows; i++)/* Calculates D^-0.5 */
    {
        if (D -> data[i][i] != 0)/* prevents calculation in 0 */
        {
            D_modify -> data[i][i] = 1.0 / sqrt(D -> data[i][i]);
        }
        else
        {
            D_modify -> data[i][i] = 0;
        }
    }

    for (a = 0; a < A -> rows; a++) /* O(n^2) algorithm :) */
    {
        for (b = 0; b < A -> rows; b++)
        {
            norm -> data[a][b] = A -> data[a][b] * D_modify -> data[a][a] * D_modify -> data[b][b];
        }
    }
    free_matrix(D_modify);
    return norm;
}

/* ---- symnmf core: one multiplicative update step ---- */
int update_H_mat(const Matrix* W, const Matrix* H_in, Matrix* H_out)
{
    /* declarations */
    Matrix *NUM;        /* W * H_in      (N x k) */
    Matrix *H_T;        /* H_in^T        (k x N) */
    Matrix *DEN_left;   /* H_in * H_in^T (N x N) */
    Matrix *DEN;        /* (H_in H_in^T) H_in  (N x k) */
    int i, j;
    double den, frac, val;
    /* sanity checks */
    if (!W || !H_in || !H_out) return 1;
    if (W->rows != W->columns) return 1;
    if (W->rows != H_in->rows) return 1;
    if (H_out->rows != H_in->rows || H_out->columns != H_in->columns) return 1;
    NUM = matrix_multiply(W, H_in);
    if (!NUM) return 1;
    H_T = matrix_transpose(H_in);
    if (!H_T) { free_matrix(NUM); return 1; }
    DEN_left = matrix_multiply(H_in, H_T);
    if (!DEN_left) { free_matrix(H_T); free_matrix(NUM); return 1; }
    DEN = matrix_multiply(DEN_left, H_in);
    if (!DEN) { free_matrix(DEN_left); free_matrix(H_T); free_matrix(NUM); return 1; }
    /* update_H_mat */
    for (i = 0; i < H_out->rows; i++) {
        for (j = 0; j < H_out->columns; j++) {
            den  = DEN->data[i][j];
            if (den == 0){den = TINY; }
            frac = NUM->data[i][j] / den;
            val  = H_in->data[i][j] * ((1.0 - BETA) + (BETA * frac));
            if (val < 0.0) val = 0.0;
            H_out->data[i][j] = val;
        }
    }
    free_matrix(DEN);
    free_matrix(DEN_left);
    free_matrix(H_T);
    free_matrix(NUM);
    return 0;
}
double compute_frobenius_diff(const  Matrix* A, const Matrix* B) /* computes the forbenius different between 2 matrices */
{
    /* declarations */
    int i, j;
    double d, sum;
    /* sanity checks */
    if (!A || !B) return INFINITY;
    if (A->rows != B->rows || A->columns != B->columns) return INFINITY;
    /* compute_frobenius_diff */
    sum = 0;
    for (i = 0; i < A -> rows; i++)
    {
        for (j = 0; j < A -> columns; j++)
        {
            d = A -> data[i][j] - B -> data[i][j];
            sum += d * d;
        }   
    }
    return sqrt(sum);    
}
int has_converged(const Matrix* H_old, const Matrix* H_new, double eps) /* checks wheter the forbenius difference is lower than epsilon */
{
    /* declarations */
    double delta;
    /* sanity checks */
    if (!H_old || !H_new) return 0;
    if (H_old->rows != H_new->rows || H_old->columns != H_new->columns) return 0;
    /* has_converged */
    delta = compute_frobenius_diff(H_new, H_old);
    return ((delta * delta) <= eps) ? 1 : 0;
}
/* ---- symnmf driver: iterate updates until convergence or till max iters ---- */
int symnmf_c(const Matrix* W, const Matrix* H_init, Matrix** H_out) /* the symnmf algorithm */
{
    /* declarations */
    Matrix *H_prev;
    Matrix *H_next;
    Matrix prev_view, next_view;
    int iters, i, j;
    int rc;
    /* sanity checks */
    if (!W || !H_init || !H_out) return 1;
    if (W->rows != W->columns) return 1;
    if (W->rows != H_init->rows) return 1;
    H_prev = allocate_matrix(H_init->rows, H_init->columns);
    if (!H_prev) return 1;
    H_next = allocate_matrix(H_init->rows, H_init->columns);
    if (!H_next) { free_matrix(H_prev); return 1; }
    /* symnmf_c */
    for (i = 0; i < H_init->rows; i++) {
        for (j = 0; j < H_init->columns; j++) {
            H_prev->data[i][j] = H_init->data[i][j];
        }
    }
    /* convergence */
    iters = 0;
    while (iters < MAX_ITER) {
        rc = update_H_mat(W, H_prev, H_next);
        if (rc != 0) { free_matrix(H_next); free_matrix(H_prev); return 1; }
        prev_view.rows = H_prev->rows; prev_view.columns = H_prev->columns; prev_view.data = H_prev->data;
        next_view.rows = H_next->rows; next_view.columns = H_next->columns; next_view.data = H_next->data;
        if (has_converged(&prev_view, &next_view, EPS)) {break;}
        {
            Matrix *tmp;
            tmp = H_prev;
            H_prev = H_next;
            H_next = tmp;
        }
        iters++;
    }
    /* h allocation */
    *H_out = allocate_matrix(H_next->rows, H_next->columns);
    if (!(*H_out)) { free_matrix(H_next); free_matrix(H_prev); return 1; }
    for (i = 0; i < H_next->rows; i++) {
        for (j = 0; j < H_next->columns; j++) {
            (*H_out)->data[i][j] = H_next->data[i][j];
        }
    }
    free_matrix(H_next);
    free_matrix(H_prev);
    return 0;
}

Matrix* matrix_multiply(const Matrix* A, const Matrix* B) /* this method perform a basic matrix multiply, with size check*/
{
    /* declarations*/
    int i,j,k;
    double sum;
    Matrix* C; /* C = A X B*/
    /* sanity checks */
    if (!A || !B) return NULL;
    if (A->columns != B->rows) return NULL;
    /* matrix multiply */
    C = allocate_matrix(A -> rows, B -> columns);
    if (!C) return NULL;
    for (i = 0; i < C -> rows; i++)
    {
        for (j = 0; j < C -> columns; j++)
        {
            sum = 0;
            for (k = 0; k < A->columns; k++)
            {
                sum += A -> data[i][k] * B -> data[k][j];
            }
            C -> data[i][j] = sum;
        }     
    }
    return C;
}
Matrix* matrix_transpose(const Matrix* A) /* this method performs a basic matrix transpose*/
{
    /* declarations*/
    Matrix *AT; /* AT = A^t*/
    int i, j;
    /* sanity checks */
    if (!A) return NULL;
    /* matrix transpose */
    AT = allocate_matrix(A -> columns, A -> rows);
    if (!AT) return NULL;
    for (i = 0; i < A -> rows; i++)
    {
        for (j = 0; j < A -> columns; j++)
        {
            AT -> data[j][i] = A -> data[i][j];
        }
    }
    
    return AT;
}
Matrix* normalize_rows(Matrix* mat) /* returns the normalized matrix of a given matrix */
{
    /* declarations*/
    Matrix *norm; /* norm = normalized Matrix */
    int i, j;
    double sum;
    /* sanity checks */
    if (!mat) return NULL;
    /* matrix transpose */
    norm = allocate_matrix(mat -> rows, mat -> columns);
    if (!norm) return NULL;
    for (i = 0; i < mat -> rows; i++)
    {
        sum = 0;
        for (j = 0; j < mat -> columns; j++)
        {
            sum += mat -> data[i][j];
        }
        if (sum == 0) /* only non negative values, therfore it implies that the whole line is 0*/
        {
            for (j = 0; j < mat -> columns; j++)
            {
                norm -> data [i][j] = 0; 
            }
        }
        else
        {
            for (j = 0; j < mat -> columns; j++)
            {
                norm -> data [i][j] = mat -> data[i][j] / sum;
            }
        }
    }    
    return norm;
}
