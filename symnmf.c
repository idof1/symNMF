#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "symnmf.h"

FILE* open_file(const char *filename);
int initialize_data_matrices(FILE *file, double **X, double **A, double **D, double **W, int *n, int *d);
void load_input_data(FILE *file, double *X, int n, int d);
void perform_goal(const char *goal, double *X, double *A, double *D, double *W, int n, int d);

/*Prints the entire matrix*/ 
void print_matrix(double *matrix, int rows, int cols) {
    int i, j;
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            printf("%.4f", matrix[i * cols + j]);
            if (j != cols - 1){
                printf(",");
            }
        }
        printf("\n");
    }
}

/* Custom strcmp function to avoid using <string.h> */
int my_strcmp(const char *str1, const char *str2) {
    while (*str1 && (*str1 == *str2)) {
        str1++;
        str2++;
    }
    return *(unsigned char *)str1 - *(unsigned char *)str2;
}

/* Function to count the number of rows (lines) in the file */
int count_rows(FILE *file) {
    int lines = 0;
    int ch;
    while ((ch = fgetc(file)) != EOF) {
        if (ch == '\n') {
            lines++;
        }
    }
    return lines;
}

/* Function to count the number of columns in the first line */
int count_columns(FILE *file) {
    int columns = 0;
    int ch;
    while ((ch = fgetc(file)) != EOF && ch != '\n') {
        if (ch == ',') {
            columns++;
        }
    }
    return columns + 1;  /* Number of columns is one more than the number of commas */
}

/* Function to compute the similarity matrix A */
void sym(double* X, double* A, int n, int d) {
    int i, j, k;
    double dist, diff;

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            if (i == j) {
                A[i * n + j] = 0.0;  /* Diagonals Are 0 */
            } else {
                dist = 0.0;
                for (k = 0; k < d; k++) {
                    diff = X[i * d + k] - X[j * d + k];
                    dist += diff * diff;  /* Compute squared Euclidean distance */
                }
                A[i * n + j] = exp(-dist / 2);
            }
        }
    }
}

/* Function to compute the diagonal degree matrix D from similarity matrix A */
void ddg(double* A, double* D, int n) {
    int i, j;
    double sum;

    /* Initialize D as a zero matrix */
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            D[i * n + j] = 0.0;
        }
    }

    /* Compute the diagonal elements of D */
    for (i = 0; i < n; i++) {
        sum = 0.0;
        for (j = 0; j < n; j++) {
            sum += A[i * n + j];  /* Sum the row for each node i */
        }
        D[i * n + i] = sum;  /* Place sum on the diagonal */
    }
}

/* Function to compute the normalized similarity matrix W = D^(-1/2) * A * D^(-1/2) */
void norm(double* A, double* D, double* W, int n) {
    int i, j;
    double Dii, Djj;

    for (i = 0; i < n; i++) {
        Dii = D[i * n + i];  /* Extract diagonal element */
        for (j = 0; j < n; j++) {
            Djj = D[j * n + j];  /* Extract diagonal element */
            if (Dii > 0 && Djj > 0) {
                W[i * n + j] = A[i * n + j] / sqrt(Dii * Djj);
            } else {
                W[i * n + j] = 0.0;  /* Handle cases where the degree is zero */
            }
        }
    }
}

/* Matrix multiplication: AB = A * B */
void matmul(double* A, double* B, double* AB, int n, int m, int k) {
    int i, j, l;

    for (i = 0; i < n; i++) {
        for (j = 0; j < k; j++) {
            AB[i * k + j] = 0.0;
            for (l = 0; l < m; l++) {
                AB[i * k + j] += A[i * m + l] * B[l * k + j];
            }
        }
    }
}

/* Frobenius norm between A and B */
double frobenious_squared(double* A, double* B, int n, int k) {
    int i, j;
    double sum = 0.0;
    double diff;

    for (i = 0; i < n; i++) {
        for (j = 0; j < k; j++) {
            diff = A[i * k + j] - B[i * k + j];
            sum += diff * diff;
        }
    }
    return sum;
}

/* Calculates the transpose of matrix A into A_T*/
void transpose(double* A, double* A_T, int n, int m) {
    int i, j;
    for (i = 0; i < n; i++) {
        for (j = 0; j < m; j++) {
            A_T[j * n + i] = A[i * m + j];
        }
    }
}

/* Symmetric Non-negative Matrix Factorization - Full algorithm */
void symnmf(double* W, double* H, int n, int k, int max_iter, double epsilon) {
    int i, j, iter;
    double* Ht = (double*)malloc(k * n * sizeof(double)); /* H^T */
    double* WH = (double*)malloc(n * k * sizeof(double));  /* W * H */
    double* HHt = (double*)malloc(n * n * sizeof(double));  /* H * H^T */
    double* HHt_H = (double*)malloc(n * k * sizeof(double));  /* (H * H ^ T) * H */
    double* H_new = (double*)malloc(n * k * sizeof(double));  /* H^(t+1) */
    double beta = 0.5;
    double change;
    /* Check memory allocation errors */
    if (WH == NULL || HHt == NULL || HHt_H == NULL || H_new == NULL) {
        printf("An Error Has Occurred");
        free(WH); free(HHt); free(HHt_H); free(H_new);
        return;
    }
    for (iter = 0; iter < max_iter; iter++) {
        matmul(W, H, WH, n, n, k); /* Step 1: Compute WH = W * H */
        transpose(H, Ht, n, k);
        matmul(H, Ht, HHt, n, k, n); /* Step 2: Compute HHt = H * H^T */
        matmul(HHt, H, HHt_H, n, n, k); /* Step 3: Compute HtH_H = (H^T * H) * H */
        /* Step 4: Update H based on the provided rule - note assumption division cant be 0 */
        for (i = 0; i < n; i++) {
            for (j = 0; j < k; j++) {
                H_new[i * k + j] = H[i * k + j] * (1.0 - beta + beta * (WH[i * k + j] / HHt_H[i * k + j]));
            }
        }
        change = frobenious_squared(H, H_new, n, k); /* Frobenius norm */
        for (i = 0; i < n * k; i++) { /* Copy H_new back to H */
            H[i] = H_new[i];
        }
        if (change < epsilon) { /* Check for convergence */
            break;
        }
    }
    /* Free allocated memory */
    free(WH);
    free(HHt);
    free(HHt_H);
    free(H_new);
}

int main(int argc, char *argv[]) {
    const char *goal; /* Variable declarations at the beginning */
    const char *filename;
    int n, d;
    double *X = NULL;
    double *A = NULL;
    double *D = NULL;
    double *W = NULL;
    FILE *file;

    if (argc < 3) { /* Check command-line arguments */
        printf("An Error Has Occurred");
        return 1;
    }
    
    goal = argv[1];
    filename = argv[2];
    
    file = open_file(filename); /* Open the input file */
    if (file == NULL) {
        return 1;
    }
    
    if (initialize_data_matrices(file, &X, &A, &D, &W, &n, &d) != 0) { /* Initialize data matrices */
        fclose(file);
        return 1;
    }
    
    load_input_data(file, X, n, d); /* Load input data into X */
    fclose(file);
    
    perform_goal(goal, X, A, D, W, n, d); /* Perform the requested operation */

    /* Free allocated memory */
    free(X);
    free(A);
    free(D);
    free(W);
    return 0;
}

FILE* open_file(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        printf("An Error Has Occurred");
    }
    return file;
}

int initialize_data_matrices(FILE *file, double **X, double **A, double **D, double **W, int *n, int *d) {
    *n = count_rows(file); /* Infer matrix dimensions `n` (rows) and `d` (columns) from the file */
    rewind(file);  
    *d = count_columns(file);
    rewind(file);  
    
    *X = (double *)malloc((*n) * (*d) * sizeof(double));  /* Data matrix */
    *A = (double *)malloc((*n) * (*n) * sizeof(double));  /* Similarity matrix */
    *D = (double *)malloc((*n) * (*n) * sizeof(double));  /* Diagonal degree matrix */
    *W = (double *)malloc((*n) * (*n) * sizeof(double));  /* Normalized similarity matrix */
    
    if (*X == NULL || *A == NULL || *D == NULL || *W == NULL) {
        printf("An Error Has Occurred");
        free(*X); free(*A); free(*D); free(*W);
        return 1;
    }
    return 0;
}

void load_input_data(FILE *file, double *X, int n, int d) {
    int i, j;
    for (i = 0; i < n; i++) {
        for (j = 0; j < d; j++) {
            if (fscanf(file, "%lf,", &X[i * d + j]) != 1) {
                printf("An Error Has Occurred");
                free(X);
                exit(1);
            }
        }
    }
}

void perform_goal(const char *goal, double *X, double *A, double *D, double *W, int n, int d) {
    if (my_strcmp(goal, "sym") == 0) {
        sym(X, A, n, d);   /* Compute similarity matrix */
        print_matrix(A, n, n);
    } else if (my_strcmp(goal, "ddg") == 0) {
        sym(X, A, n, d);   /* Compute similarity matrix first */
        ddg(A, D, n);      /* Compute diagonal degree matrix */
        print_matrix(D, n, n);
    } else if (my_strcmp(goal, "norm") == 0) {
        sym(X, A, n, d);   /* Compute similarity matrix first */
        ddg(A, D, n);      /* Compute degree matrix */
        norm(A, D, W, n);  /* Compute normalized similarity matrix */
        print_matrix(W, n, n);
    } else {
        printf("An Error Has Occurred");
        free(X); free(A); free(D); free(W);
        exit(1);
    }
}

