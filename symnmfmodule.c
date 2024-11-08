#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "symnmf.h"
#include <stdlib.h>
#include <stdio.h>

// Prints the entire matrix, rewritten in this file as well for debug purposes.
void print_matrix2(double *matrix, int rows, int cols) {
    int i,j;
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            printf("%f ", matrix[i*cols + j]);
        }
        printf("\n");
    }
}

// Helper to convert Python object to NumPy-like C array
static double* convert_to_c_array(PyObject* array_obj, int* n, int* d) {
    int i,j;

    if (!PyList_Check(array_obj)) {
        printf("An Error Has Occurred\n");
        return NULL;
    }
    *n = PyList_Size(array_obj);
    PyObject* first_row = PyList_GetItem(array_obj, 0);
    if (!PyList_Check(first_row)) {
        printf("An Error Has Occurred\n");
        return NULL;
    }
    *d = PyList_Size(first_row);


    double* array = (double*)malloc((*n) * (*d) * sizeof(double));
    if (array == NULL) {
        printf("An Error Has Occurred\n");
        return NULL;
    }

    for (i = 0; i < *n; i++) {
        PyObject* row = PyList_GetItem(array_obj, i);
        for (j = 0; j < *d; j++) {
            PyObject* item = PyList_GetItem(row, j);
            array[i * (*d) + j] = PyFloat_AsDouble(item);
        }
    }
    return array;
}

// Helper to create Python list from C array
static PyObject* create_python_list_from_c_array(double* array, int n, int d) {
    PyObject* list = PyList_New(n);
    for (int i = 0; i < n; i++) {
        PyObject* row = PyList_New(d);
        for (int j = 0; j < d; j++) {
            PyList_SetItem(row, j, PyFloat_FromDouble(array[i * d + j]));
        }
        PyList_SetItem(list, i, row);
    }
    return list;
}

// Wrapper for sym (Similarity matrix computation)
static PyObject* symnmf_sym(PyObject* self, PyObject* args) {
    PyObject* X_py;
    // Parse the Python arguments (expecting a list of lists)
    if (!PyArg_ParseTuple(args, "O", &X_py)) {
        printf("An Error Has Occurred\n");
        return NULL;
    }
    int n, d;
    double* X = convert_to_c_array(X_py, &n, &d);
    // if (X == NULL) return NULL;

    // Dynamically allocate memory for the similarity matrix A (n x n matrix)
    double* A = (double*)malloc(n * n * sizeof(double));
    if (A == NULL) {
        printf("An Error Has Occurred\n");
        free(X);
        return NULL;
    }
    // Call the sym function (assumed to be implemented elsewhere)
    sym(X, A, n, d);
    // Convert the similarity matrix A back to a Python list
    PyObject* result = create_python_list_from_c_array(A, n, n);
    // Free dynamically allocated memory
    free(X);
    free(A);
    return result;
}

// Wrapper for ddg (Diagonal degree matrix computation)
static PyObject* symnmf_ddg(PyObject* self, PyObject* args) {
    PyObject* A_py;
    // Parse the Python arguments (expecting a list of lists)
    if (!PyArg_ParseTuple(args, "O", &A_py)) {
        printf("An Error Has Occurred\n");
        return NULL;
    }
    int n, dummy;
    double* A = convert_to_c_array(A_py, &n, &dummy);  // Only need n, ignore dummy
    if (A == NULL) return NULL;
    // Dynamically allocate memory for the diagonal degree matrix D (n x n)
    double* D = (double*)malloc(n * n * sizeof(double));
    if (D == NULL) {
        printf("An Error Has Occurred\n");
        free(A);
        return NULL;
    }
    // Call the ddg function (assumed to be implemented elsewhere)
    ddg(A, D, n);
    // Convert the similarity matrix D back to a Python list
    PyObject* result = create_python_list_from_c_array(D, n, n);
    // Free dynamically allocated memory
    free(A);
    free(D);
    return result;
}

// Wrapper for norm (Normalized similarity matrix computation)
static PyObject* symnmf_norm(PyObject* self, PyObject* args) {
    PyObject* A_py, * D_py;

    // Parse the Python arguments (expecting two lists: A and D)
    if (!PyArg_ParseTuple(args, "OO", &A_py, &D_py)) {
        printf("An Error Has Occurred\n");
        return NULL;
    }

    int n, dummy;
    double* A = convert_to_c_array(A_py, &n, &dummy);  // Only need n, ignore dummy
    double* D = convert_to_c_array(D_py, &dummy, &dummy);  // Only need n, ignore dummy
    if (A == NULL || D == NULL) return NULL;

    // Dynamically allocate memory for the normalized similarity matrix W (n x n matrix)
    double* W = (double*)malloc(n * n * sizeof(double));
    if (W == NULL) {
        printf("An Error Has Occurred\n");
        free(A);
        free(D);
        return NULL;
    }

    // Call the norm function (assumed to be implemented elsewhere)
    norm(A, D, W, n);

    // Convert the normalized similarity matrix W back to a Python list
    PyObject* result = create_python_list_from_c_array(W, n, n);

    // Free dynamically allocated memory
    free(A);
    free(D);
    free(W);

    return result;
}

// Wrapper for symnmf (SymNMF matrix factorization)
static PyObject* symnmf_symnmf(PyObject* self, PyObject* args) {
    PyObject* W_py, * H_py;
    int max_iter;
    double epsilon;

    // Parse the Python arguments (expecting two lists and two scalar values)
    if (!PyArg_ParseTuple(args, "OOid", &W_py, &H_py, &max_iter, &epsilon)) {
        printf("An Error Has Occurred");
        return NULL;
    }

    int n, k, dummy;
    double* W = convert_to_c_array(W_py, &n, &dummy);  // Only need n
    double* H = convert_to_c_array(H_py, &dummy, &k);  // Only need k

    if (W == NULL || H == NULL) return NULL;



    // Call the symnmf function (assumed to be implemented elsewhere)
    symnmf(W, H, n, k, max_iter, epsilon);

    // Convert the updated H matrix back to a Python list
    PyObject* result = create_python_list_from_c_array(H, n, k);

    // Free dynamically allocated memory
    free(W);
    free(H);

    return result;
}

// Define method table
static PyMethodDef SymnmfMethods[] = {
    {"sym", symnmf_sym, METH_VARARGS, "Compute similarity matrix"},
    {"ddg", symnmf_ddg, METH_VARARGS, "Compute diagonal degree matrix"},
    {"norm", symnmf_norm, METH_VARARGS, "Compute normalized similarity matrix"},
    {"symnmf", symnmf_symnmf, METH_VARARGS, "Perform SymNMF"},
    {NULL, NULL, 0, NULL}
};

// Module definition structure
static struct PyModuleDef symnmfmodule = {
    PyModuleDef_HEAD_INIT,
    "symnmfmodule",
    NULL,
    -1,
    SymnmfMethods
};

// Module initialization function
PyMODINIT_FUNC PyInit_symnmfmodule(void) {
    PyObject *m;
    m= PyModule_Create(&symnmfmodule);
    if (!m) {
        return NULL;
    }
    return m;
}
