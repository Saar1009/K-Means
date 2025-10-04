/* === python-c API wrapper for symnmf === */
#define PY_SSIZE_T_CLEAN
#include "symnmf.h"
#include <Python.h>
#include <stdlib.h>

/* ---- usable methods ---- */
static int dims_from_pylist(PyObject *obj, Py_ssize_t *rows, Py_ssize_t *cols) /* helper that validates that a given object is a list of lists */
{
    /* declarations */
    PyObject *row0;
    /* dims_from_pylist */
    if (!PyList_Check(obj)) 
    { 
        PyErr_SetString(PyExc_TypeError, "expected list of rows"); 
        return 0; 
    }
    *rows = PyList_Size(obj); 
    if (*rows <= 0) 
    { 
        PyErr_SetString(PyExc_ValueError, "matrix needs rows"); 
        return 0; 
    }
    row0 = PyList_GetItem(obj, 0); 
    if (!PyList_Check(row0)) 
    { 
        PyErr_SetString(PyExc_TypeError, "each row must be list"); 
        return 0; 
    }
    *cols = PyList_Size(row0); 
    if (*cols <= 0) 
    { 
        PyErr_SetString(PyExc_ValueError, "matrix needs columns"); 
        return 0; 
    }
    return 1;
}

static Matrix* matrix_alloc(int r, int c) /* allocates a matrix */
{
    /* declarations */
    int i; 
    Matrix *m = (Matrix*)malloc(sizeof(Matrix));
    /* sanity checks */
    if (!m) { PyErr_NoMemory(); return NULL; }
    /* matrix_alloc */
    m->rows = r; 
    m->columns = c; 
    m->data = (double**)malloc(r * sizeof(double*));
    if (!m->data) 
    { 
        free(m); 
        PyErr_NoMemory(); 
        return NULL; 
    }
    for (i = 0; i < r; ++i) {
        m->data[i] = (double*)malloc(c * sizeof(double));
        if (!m->data[i]) {
            int k; for (k = 0; k < i; ++k) free(m->data[k]); 
            free(m->data); 
            free(m);
            PyErr_NoMemory(); 
            return NULL;
        }
    }
    return m;
}

static PyObject* matrix_to_pylist(const Matrix *m)/* converts a c matrix into python list[list[float]] */
{
    /* declarations */
    int i, j; 
    PyObject *out, *row, *val;
    /* sanity checks */
    if (!m) { Py_RETURN_NONE; }
    out = PyList_New(m->rows); 
    if (!out) return NULL;
    /* matrix_to_pykist*/
    for (i = 0; i < m->rows; ++i) {
        row = PyList_New(m->columns); 
        if (!row) 
        { 
            Py_DECREF(out); 
            return NULL; 
        }
        for (j = 0; j < m->columns; ++j) {
            val = PyFloat_FromDouble(m->data[i][j]); 
            if (!val) 
            { 
                Py_DECREF(row); 
                Py_DECREF(out); 
                return NULL; 
            }
            PyList_SET_ITEM(row, j, val);
        }
        PyList_SET_ITEM(out, i, row);
    }
    return out;
}

static Matrix* pylist_to_matrix(PyObject *obj) /* converts python list[list[float]] into a c matrix */
{
    /* declarations */
    Py_ssize_t rows, cols; 
    int i, j; 
    Matrix *m; 
    PyObject *row, *cell; 
    double v;
    /* sanity checks */
    if (!dims_from_pylist(obj, &rows, &cols)) return NULL;
    m = matrix_alloc((int)rows, (int)cols); if (!m) return NULL;
    /* pylist_to_matrix */
    for (i = 0; i < (int)rows; ++i) {
        row = PyList_GetItem(obj, i); 
        if (!PyList_Check(row) || PyList_Size(row) != cols) 
        { 
            free_matrix(m); 
            PyErr_SetString(PyExc_ValueError, "ragged rows"); 
            return NULL; 
        }
        for (j = 0; j < (int)cols; ++j) {
            cell = PyList_GetItem(row, j); v = PyFloat_AsDouble(cell);
            if (PyErr_Occurred()) 
            { 
                free_matrix(m); return NULL; 
            }
            m->data[i][j] = v;
        }
    }
    return m;
}

/* ---- c path ---- */
static PyObject* py_sym(PyObject *self, PyObject *args) /* create_similarity_matrix wrapper */
{
    /* declarations */
    const char *filename; 
    Matrix *X = NULL, *S = NULL; 
    PyObject *out = NULL;
    /* sanity checks */
    if (!PyArg_ParseTuple(args, "s", &filename)) return NULL;
    /* py_sym */
    X = load_matrix_from_file(filename); 
    if (!X) /* sanity check */
    { 
        PyErr_SetString(PyExc_RuntimeError, "failed to load file"); 
        return NULL; 
    }
    S = create_similarity_matrix(X); 
    free_matrix(X);
    if (!S) /* sanity check */
    { PyErr_SetString(PyExc_RuntimeError, "failed to build S"); 
        return NULL; 
    }
    out = matrix_to_pylist(S); free_matrix(S);
    return out;
}

static PyObject* py_ddg(PyObject *self, PyObject *args) /* create_diagonal_degree_matrix wrapper */
{
    /* declarations */
    const char *filename; 
    Matrix *X = NULL, *S = NULL, *D = NULL; 
    PyObject *out = NULL;
    /* sanity checks */
    if (!PyArg_ParseTuple(args, "s", &filename)) return NULL;
    /* py_ddg */
    X = load_matrix_from_file(filename); 
    if (!X) /* sanity check */
    { 
        PyErr_SetString(PyExc_RuntimeError, "failed to load file"); 
        return NULL; 
    }
    S = create_similarity_matrix(X); 
    free_matrix(X);
    if (!S) /* sanity check */
    { 
        PyErr_SetString(PyExc_RuntimeError, "failed to build S"); 
        return NULL; 
    }
    D = create_diagonal_degree_matrix(S); 
    free_matrix(S);
    if (!D) /* sanity check */
    { 
        PyErr_SetString(PyExc_RuntimeError, "failed to build D"); 
        return NULL; 
    }
    out = matrix_to_pylist(D); 
    free_matrix(D);
    return out;
}

static PyObject* py_norm(PyObject *self, PyObject *args) /* create_normalized_similarity_matrix wrapper */
{
    /* declarations */
    const char *filename; 
    Matrix *X = NULL, *S = NULL, *D = NULL, *N = NULL; 
    PyObject *out = NULL;
    /* sanity checks */
    if (!PyArg_ParseTuple(args, "s", &filename)) return NULL;
    /* py_norm */
    X = load_matrix_from_file(filename); 
    if (!X) /* sanity check */
    { 
        PyErr_SetString(PyExc_RuntimeError, "failed to load file"); 
        return NULL; 
    }
    S = create_similarity_matrix(X); 
    free_matrix(X);
    if (!S) /* sanity check */
    { 
        PyErr_SetString(PyExc_RuntimeError, "failed to build S"); 
        return NULL; 
    }
    D = create_diagonal_degree_matrix(S); 
    if (!D) /* sanity check */
    { 
        free_matrix(S); 
        PyErr_SetString(PyExc_RuntimeError, "failed to build D"); 
        return NULL; 
    }
    N = create_normalized_similarity_matrix(S, D); 
    free_matrix(S); 
    free_matrix(D);
    if (!N) /* sanity check */
    { 
        PyErr_SetString(PyExc_RuntimeError, "failed to build N"); 
        return NULL; 
    }
    out = matrix_to_pylist(N); 
    free_matrix(N);
    return out;
}

/* ---------- symnmf path ---------- */
static PyObject* py_symnmf(PyObject *self, PyObject *args, PyObject *kwargs) /* symnmf wrapper */
{
    /* declarations */
    static char *kwlist[] = {"N", "H0", "max_iters", "eps", "alpha", NULL};
    PyObject *N_obj = NULL, *H0_obj = NULL; 
    int max_iters = 300; 
    double eps = 1e-4, alpha = 1.0;
    Matrix *N = NULL, *H0 = NULL, *H = NULL; 
    PyObject *out = NULL; int rc;
    /* sanity checks */
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|idd", kwlist, &N_obj, &H0_obj, &max_iters, &eps, &alpha))
        return NULL;
    (void)max_iters; (void)eps; (void)alpha; /* ignored by c core  */
    /* py_symnmf */
    N = pylist_to_matrix(N_obj); 
    if (!N) goto done; /* sanity check */
    H0 = pylist_to_matrix(H0_obj); 
    if (!H0) goto done; /* sanity check */
    if (N->rows != N->columns) /* sanity check */
    { 
        PyErr_SetString(PyExc_ValueError, "N must be square"); 
        goto done; 
    }
    if (H0->rows != N->rows) /* sanity check */
    { 
        PyErr_SetString(PyExc_ValueError, "H0 rows must match N"); 
        goto done; 
    }
    rc = symnmf_c(N, H0, &H);
    if (rc != 0 || !H) /* sanity check */
    { 
        PyErr_SetString(PyExc_RuntimeError, "SymNMF failed"); 
        goto done; 
    }
    out = matrix_to_pylist(H);
  done: /* end of program */
    if (N) free_matrix(N);
    if (H0) free_matrix(H0);
    if (H) free_matrix(H);
    return out;
}

/* ---------- methods table & module ---------- */
static PyMethodDef SymNMF_FunctionsTable[] = {
    {"sym",  (PyCFunction)py_sym,  METH_VARARGS, "Compute S from file"},
    {"ddg",  (PyCFunction)py_ddg,  METH_VARARGS, "Compute D from file"},
    {"norm", (PyCFunction)py_norm, METH_VARARGS, "Compute N from file"},
    {"symnmf", (PyCFunction)py_symnmf, METH_VARARGS | METH_KEYWORDS,
        "Run SymNMF from in-memory matrices: H = SymNMF(N, H0; max_iters, eps, alpha)"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef SymNMFModule = {
    PyModuleDef_HEAD_INIT,
    "symnmfmodule",
    "SymNMF Python wrapper (compact API)",
    -1,
    SymNMF_FunctionsTable
};

PyMODINIT_FUNC PyInit_symnmfmodule(void)
{
    return PyModule_Create(&SymNMFModule);
}
