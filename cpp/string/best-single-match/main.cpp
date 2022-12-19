#include <Python.h>
#include "algo/bmh-sbndm.cpp"
#include "algo/fjs.cpp"
#include "algo/fsbndm.cpp"
#include "algo/hash3.cpp"
#include "algo/tvsbs.cpp"

namespace main{
int search(const char *x, int m, const char *y, int n){
    // if(m<=8){
    //     return fjs::search(x, m, y, n);
    // }
    // else if (m<=32){
    //     return bmh_sbndm::search(x, m, y, n);
    // }
    // else if (m<=256){
    //     return fsbndm::search(x, m, y, n);
    // }
    // else if (m<=1024){
    //     return tvsbs::search(x, m, y, n);
    // }
    // else {
    //     return hash3::search(x, m, y, n);
    // }
    return fjs::search(x, m, y, n);
}
}

/* Python Wrapper Functions*/
/* Destructor function for points */
static void del_(PyObject *obj) {
    free(PyCapsule_GetPointer(obj, NULL));
}

/* Utility functions */
// static kmp_next *Pykmp_Askmpnext(PyObject *obj) {
//     return (kmp_next *) PyCapsule_GetPointer(obj, "kmp_next");
// }

// static PyObject *Pykmp_Fromkmpnext(kmp_next *next, int must_free) {
//     return PyCapsule_New(next, "kmp_next", must_free ? del_kmpnext : NULL);
// }

// search
static PyObject *py_search(PyObject *self, PyObject *args) {
    const char *text;
    const char *pattern;
    if (!PyArg_ParseTuple(args, "ss", &text, &pattern)) {
        return NULL;
    }
    int result = main::search(pattern, strlen(pattern), text, strlen(text));
    return Py_BuildValue("i", result);
}

/* Module method table */
static PyMethodDef strutilMethods[] = {
    {"search", py_search, METH_VARARGS, "Search pattern in text"},
    {NULL, NULL, 0, NULL}
};

/* Module definition structure */
static struct PyModuleDef strutilmodule = {
    PyModuleDef_HEAD_INIT,  /* m_base */
    "strutil",                  /* name of module */
    "strutil",                  /* module documentation, may be NULL */
    -1,                     /* size of per-interpreter state or -1 */
    strutilMethods              /* method table */
};

/* Module initialization function */
PyMODINIT_FUNC PyInit_strutil(void) {
    Py_Initialize();
    return PyModule_Create(&strutilmodule);
};