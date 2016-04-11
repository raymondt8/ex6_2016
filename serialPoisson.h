#ifndef SERIALPOISSON_H
#define SERIALPOISSON_H

typedef double real;
#include <stdbool.h>

// Function prototypes
real *mk_1D_array(int n, bool zero);
real **mk_2D_array(int n1, int n2, bool zero);
void transpose(real **bt, real **b, int m);
real rhs(real x, real y);
void fst_(real *v, int *n, real *w, int *nn);
void fstinv_(real *v, int *n, real *w, int *nn);
void serialPoisson(int n);
void free_1D_array(real* vector_array);
void free_2D_array(real** matrix_array);
#endif
