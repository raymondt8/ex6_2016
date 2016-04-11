/**
 * C program to solve the two-dimensional Poisson equation on
 * a unit square using one-dimensional eigenvalue decompositions
 * and fast sine transforms.
 *
 * Einar M. Rønquist
 * NTNU, October 2000
 * Revised, October 2001
 * Revised by Eivind Fonn, February 2015
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "serialPoisson.h"

#define PI 3.14159265358979323846
#define true 1
#define false 0

void serialPoisson(int n)
{
    // The number of grid points in each direction is n+1
    // The number of degrees of freedom in each direction is n-1

    int m = n - 1;
    int nn = 4 * n;
    real h = 1.0 / n;

    // Grid points
    real *grid = mk_1D_array(n+1, false);
    for (int i = 0; i < n+1; i++) {
        grid[i] = i * h;
    }

    // The diagonal of the eigenvalue matrix of T
    real *diag = mk_1D_array(m, false);
    for (int i = 0; i < m; i++) {
        diag[i] = 2.0 * (1.0 - cos((i+1) * PI / n));
    }

    // Initialize the right hand side data
    real **b = mk_2D_array(m, m, false);
    real **bt = mk_2D_array(m, m, false);
    real *z = mk_1D_array(nn, false);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            b[i][j] = h * h * rhs(grid[i], grid[j]);
        }
    }
 
    // Calculate Btilde^T = S^-1 * (S * B)^T
    for (int i = 0; i < m; i++) {
        fst_(b[i], &n, z, &nn);
    }
     transpose(bt, b, m);



    for (int i = 0; i < m; i++) {
        fstinv_(bt[i], &n, z, &nn);
    }
 
    // Solve Lambda * Xtilde = Btilde
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            bt[i][j] = bt[i][j] / (diag[i] + diag[j]);
        }
    }

    // Calculate X = S^-1 * (S * Xtilde^T)
    for (int i = 0; i < m; i++) {
        fst_(bt[i], &n, z, &nn);
    }

    transpose(b, bt, m);


   for (int i = 0; i < m; i++) {
        fstinv_(b[i], &n, z, &nn);
    }

    // Calculate maximal value of solution
    double u_max = 0.0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            u_max = u_max > b[i][j] ? u_max : b[i][j];
        }
    }

    printf("u_max = %e\n", u_max);
;
    free_1D_array(diag);
    free_1D_array(grid);
    free_1D_array(z);
    free_2D_array(b);
    free_2D_array(bt);

}

real rhs(real x, real y) {
    return 2 * (y - y*y + x - x*x);
}

void transpose(real **bt, real **b, int m)
{
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            bt[i][j] = b[j][i];
        }
    }
}

real *mk_1D_array(int n, bool zero)
{
    if (zero) {
        return (real *)calloc(n, sizeof(real));
    }
    return (real *)malloc(n * sizeof(real));
}

real **mk_2D_array(int n1, int n2, bool zero)
{
    real **ret = (real **)malloc(n1 * sizeof(real *));

    if (zero) {
        ret[0] = (real *)calloc(n1 * n2, sizeof(real));
    }
    else {
        ret[0] = (real *)malloc(n1 * n2 * sizeof(real));
    }

    for (int i = 1; i < n1; i++) {
        ret[i] = ret[i-1] + n2;
    }
    return ret;
}

void free_1D_array(real* vector_array)
{
	free(vector_array);
}
void free_2D_array(real** matrix_array)
{
	free(matrix_array[0]);
	free(matrix_array);
}
