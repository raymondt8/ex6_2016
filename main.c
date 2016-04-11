#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdbool.h>

#include <mpi.h>
#include "omp.h"

#include "parallelPoisson.h"
#include "serialPoisson.h"

#define PI 3.14159265358979323846
//#define true 1
//#define false 0


//#define HAVE_MPI
//#define HAVE_OPENMP

int main(int argc,char **argv)
{
	if (argc < 2) {
	printf("Usage:\n");
    	printf("  poisson n\n\n");
    	printf("Arguments:\n");
 
   	printf("  n: the problem size (must be a power of 2)\n");
	}

     	// The number of grid points in each direction is n+1
     	// The number of degrees of freedom in each direction is n-1
    	int problemSize = atoi(argv[1]);
	
	#ifdef HAVE_MPI
	//For reference on small sets:
	if(problemSize <=64)
	{
		serialPoisson(problemSize);
	}

	int rank,size;
	MPI_Comm WorldComm;
	//MPI_Comm SelfComm;

	//Initialize application - MPI & openMP
	initMPIApplication(argc, argv, &rank, &size,&WorldComm);
	//	printf("Start parallelPoisson! rank: %i\n",rank); 	
	parallelPoisson(problemSize,WorldComm);	
	//Closing application - MPI & openMP
	closeMPIApplication(&WorldComm);	
	#else
	serialPoisson(problemSize);
	   	
	#endif
	return 0;
}  
