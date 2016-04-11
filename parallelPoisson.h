#ifndef PARALLELPOISSON_H
#define PARALLELPOISSON_H

//#define HAVE_MPI
//#define USE_OPENMP
void fst_(double *v, int *n, double *w, int *nn);
void fstinv_(double *v, int *n, double *w, int *nn);

extern MPI_Comm WorldComm;

void initMPIApplication(
	int argc, 	//Number of arguments to main function
	char** argv, 	//list of arguments to main function
	int* rank, 	//returns local rank
	int* size,	//returns MPI domain size
	MPI_Comm* WorldComm
	);//Initiate MPI and openMP if enabled in main.

void closeMPIApplication(
	MPI_Comm* comm	//MPI communicator to be closed
	);//Closes MPI

//Sructure describing a columnvise stored matrix and vectors for MPI applications. Matrix: one element is one clumn. Vector: one element is one data pointo
typedef struct{
	double* data;	//matrix data
	int localSize;	//number of elements for the local processor
	int globalSize;	//size of the vector in one dimention
	#ifdef HAVE_MPI
	MPI_Comm* comm;	//MPI communicator the matrix is spil aross
	#endif
	int commSize;	//size of communicator
	int commRank;	//rank of calling process
	int* displacement;	//displacement between each block for each process
	int* blockSize;	//number of elements for each process	
} columnMatrix_t, vector_t;

typedef columnMatrix_t* ColumnMatrix;
typedef vector_t* Vector;

Vector createVector(
	int length	//length of vector
	);//Creates and returns a vector of requested length

void freeVector(
	Vector vector	//Vector to be freed
	);//Frees the vector structure 

Vector createVectorMPI(
	int length,	//length of vector
	MPI_Comm* comm	//MPI communicaor
	);//Creates a vector that is splittet over the MPI processors

void freeVectorMPI(
	Vector vector	//Vector to be freed
	);//Frees the mpi vector structure

Vector createColumnMatrixMPI(
	int matrixSize,	//size of a sqare matrix in one direction
	MPI_Comm* comm	//MPI communicator to split matrix across
	);//Creates a column stored matrix splitted over the MPI processors

void freeColumnMatrixMPI(
	ColumnMatrix matrix	//matrix to be freed
	);//frees the data allocated to the column matrix

void splitVector(
	int globalSize,	//Lengt of vector to be distributed
	int mpiSize, 	//Size of MPI communication domain
	int** blockSize,  //List of the local sizes of the splitted vector
	int** displacement//starting point of each local vector relative to the "global" vector
	);//Splits the vector between processors and stores the result into the pointers given	

double rightHandSide(
	double x, 	//position in x-direction (grid_i)
	double y	//position in y-direction (grid_j)
	);//Calculares the right hand side of the poisson problem for the given positions

void parallelPoisson(
	int problemSize, //size of problem to be solved
	MPI_Comm comm	//MPI communication domain
	);//Takes in the problem size and the MPI comminication domain and solves the poisson problem usin MPI and OpenMP if those are enabled

void diagonalEigenvalues(
	Vector diagonal	//vector to place eigenvalues
	);//Computes the eigenvalues of the poisson problem and places them in the "diagonal vector" 

void initRightHandSide(
	ColumnMatrix local_b,	//Vector to store rhs
	double stepSize		//stepsize
	);//Claculates the right hand side of the poisson problem and stores it in local_b

void fastSineTransform(
	ColumnMatrix localMatrix, 	//Columns to be sent to fst
	Vector fstBuffer	//Vector for use of fst
	);//Calls fast sine transform for each column

void fastSineTransformInv(
	ColumnMatrix localMatrix, 	//Columns to be sent to fstinv
	Vector fstBuffer	//Vector for use of fstinv
	);//Calls fast sine transform inv. for each column

void systemSolver(
	ColumnMatrix local_bt,	//System to solve
	Vector diagonal	//vector of diagonal eigenvalues
	);//Solves the system: lambda * Xtilde=Btilde

void mpiColumnMatrixTranspose(
	ColumnMatrix recvData,		//Local transposed matrix to store the data
	ColumnMatrix recvBuffer,	//buffer to receive the data into
	ColumnMatrix sendData,		//Local mcolumnMatrix to be transposed
	ColumnMatrix sendBuffer 	//buffer to locally transpose the matrix before distribution
	);//Transposes the columnMatrix from the sendData into the recvData by using two buffers, send- and recvBuffer to locally transpose the data to be distributed between processors

void createMPIColumnSendType(
	ColumnMatrix matrixBuffer,	//vectorstruct to create datatype from
	MPI_Datatype* columnSendType	//Datatype to be created and committed
	);//Creates a new derived MPI datatype which selects relevant part of columns to send to each process
void fillRecvCountAndRecvDispl(
	int** recvCounts,	//Adress of pointer to connect to array in order to fill with receiving number of elements
	int** recvDispl,	//Adress of pointer to  Array to fill with displacements of receiving data
	ColumnMatrix recvBuffer	//The receiving buffer is used to get the comm size, and local block sizes
	);//Takes in two pointer adresses type int, conneces them to an array with malloc and fills them with the maximal number of elements to be received from each processor in recvCounts and displacement of the receiving data to be placed in recvDispl

void packTansposeData(
	ColumnMatrix sendData,	//Local columnMatrix to be transposed
	ColumnMatrix sendBuffer //Buffer to temporarily store the data to be distributed
	);//Takes in the local columnMatrix and pack the data into blocks to be sendt for each processor

void unpackTransposeData(
	ColumnMatrix recvData, 	//Local columnMatrix to store transposed data
	ColumnMatrix recvBuffer //receiving Buffer to unpack the data from
	);//Places the transposed columnparts into correct order in the receiving matrix 
void findAndPrintUmax(
	ColumnMatrix local_b	//matrix to locate uMax from
	);//Finds and prints the uMax from the diatributed matrix with MPI

void findUmax(
	double* uMax,	//variable to store the result 
	double* dataArray,	//array to find the maximal value from
	int arrayLength	//array length
	);//Finds the maximal value in a array and stores the result in given variable

double WallTime(
	);//Returns the current walltime in seconds

#endif
