#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdbool.h>

#include <mpi.h>
#include "omp.h"

#include "parallelPoisson.h"

#define PI 3.14159265358979323846

void parallelPoisson(int problemSize,MPI_Comm comm)
{
	int numberOfColumns = problemSize-1;//numberOfUnknowns
	int fstBufferSize = 4*problemSize;
	double stepSize = 1.0/problemSize;

	double startTime, endTime;
	Vector diagonal = createVector(numberOfColumns);
	Vector fstBuffer = createVector(fstBufferSize);

	ColumnMatrix local_b = createColumnMatrixMPI(numberOfColumns,&comm);
	ColumnMatrix local_bt = createColumnMatrixMPI(numberOfColumns,&comm);
	ColumnMatrix sendBuffer = createColumnMatrixMPI(numberOfColumns,&comm);
	ColumnMatrix recvBuffer = createColumnMatrixMPI(numberOfColumns,&comm);

//	MPI_Datatype columnSendType; 
//	createMPIColumnSendType(local_b, &columnSendType);
	startTime = WallTime();
	diagonalEigenvalues(diagonal);

	initRightHandSide(local_b,stepSize);

	fastSineTransform(local_b,fstBuffer);

	mpiColumnMatrixTranspose(local_bt, recvBuffer, local_b, sendBuffer);

	fastSineTransformInv(local_bt,fstBuffer);
	
	systemSolver(local_bt,diagonal);

	fastSineTransform(local_bt,fstBuffer);
	
	mpiColumnMatrixTranspose(local_b, recvBuffer,local_bt,sendBuffer);
	
	fastSineTransformInv(local_b,fstBuffer);



	findAndPrintUmax(local_b);
	endTime = WallTime();
	if(local_b->commRank ==0)
		printf("Runtime: %fs\n",endTime-startTime);
	freeVectorMPI(diagonal);
	freeColumnMatrixMPI(local_b);
	freeColumnMatrixMPI(local_bt);
	freeVector(fstBuffer);
	freeColumnMatrixMPI(sendBuffer);
	freeColumnMatrixMPI(recvBuffer);
	//MPI_Type_free(&columnSendType);		

}

void initMPIApplication(int argc, char** argv, int* rank, int* size,MPI_Comm* WorldComm)
{
#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, size);
  MPI_Comm_rank(MPI_COMM_WORLD, rank);
  MPI_Comm_dup(MPI_COMM_WORLD, WorldComm);
#else
  *rank = 0;
  *size = 1;
#endif
}

void closeMPIApplication(MPI_Comm *comm)
{
#ifdef HAVE_MPI
  MPI_Comm_free(comm);
  MPI_Finalize();
#endif
}

Vector createVector(int length)
{
	Vector result = (Vector)malloc(sizeof(vector_t));
 	result->data = calloc(length,sizeof(double));
  	result->localSize = result->globalSize = length;
  	result->comm = NULL;
  	result->commSize = 1;
  	result->commRank = 0;
  	result->blockSize = NULL;
  	result->displacement = NULL;

	return result;
}
void freeVector(Vector vector)
{ 
	free(vector->data);
	free(vector);
}
Vector createVectorMPI(int length, MPI_Comm* comm)
{
	Vector result = (Vector)malloc(sizeof(vector_t));
	result->comm = comm;
	MPI_Comm_size(*comm,&result->commSize);
	MPI_Comm_rank(*comm,&result->commRank);
	splitVector(length, result->commSize,&result->blockSize,&result->displacement);
	result->localSize = result->blockSize[result->commRank];
	result->data = malloc(result->localSize);
	result->globalSize = length;
	
	return result;
}
void freeVectorMPI(Vector vectorMPI)
{
 	free(vectorMPI->data);
	free(vectorMPI->blockSize);
	free(vectorMPI->displacement);
	free(vectorMPI);
}
Vector createColumnMatrixMPI(int matrixSize, MPI_Comm* comm)
{
	ColumnMatrix result = (ColumnMatrix)malloc(sizeof(columnMatrix_t));
	result->comm = comm;
	MPI_Comm_size(*comm,&result->commSize);
	MPI_Comm_rank(*comm,&result->commRank);
	splitVector(matrixSize, result->commSize,&result->blockSize,&result->displacement);
	result->localSize = result->blockSize[result->commRank];
	result->data = malloc(result->localSize*matrixSize*sizeof(double));
	result->globalSize = matrixSize;
	
	return result;
}

void freeColumnMatrixMPI(ColumnMatrix matrix)
{ 
	free(matrix->data);
	free(matrix->blockSize);
	free(matrix->displacement);
	free(matrix);
}
void splitVector(int globalSize, int mpiSize, int** blockSize, int** displacement)
{
	*blockSize = calloc(mpiSize,sizeof(int));
	*displacement = calloc(mpiSize,sizeof(int));

  	for(int i=0;i<mpiSize;++i) 
	{
    		(*blockSize)[i] = globalSize/mpiSize;
    		if (globalSize % mpiSize && i >= (mpiSize - globalSize % mpiSize))
      		{
			(*blockSize)[i]++;
		}
    		if (i < mpiSize-1)
      		{
			(*displacement)[i+1] = (*displacement)[i]+(*blockSize)[i];
		}  
	}
}
void diagonalEigenvalues(Vector diagonal)
{
	int problemSize = diagonal->globalSize +1;
	#ifdef USE_OPENMP
	#pragma omp parallel for schedule(static)
	for(int i=0; i<diagonal->globalSize;i++)
	{
		diagonal->data[i] = 2.0*(1.0 -cos((i+1)*PI/problemSize));
	}
	#else
	for(int i=0; i<diagonal->globalSize;i++)
	{
		diagonal->data[i] = 2.0*(1.0 -cos((i+1)*PI/problemSize));
	}
	
	#endif
}
void systemSolver(ColumnMatrix localMatrix,Vector diagonal)
{
	#ifdef USE_OPENMP
	#pragma omp parallel for schedule(static)
	for(int column=0;column<localMatrix->localSize;column++)
	{
		for(int row=0;row<localMatrix->globalSize;row++)
		{
			localMatrix->data[row + column*localMatrix->globalSize] = localMatrix->data[row + column*localMatrix->globalSize] /(diagonal->data[column+localMatrix->displacement[localMatrix->commRank]]+diagonal->data[row]);
		}
	}
	#else
	for(int column=0;column<localMatrix->localSize;column++)
	{
		for(int row=0;row<localMatrix->globalSize;row++)
		{
			localMatrix->data[row + column*localMatrix->globalSize] = localMatrix->data[row + column*localMatrix->globalSize] /(diagonal->data[column+localMatrix->displacement[localMatrix->commRank]]+diagonal->data[row]);
		}
	}
	#endif

}
void initRightHandSide(ColumnMatrix local_b,double stepSize)
{
	#ifdef USE_OPENMP
	#pragma omp parallel for schedule(static)
	for(int column=0;column<local_b->localSize;column++)
	{
		for(int row=0;row<local_b->globalSize;row++)
		{
			local_b->data[row + column*local_b->globalSize] = stepSize*stepSize * rightHandSide((column+local_b->displacement[local_b->commRank])*stepSize,row*stepSize);
		}
	}
	#else
	for(int column=0;column<local_b->localSize;column++)
	{
		for(int row=0;row<local_b->globalSize;row++)
		{
			local_b->data[row + column*local_b->globalSize] = stepSize*stepSize * rightHandSide((column+local_b->displacement[local_b->commRank])*stepSize,row*stepSize);
		}
}
	#endif
}
double rightHandSide(double x, double y)
{
	return 2 * (y - y*y + x - x*x);
}
void fastSineTransform(ColumnMatrix localMatrix, Vector fstBuffer)
{
	int problemSize = localMatrix->globalSize+1;

	for(int column=0;column<localMatrix->localSize;column++)
	{
		fst_(&localMatrix->data[column*localMatrix->globalSize], &problemSize, fstBuffer->data, &fstBuffer->localSize);
	}
}
void fastSineTransformInv(ColumnMatrix localMatrix, Vector fstBuffer)
{
	int problemSize = localMatrix->globalSize+1;

	for(int column=0;column<localMatrix->localSize;column++)
	{	
		fstinv_(&localMatrix->data[column*localMatrix->globalSize], &problemSize, fstBuffer->data, &fstBuffer->localSize);
	}
}
/*
void createMPIColumnSendType(ColumnMatrix matrixBuffer,MPI_Datatype* columnSendType)
{	
	MPI_Type_vector(matrixBuffer->localSize,1,matrixBuffer->globalSize,MPI_DOUBLE,columnSendType);

	MPI_Type_commit(columnSendType);
}

void fillRecvCountAndRecvDispl(int** recvCounts,int** recvDispl, ColumnMatrix recvBuffer)
{
	*recvCounts = malloc(recvBuffer->commSize*sizeof(int));
	*recvDispl = malloc(recvBuffer->commSize*sizeof(int));
	for(int i=0;i<recvBuffer->commSize;i++)
		{
			*recvCounts[i] = recvBuffer->localSize*recvBuffer->blockSize[i];
			if(i<recvBuffer->commSize)
			{
				*recvDispl[i+1] = *recvDispl[i]+*recvCounts[i];
			}
		}
}
*/
void packTansposeData(ColumnMatrix sendData, ColumnMatrix sendBuffer)
{
	int i =0;
	//Copies each block of size l into the sendBuffer, s.t. all data to process p is stored sucsessively.
for(int p=0; p < sendData->commSize;p++)
	{
		for(int k=0;k<sendData->localSize;k++)
		{
			for(int l=0;l<sendData->blockSize[p];l++)
			{
				sendBuffer->data[i] = sendData->data[l+k*sendData->globalSize+sendData->displacement[p]];
				i +=1;
			}
		}
		sendBuffer->blockSize[p] = sendData->blockSize[p]*sendData->localSize;
		if(p<sendData->commSize)
		{
			sendBuffer->displacement[p+1] = sendBuffer->displacement[p] + sendBuffer->blockSize[p];
		}
	}
}
void unpackTansposeData(ColumnMatrix recvData, ColumnMatrix recvBuffer)
{
	int i =0;
	//Each block k is a row in the transposed matrix. Copies all blocks that are stored successively in recvBuffer, and stores them as rows in the column matrix recvData.
	for(int k =0; k<recvData->globalSize;k++)
	{
		for(int l=0;l<recvData->localSize;l++)
		{
			recvData->data[k+l*recvData->globalSize] = recvBuffer->data[i];
			i +=1;
		}
	}
}
void mpiColumnMatrixTranspose(ColumnMatrix recvData,ColumnMatrix recvBuffer, ColumnMatrix sendData, ColumnMatrix sendBuffer)
{
	packTansposeData(sendData, sendBuffer);	

	//All processors send and receive the same amount of doubles with the same processor.
	MPI_Alltoallv(sendBuffer->data,sendBuffer->blockSize,sendBuffer->displacement,MPI_DOUBLE,recvBuffer->data,sendBuffer->blockSize,sendBuffer->displacement,MPI_DOUBLE,*sendData->comm); 

	unpackTansposeData(recvData, recvBuffer);
}
void findAndPrintUmax(ColumnMatrix localMatrix)
{
	double uMaxLocal = 0.0, uMaxGlobal =0.0;
	Vector uMaxArray = createVector(localMatrix->globalSize);
	findUmax(&uMaxLocal,localMatrix->data,localMatrix->localSize*localMatrix->globalSize);

	//ERROR: MPI_Reduce have unexpected behaviour due to a strange error message concerning a call to free() within the call to MPI_Reduce
	//MPI_Reduce(&uMaxLocal,&uMaxGlobal,1,MPI_DOUBLE,MPI_MAX,0,*localMatrix->comm);

	MPI_Gather(&uMaxLocal,1,MPI_DOUBLE,uMaxArray->data,1,MPI_DOUBLE,0,*localMatrix->comm);
	findUmax(&uMaxGlobal,uMaxArray->data,uMaxArray->globalSize);
	if(localMatrix->commRank == 0)
	{
		printf("u max: %f\n",uMaxGlobal);
	}
}

void findUmax(double* uMax,double* dataArray,int arrayLength)
{
/*	#ifdef USE_OPENMP
	#pragma omp parallel for schedule(static) 
	for(int i=0; i < arrayLength; i++)
	{
		*uMax  = *uMax > dataArray[i] ? *uMax : dataArray[i];
	}
	#else
*/	for(int i=0; i < arrayLength; i++)
	{
		*uMax  = *uMax > dataArray[i] ? *uMax : dataArray[i];
	}
//	#endif
}
double WallTime ()
{
#ifdef HAVE_OPENMP
  return omp_get_wtime();
#else
  struct timeval tmpTime;
  gettimeofday(&tmpTime,NULL);
  return tmpTime.tv_sec + tmpTime.tv_usec/1.0e6;
#endif
}
