/**
 *
 * Date         12/07/2009
 * ====
 *
 * Authors      Vincent Garcia
 * =======      Eric    Debreuve
 *              Michel  Barlaud
 *
 * Description  Given a reference point set and a query point set, the program returns
 * ===========  firts the distance between each query point and its k nearest neighbors in
 *              the reference point set, and second the indexes of these k nearest neighbors.
 *              The computation is performed using the API NVIDIA CUDA.
 *
 * Paper        Fast k nearest neighbor search using GPU
 * =====
 *
 * BibTeX       @INPROCEEDINGS{2008_garcia_cvgpu,
 * ======         author = {V. Garcia and E. Debreuve and M. Barlaud},
 *                title = {Fast k nearest neighbor search using GPU},
 *                booktitle = {CVPR Workshop on Computer Vision on GPU},
 *                year = {2008},
 *                address = {Anchorage, Alaska, USA},
 *                month = {June}
 *              }
 *
 */


// If the code is used in Matlab, set MATLAB_CODE to 1. Otherwise, set MATLAB_CODE to 0.
#define MATLAB_CODE 0


// Includes
#include <stdio.h>
#include <math.h>
#include "cuda.h"
#include "cublas.h"
#if MATLAB_CODE == 1
#include "mex.h"
#else
#include <time.h>
#endif


// Constants used by the program
#define MAX_PITCH_VALUE_IN_BYTES       262144
#define MAX_TEXTURE_WIDTH_IN_BYTES     65536
#define MAX_TEXTURE_HEIGHT_IN_BYTES    32768
#define MAX_PART_OF_FREE_MEMORY_USED   0.9
#define BLOCK_DIM                      16
#define NUM_ITERATION 10


//-----------------------------------------------------------------------------------------------//
//                                            KERNELS                                            //
//-----------------------------------------------------------------------------------------------//


/**
 * Given a matrix of size width*height, compute the square norm of each column.
 *
 * @param mat    : the matrix
 * @param width  : the number of columns for a colum major storage matrix
 * @param height : the number of rowm for a colum major storage matrix
 * @param norm   : the vector containing the norm of the matrix
 */
__global__ void cuComputeNorm(float *mat, int width, int pitch, int height, float *norm){
  unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
  if (xIndex<width){
    float val, sum=0;
    int i;
    for (i=0;i<height;i++){
      val  = mat[i*pitch+xIndex];
      sum += val*val;
    }
    norm[xIndex] = sum;
  }
}



/**
 * Given the distance matrix of size width*height, adds the column vector
 * of size 1*height to each column of the matrix.
 *
 * @param dist   : the matrix
 * @param width  : the number of columns for a colum major storage matrix
 * @param pitch  : the pitch in number of column
 * @param height : the number of rowm for a colum major storage matrix
 * @param vec    : the vector to be added
 */
__global__ void cuAddRNorm(float *dist, int width, int pitch, int height, float *vec){
  unsigned int tx = threadIdx.x;
  unsigned int ty = threadIdx.y;
  unsigned int xIndex = blockIdx.x * blockDim.x + tx;
  unsigned int yIndex = blockIdx.y * blockDim.y + ty;
  __shared__ float shared_vec[16];
  if (tx==0 && yIndex<height)
    shared_vec[ty]=vec[yIndex];
  __syncthreads();
  if (xIndex<width && yIndex<height)
    dist[yIndex*pitch+xIndex]+=shared_vec[ty];
}



/**
 * Gathers k-th smallest distances for each column of the distance matrix in the top.
 *
 * @param dist        distance matrix
 * @param dist_pitch  pitch of the distance matrix given in number of columns
 * @param ind         index matrix
 * @param ind_pitch   pitch of the index matrix given in number of columns
 * @param width       width of the distance matrix and of the index matrix
 * @param height      height of the distance matrix and of the index matrix
 * @param k           number of neighbors to consider
 */
__global__ void cuInsertionSort(float *dist, int dist_pitch, int *ind, int ind_pitch, int width, int height, int k){

  // Variables
  int l, i, j;
  float *p_dist;
  int   *p_ind;
  float curr_dist, max_dist;
  int   curr_row,  max_row;
  unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;

  if (xIndex<width){

    // Pointer shift, initialization, and max value
    p_dist   = dist + xIndex;
    p_ind    = ind  + xIndex;
    max_dist = p_dist[0];
    p_ind[0] = 1;

    // Part 1 : sort kth firt elementZ
    for (l=1; l<k; l++){
      curr_row  = l * dist_pitch;
      curr_dist = p_dist[curr_row];
      if (curr_dist<max_dist){
        i=l-1;
        for (int a=0; a<l-1; a++){
          if (p_dist[a*dist_pitch]>curr_dist){
            i=a;
            break;
          }
        }
        for (j=l; j>i; j--){
          p_dist[j*dist_pitch] = p_dist[(j-1)*dist_pitch];
          p_ind[j*ind_pitch]   = p_ind[(j-1)*ind_pitch];
        }
        p_dist[i*dist_pitch] = curr_dist;
        p_ind[i*ind_pitch]   = l+1;
      }
      else
        p_ind[l*ind_pitch] = l+1;
      max_dist = p_dist[curr_row];
    }

    // Part 2 : insert element in the k-th first lines
    max_row = (k-1)*dist_pitch;
    for (l=k; l<height; l++){
      curr_dist = p_dist[l*dist_pitch];
      if (curr_dist<max_dist){
        i=k-1;
        for (int a=0; a<k-1; a++){
          if (p_dist[a*dist_pitch]>curr_dist){
            i=a;
            break;
          }
        }
        for (j=k-1; j>i; j--){
          p_dist[j*dist_pitch] = p_dist[(j-1)*dist_pitch];
          p_ind[j*ind_pitch]   = p_ind[(j-1)*ind_pitch];
        }
        p_dist[i*dist_pitch] = curr_dist;
        p_ind[i*ind_pitch]   = l+1;
        max_dist             = p_dist[max_row];
      }
    }
  }
}


/**
 * Computes the square root of the first line (width-th first element)
 * of the distance matrix.
 *
 * @param dist    distance matrix
 * @param width   width of the distance matrix
 * @param pitch   pitch of the distance matrix given in number of columns
 * @param k       number of neighbors to consider
 */
__global__ void cuAddQNormAndSqrt(float *dist, int width, int pitch, float *q, int k){
  unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
  if (xIndex<width && yIndex<k)
    dist[yIndex*pitch + xIndex] = sqrt(dist[yIndex*pitch + xIndex] + q[xIndex]);
}



//-----------------------------------------------------------------------------------------------//
//                                   K-th NEAREST NEIGHBORS                                      //
//-----------------------------------------------------------------------------------------------//


/**
 * Prints the error message return during the memory allocation.
 *
 * @param error        error value return by the memory allocation function
 * @param memorySize   size of memory tried to be allocated
 */
void printErrorMessage(cudaError_t error, int memorySize){
  printf("==================================================\n");
  printf("MEMORY ALLOCATION ERROR  : %s\n", cudaGetErrorString(error));
  printf("Whished allocated memory : %d\n", memorySize);
  printf("==================================================\n");
#if MATLAB_CODE == 1
  mexErrMsgTxt("CUDA ERROR DURING MEMORY ALLOCATION");
#endif
}


/**
 * K nearest neighbor algorithm
 * - Initialize CUDA
 * - Allocate device memory
 * - Copy point sets (reference and query points) from host to device memory
 * - Compute the distances + indexes to the k nearest neighbors for each query point
 * - Copy distances from device to host memory
 *
 * @param ref_host      reference points ; pointer to linear matrix
 * @param ref_width     number of reference points ; width of the matrix
 * @param query_host    query points ; pointer to linear matrix
 * @param query_width   number of query points ; width of the matrix
 * @param height        dimension of points ; height of the matrices
 * @param k             number of neighbor to consider
 * @param dist_host     distances to k nearest neighbors ; pointer to linear matrix
 * @param dist_host     indexes of the k nearest neighbors ; pointer to linear matrix
 *
 */
void knn(float* ref_host, int ref_width, float* query_host, int query_width, int height, int k, float* dist_host, int* ind_host){

  unsigned int size_of_float = sizeof(float);
  unsigned int size_of_int   = sizeof(int);

  // Variables
  float        *query_dev;
  float        *ref_dev;
  float        *dist_dev;
  float        *query_norm;
  float        *ref_norm;
  int          *ind_dev;
  cudaError_t  result;
  size_t       query_pitch;
  size_t       query_pitch_in_bytes;
  size_t       ref_pitch;
  size_t       ref_pitch_in_bytes;
  size_t       ind_pitch;
  size_t       ind_pitch_in_bytes;
  size_t       max_nb_query_traited;
  size_t       actual_nb_query_width;
  size_t       memory_total;
  size_t       memory_free;

  // CUDA Initialisation
  cuInit(0);

  // Check free memory using driver API ; only (MAX_PART_OF_FREE_MEMORY_USED*100)% of memory will be used
  CUcontext cuContext;
  CUdevice  cuDevice=0;
  cuCtxCreate(&cuContext, 0, cuDevice);
  cuMemGetInfo(&memory_free, &memory_total);
  cuCtxDetach (cuContext);

  // Determine maximum number of query that can be treated
  max_nb_query_traited = ( memory_free * MAX_PART_OF_FREE_MEMORY_USED - size_of_float * ref_width*(height+1) ) / ( size_of_float * (height + ref_width + 1) + size_of_int * k);
  max_nb_query_traited = min( query_width, ((int)max_nb_query_traited / 16) * 16 );
  printf("max_nb_query_traited = %d\n", (int)max_nb_query_traited);

  // Allocation of global memory for query points and for distances
  result = cudaMallocPitch( (void **) &query_dev, &query_pitch_in_bytes, max_nb_query_traited * size_of_float, height + ref_width + 1);
  if (result){
    printErrorMessage(result, max_nb_query_traited*size_of_float*(height+ref_width));
    return;
  }
  query_pitch = query_pitch_in_bytes/size_of_float;
  query_norm  = query_dev  + height * query_pitch;
  dist_dev    = query_norm + query_pitch;

  // Allocation of global memory for reference points and ||query||
  result = cudaMallocPitch((void **) &ref_dev, &ref_pitch_in_bytes, ref_width * size_of_float, height+1);
  if (result){
    printErrorMessage(result, ref_width * size_of_float * ( height+1 ));
    cudaFree(query_dev);
    return;
  }
  ref_pitch = ref_pitch_in_bytes / size_of_float;
  ref_norm  = ref_dev + height * ref_pitch;

  // Allocation of global memory for indexes
  result = cudaMallocPitch( (void **) &ind_dev, &ind_pitch_in_bytes, max_nb_query_traited * size_of_int, k);
  if (result){
    printErrorMessage(result, max_nb_query_traited*size_of_int*k);
    cudaFree(ref_dev);
    cudaFree(query_dev);
    return;
  }
  ind_pitch = ind_pitch_in_bytes/size_of_int;

  // Memory copy of ref_host in ref_dev
  result = cudaMemcpy2D(ref_dev, ref_pitch_in_bytes, ref_host, ref_width*size_of_float, ref_width*size_of_float, height, cudaMemcpyHostToDevice);

  cudaError_t err;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float elapsed;
  float kernel1_time = 0.0f;
  float kernel2_time = 0.0f;
  float kernel3_time = 0.0f;
  float kernel4_time = 0.0f;
  float kernel5_time = 0.0f;
  float kernel6_time = 0.0f;

  for (int iter = 1; iter <= NUM_ITERATION; ++iter) {

  // Computation of reference square norm
  dim3 ref_grid(ref_width/256, 1, 1);
  dim3 ref_thread(256, 1, 1);
  if (ref_width%256 != 0) ref_grid.x += 1;
  cudaEventRecord(start);
  cuComputeNorm<<<ref_grid,ref_thread>>>(ref_dev, ref_width, ref_pitch, height, ref_norm);
  cudaDeviceSynchronize();
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
    return;
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed, start, stop);
  kernel1_time += elapsed;

  // Split queries to fit in GPU memory
  for (int i=0; i<query_width; i+=max_nb_query_traited){

    // Number of query points considered
    actual_nb_query_width = min( (int)max_nb_query_traited, query_width-i );

    // Copy of part of query actually being treated
    cudaMemcpy2D(query_dev, query_pitch_in_bytes, &query_host[i], query_width*size_of_float, actual_nb_query_width*size_of_float, height, cudaMemcpyHostToDevice);

    // Computation of Q square norm
    dim3 query_grid_1(actual_nb_query_width/256, 1, 1);
    dim3 query_thread_1(256, 1, 1);
    if (actual_nb_query_width%256 != 0) query_grid_1.x += 1;
    cudaEventRecord(start);
    cuComputeNorm<<<query_grid_1,query_thread_1>>>(query_dev, actual_nb_query_width, query_pitch, height, query_norm);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("CUDA error: %s\n", cudaGetErrorString(err));
      return;
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    kernel2_time += elapsed;

    // Computation of Q*transpose(R)
    cudaEventRecord(start);
    cublasSgemm('n', 't', (int)query_pitch, (int)ref_pitch, height, (float)-2.0, query_dev, query_pitch, ref_dev, ref_pitch, (float)0.0, dist_dev, query_pitch);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    kernel3_time += elapsed;

    // Add R norm to distances
    dim3 query_grid_2(actual_nb_query_width/16, ref_width/16, 1);
    dim3 query_thread_2(16, 16, 1);
    if (actual_nb_query_width%16 != 0) query_grid_2.x += 1;
    if (ref_width%16 != 0) query_grid_2.y += 1;
    cudaEventRecord(start);
    cuAddRNorm<<<query_grid_2,query_thread_2>>>(dist_dev, actual_nb_query_width, query_pitch, ref_width, ref_norm);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("CUDA error: %s\n", cudaGetErrorString(err));
      return;
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    kernel4_time += elapsed;

    // Sort each column
    cudaEventRecord(start);
    cuInsertionSort<<<query_grid_1,query_thread_1>>>(dist_dev, query_pitch, ind_dev, ind_pitch, actual_nb_query_width, ref_width, k);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("CUDA error: %s\n", cudaGetErrorString(err));
      return;
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    kernel5_time += elapsed;

    // Add Q norm and compute Sqrt ONLY ON ROW K-1
    cudaEventRecord(start);
    cuAddQNormAndSqrt<<<query_grid_2,query_thread_2>>>( dist_dev, actual_nb_query_width, query_pitch, query_norm, k);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("CUDA error: %s\n", cudaGetErrorString(err));
      return;
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    kernel6_time += elapsed;

    // Memory copy of output from device to host
    cudaMemcpy2D(&dist_host[i], query_width*size_of_float, dist_dev, query_pitch_in_bytes, actual_nb_query_width*size_of_float, k, cudaMemcpyDeviceToHost);
    cudaMemcpy2D(&ind_host[i],  query_width*size_of_int,   ind_dev,  ind_pitch_in_bytes,   actual_nb_query_width*size_of_int,   k, cudaMemcpyDeviceToHost);
  }

  }

  printf("Average time of %d runs:\n", NUM_ITERATION);

  printf("cuComputeNorm(ref) %.3f ms\n", kernel1_time / NUM_ITERATION);
  printf("cuComputeNorm(query) %.3f ms\n", kernel2_time / NUM_ITERATION);
  printf("cublasSgemm %.3f ms\n", kernel3_time / NUM_ITERATION);
  printf("cuAddRNorm %.3f ms\n", kernel4_time / NUM_ITERATION);
  printf("cuInsertionSort %.3f ms\n", kernel5_time / NUM_ITERATION);
  printf("cuAddQNormAndSqrt %.3f ms\n", kernel6_time / NUM_ITERATION);

  float dist_comp_time =
      kernel1_time + kernel2_time + kernel3_time + kernel4_time + kernel6_time;
  printf("DistanceComputation %.3f ms\n", dist_comp_time / NUM_ITERATION);
  printf("NeighborSelection %.3f ms\n", kernel5_time / NUM_ITERATION);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // Free memory
  cudaFree(ind_dev);
  cudaFree(ref_dev);
  cudaFree(query_dev);
}



//-----------------------------------------------------------------------------------------------//
//                                MATLAB INTERFACE & C EXAMPLE                                   //
//-----------------------------------------------------------------------------------------------//


#if MATLAB_CODE == 1

/**
 * Interface to use CUDA code in Matlab (gateway routine).
 *
 * @param nlhs  	Number of expected mxArrays (Left Hand Side)
 * @param plhs 	Array of pointers to expected outputs
 * @param nrhs 	Number of inputs (Right Hand Side)
 * @param prhs 	Array of pointers to input data. The input data is read-only and should not be altered by your mexFunction .
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){

  // Variables
  float* ref;
  int    ref_width;
  int    ref_height;
  float* query;
  int    query_width;
  int    query_height;
  float* dist;
  int*   ind;
  int    k;

  // Reference points
  ref          = (float *) mxGetData(prhs[0]);
  ref_width    = mxGetM(prhs[0]);
  ref_height   = mxGetN(prhs[0]);

  // Query points
  query        = (float *) mxGetData(prhs[1]);
  query_width  = mxGetM(prhs[1]);
  query_height = mxGetN(prhs[1]);

  // Number of neighbors to consider
  k            = (int)mxGetScalar(prhs[2]);

  // Verification of the reference point and query point sizes
  if (ref_height!=query_height)
    mexErrMsgTxt("Data must have the same dimension");
  if (ref_width*sizeof(float)>MAX_PITCH_VALUE_IN_BYTES)
    mexErrMsgTxt("Reference number is too large for CUDA (Max=65536)");
  if (query_width*sizeof(float)>MAX_PITCH_VALUE_IN_BYTES)
    mexErrMsgTxt("Query number is too large for CUDA (Max=65536)");

  // Allocation of output arrays
  dist = (float *) mxGetPr(plhs[0] = mxCreateNumericMatrix(query_width, k, mxSINGLE_CLASS, mxREAL));
  ind  =   (int *) mxGetPr(plhs[1] = mxCreateNumericMatrix(query_width, k, mxINT32_CLASS,  mxREAL));

  // Call KNN CUDA
  knn(ref, ref_width, query, query_width, ref_height, k, dist, ind);
}

#else // C code

/**
 * Example of use of kNN search CUDA.
 */
int main(int argc, char **argv){

  // Variables and parameters
  float* ref;                 // Pointer to reference point array
  float* query;               // Pointer to query point array
  float* dist;                // Pointer to distance array
  int*   ind;                 // Pointer to index array
  int    ref_nb     = 100000; // Reference point number, max=65535
  int    query_nb   = 100;    // Query point number,     max=65535
  int    dim        = 100;    // Dimension of points,    max=8192
  int    k          = 100;    // Nearest neighbors to consider

  // Memory allocation
  ref    = (float *) malloc(ref_nb   * dim * sizeof(float));
  query  = (float *) malloc(query_nb * dim * sizeof(float));
  dist   = (float *) malloc(query_nb * k * sizeof(float));
  ind    = (int *)   malloc(query_nb * k * sizeof(float));

  // Init
  FILE *file_reference = fopen(argv[1], "rb");
  fread(ref, sizeof(float), ref_nb * dim, file_reference);
  fclose(file_reference);

  FILE *file_query = fopen(argv[2], "rb");
  fread(query, sizeof(float), query_nb * dim, file_query);
  fclose(file_query);

  // Display informations
  printf("Number of reference points      : %6d\n", ref_nb  );
  printf("Number of query points          : %6d\n", query_nb);
  printf("Dimension of points             : %4d\n", dim     );
  printf("Number of neighbors to consider : %4d\n", k       );
  printf("Processing kNN search           : \n"             );

  // Call kNN search CUDA
  knn(ref, ref_nb, query, query_nb, dim, k, dist, ind);

  // Both dist and ind are k by query_nb. Column c has the nearest neighbors of
  // query c.

  FILE *file_knn_dist = fopen("dist.txt", "w");
  for (int i = 0; i < query_nb; ++i) {
    for (int j = 0; j < k; ++j) {
      if (j > 0) fprintf(file_knn_dist, " ");
      fprintf(file_knn_dist, "%.5f", dist[j * query_nb + i]);
    }
    fprintf(file_knn_dist, "\n");
  }
  fclose(file_knn_dist);

  FILE *file_knn_idx = fopen("idx.txt", "w");
  for (int i = 0; i < query_nb; ++i) {
    for (int j = 0; j < k; ++j) {
      if (j > 0) fprintf(file_knn_idx, " ");
      fprintf(file_knn_idx, "%d", ind[j * query_nb + i] - 1);  // convert to indexing from 0
    }
    fprintf(file_knn_idx, "\n");
  }
  fclose(file_knn_idx);

  // Free memory
  free(ind);
  free(dist);
  free(query);
  free(ref);
}

#endif
