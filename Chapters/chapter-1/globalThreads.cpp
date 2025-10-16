
// __Global__ indicates that the subroutine can be executed from CPU and runs on the GPU.
__global__ void process(double *a, const double *b, int n) {

    // A unique index number is calculated for each thread id
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    // and the number of threads is stored in the threadCount.
    int threadCount = gridDim.x * blockDim.x;