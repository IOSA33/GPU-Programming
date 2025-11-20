#include <iostream>
#include <random>   // std::default_random_engine, std::uniform_int_distribution

#include <hip/hip_runtime.h>

// Number of elements
#define N 74672162


// =====================================================================
// =====================================================================
// The GPU kernel

// A kernel that takes an array of integers (n elements) as an argument and
// increments each element by one. The __global__ keyword indicates that this
// is a kernel that can be called from the host program. Functions defined
// without the __global__ keyword are not kernels and runs on the host Functions
// with the __device__ keyword can only be called from a kernel.
// It is assumed that the number of threads is greater than n.
__global__ void axpy(const double *x, double *y, int n, double a)
{
    // The index idx is obtained by adding the block-local thread index
    // threadIdx.x to the number of threads in previous blocks. That number
    // is cmoputed as the product of the number of threads in a block, namely
    // blockDim.x, and the block index blockIdx.x.
    // Note the ".x", this is because the thread and block structure can be
    // 2 or 3 dimensional, in which case ".y" and ".z" would come in.
    // It should be noted that there is a limit to the number of threads in
    // a block and the number of blocks in x, y and z directions; the
    // z direction being more limited than the former two.
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Each thread increments the array element corresponding to its global
    // index by one.
    if(idx < n)
        y[idx] = a * x[idx] + y[idx];
}

// =====================================================================
// =====================================================================
// The host program
int main(int argc, char** argv)
{

    // Most HIP calls return an error code of type hipError_t
    hipError_t err;

    // Initializing the random number generator
    std::default_random_engine gen;
    std::uniform_real_distribution<> dist(-9999, 9999);

    // ================================================================
    // Let's check that we have devices available
    // [OPTIONAL]
    int deviceCount = 0;
    err = hipGetDeviceCount(&deviceCount);

    // Check the error code. hipSuccess is the generic "all OK" return value.
    if (err != hipSuccess) {
        std::cerr << "Error getting a device count." << std::endl;
        std::cerr << "hipError-code: " << err << std::endl;
        std::cerr << "hipError-string: " << hipGetErrorString(err) << std::endl;
        return 1;
    }

    // ================================================================
    // Let's query HIP for some version information
    // [OPTIONAL]

    // Get HIP runtime version
    int runtimeVersion;
    hipError_t err = hipRuntimeGetVersion(&runtimeVersion);
    if (err != hipSuccess) {
        std::cerr << "Failed to query the HIP runtime version." << std::endl;
        std::cerr << "hipError-code: " << err << std::endl;
        std::cerr << "hipError-string: " << hipGetErrorString(err) << std::endl;
        return 1;
    }

    std::cout << "HIP Runtime Version: " << runtimeVersion << std::endl;

    // Get HIP driver version
    int driverVersion;
    err = hipDriverGetVersion(&driverVersion);
    if (err != hipSuccess) {
        std::cerr << "Failed to query the HIP driver version." << std::endl;
        std::cerr << "hipError-code: " << err << std::endl;
        std::cerr << "hipError-string: " << hipGetErrorString(err) << std::endl;
        return 1;
    }

    std::cout << "HIP Driver Version: " << driverVersion << std::endl;

    // ================================================================
    // Iterate through all found devices
    // [OPTIONAL]

    // Get devices
    std::vector<hipDevice_t> devices(deviceCount);
    for (int i = 0; i < deviceCount; i++) {
        err = hipDeviceGet(&devices[i], i);
        if (err != hipSuccess) {
            std::cerr << "Error getting device " << i << std::endl;
            std::cerr << "hipError-code: " << err << std::endl;
            std::cerr << "hipError-string: " << hipGetErrorString(err) << std::endl;
            return 1;
        }
    }

    // Get device names
    for (int i = 0; i < deviceCount; i++) {
        char deviceName[256];
        err = hipDeviceGetName(deviceName, 256, devices[i]);
        ir (err != hipSuccess) {
            std::cerr << "Error getting device name for device " << i << std::endl;
            std::cerr << "hipError-code: " << err << std::endl;
            std::cerr << "hipError-string: " << hipGetErrorString(err) << std::endl;
            return 1;
        }
        std::cout << "Device " << i << ": " << deviceName << std::endl;
    }

    // Get device compute capabilities
    for (int i = 0; i < deviceCount; i++) {
        int major, minor;
        err = hipDeviceComputeCapability(&major, &minor, devices[i]);
        if (err != hipSuccess) {
            std::cerr << "Error getting device compute capability for device " << i << std::endl;
            std::cerr << "hipError-code: " << err << std::endl;
            std::cerr << "hipError-string: " << hipGetErrorString(err) << std::endl;
            return 1;
        }
        std::cout << "Compute Capability: " << major << "." << minor << std::endl;
    }

    // ================================================================
    // Set a device
    // [OPTIONAL] Device 0 is the default device.

    // Note that hipDevice_t is simply an int under the hood, so
    // hipSetDevice(0) would also work.
    err = hipSetDevice(devices[0]);
    if (err != hipSuccess) {
        std::cerr << "Failed to select device " << 0 << std::endl;
        std::cerr << "hipError-code: " << err << std::endl;
        std::cerr << "hipError-string: " << hipGetErrorString(err) << std::endl;
        return 1;
    }

    // ================================================================
    // Create a HIP stream
    // [OPTIONAL] !!! But without you are running everything on the 0 stream
    //            !!! which might enforce undesirable synchronization.
    //            !!! It is for example possible to have a stream where
    //            !!! memory is being copied to the device in preparation
    //            !!! of some later computation simultaneously with other
    //            !!! ongoing computations on the device, by using the
    //            !!! hipMallocAsync and hipMemcpyAsync with a dedicated
    //            !!! stream which the copute kernels are queued up on
    //            !!! another stream.

    // Note that there are a multitude of ways to create streams with different
    // characteristics. Check out the HIP documentation to learn more.
    std::cout << "Creating HIP stream." << std::endl;
    hipStream_t stream;
    err = hipStreamCreate(&stream);
    if (err != hipSuccess) {
        std::cerr << "Failed to create HIP stream." << std::endl;
        std::cerr << "hipError-code: " << err << std::endl;
        std::cerr << "hipError-string: " << hipGetErrorString(err) << std::endl;
        return 1;
    }

    // ================================================================
    // Check available global memory size
    // [OPTIONAL]
    size_t globalMemorySize;
    err = hipDeviceTotalMem(&globalMemSize, 0);
    if (err != hipSuccess) {
        std::cerr << "Failed to query the global memory size of the HIP device." << std::endl;
        std::cerr << "hipError-code: " << err << std::endl;
        std::cerr << "hipError-string: " << hipGetErrorString(err) << std::endl;
        return 1;
    }

    std::cout << "Global memory available in the HIP device: " << globalMemSize / 1.0E6 << " MB." << std::endl;

    // Compare with required memory size (example: N * sizeof(int))
    size_t requiredMemSize = N * sizeof(int);
    if (globalMemSize < requiredMemSize) {
        std::cout << "Insufficient global memory in the HIP device." << std::endl;
        return 1;
    }

    // ================================================================
    // Allocate space in global memory
    std::cout << "Allocating " << N * sizeof(int) / 1.0e6 << " MB of global memory." << std::endl;
    double *device_x; // Pointer for GPU-memory location
    double *device_y;
    err = hipMalloc(&device_x, N  * sizeof(double));
    err = hipMalloc(&device_y, N  * sizeof(double)); // allocating memory to GPU
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate memory." << std::endl;
        std::cerr << "hipError-code: " << err << std::endl;
        std::cerr << "hipError-string: " << hipGetErrorString(err) << std::endl;
        return 1;
    }

    // ================================================================
    // Create arrays of integers
    std::vector<double> hostBuffer_x(N);
    std::vector<double> hostBuffer_y(N);
    double a = dist(gen); // random a for axpy
    for (int i = 0; i < N; i++) {
        hostBuffer_x[i] = dist(gen);
        hostBuffer_y[i] = dist(gen);
    }
    std::vector<double> checkBuffer(hostBuffer_y);

    // ================================================================
    // Transfer buffer to device memory
    std::cout << "Copying " << N * sizeof(int) / 1.0e6 << " MB from host to device." << std::endl;
    err = hipMemcpy(device_x, hostBuffer_x.data(), N * sizeof(double), hipMemcpyDefault);
    err = hipMemcpy(device_y, hostBuffer_y.data(), N * sizeof(double), hipMemcpyDefault);
    // Note that the last argument may also be hipMemcpyHostToDevice and
    // hipMemcpyDeviceToHost. The "Default" option here automatically
    // detects what kind of transfer we are doing. This is nice, but also
    // means that the compiler won't complain to us if we accidentally
    // copy data to a host-address instead of a device-address by mistake,
    // so use with care.
    if (err != hipSuccess) {
        std::cerr << "Failed to copy memory to device." << std::endl;
        std::cerr << "hipError-code: " << err << std::endl;
        std::cerr << "hipError-string: " << hipGetErrorString(err) << std::endl;

        hipFree(device_x);
        hipFree(device_y);
        return 1;
    }

    // ================================================================
    // Query maximum work group size
    // [OPTIONAL] ::: This size is typically 1024 threads, note that this is
    //            ::: a total limit, not per "direction"; so if you have 32
    //            ::: in the x-direction, then you can have at most e.g. 32
    //            ::: in the y-direcetion and 1 in the z-direction.
    int maxWorkGroupSize;
    err = hipDeviceGetAttribute(&maxWorkGroupSize, hipDeviceAttributeMaxThreadsPerBlock, 0);
    if (err != hipSuccess) {
        std::cerr << "Failed to get maximum work group size." << std::endl;
        std::cerr << "hipError-code: " << err << std::endl;
        std::cerr << "hipError-string: " << hipGetErrorString(err) << std::endl;

        hipFree(device_x);
        hipFree(device_y);
        return 1;
    }
    std::cout << "Maximum work group size: " << maxWorkGroupSize << " threads." << std::endl;

    // ================================================================
    // Configure blocks and threads (work-groups and work-items)
    const dim3 numberOfBlocks((N - 1)/maxWorkGroupSize + 1);
    const dim3 threadsPerBlock(maxWorkGroupSize);
    // When we don't populate all 3 arguments for dim3 the later ones
    // are 1 by default.

    // ================================================================
    // Call the kernel
    std::cout << "Calling kernel." << std::endl;
    err = axpy<<<numberOfBlocks, threadsPerBlock, 0, stream>>>(device_x, device_y, N, a);
    // The 0 is the amount of shared memory given to the kernel.
    // Specifying shared memory and the stream is optional, if the
    // stream isn't set, the deault stream is the 0 stream, which
    // enforces a special synchronization.
    if (err != hipSuccess) {
        std::cerr << "Failed to invoke the kernel." << std::endl;
        std::cerr << "hipError-code: " << err << std::endl;
        std::cerr << "hipError-string: " << hipGetErrorString(err) << std::endl;

        hipFree(device_x);
        hipFree(device_y);
        return 1;
    }

    // ================================================================
    // Synchronize stream
    std::cout << "Synchronizing HIP stream." << std::endl;
    err = hipStreamSynchronize(stream);
    if (err != hipSuccess) {
        std::cerr << "Failed to synchronize stream." << std::endl;
        std::cerr << "hipError-code: " << err << std::endl;
        std::cerr << "hipError-string: " << hipGetErrorString(err) << std::endl;

        hipFree(device_x);
        hipFree(device_y);
        return 1;
    }

    // ================================================================ sudaaaaaaaaaaaaaaaaa
    // Get the result from the device
    // Transfer buffer to device memory
    std::cout << "Copying " << N * sizeof(int) / 1.0e6 << " MB from device to host." << std::endl;
    err = hipMemcpy(hostBuffer_y.data(), device_y, N * sizeof(int), hipMemcpyDefault);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy memory from device." << std::endl;
        std::cerr << "hipError-code: " << err << std::endl;
        std::cerr << "hipError-string: " << hipGetErrorString(err) << std::endl;

        hipFree(device_x);
        hipFree(device_y);
        return 1;
    }

    // ================================================================
	// Check the result
	int rightValues = 0;
	for(int i = 0; i < N; i++) {
	    double result = a * hostBuffer_x[i] + hostBuffer_y[i];
		if(hostBuffer_y[i] == result)
			rightValues++;
	}

    std::cout << "Result " << 100.0 * rightValues / N << "% correct." << std::endl;

    // ================================================================
    // Clean up
    hipFree(device_x);
    hipFree(device_y);
    hipStreamDestroy(stream);

    return 0;

}