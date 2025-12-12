#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

static const int BLOCKS  = 64;
static const int THREADS = 512;

#define CSC(call)  									                \
do {											                    \
	cudaError_t res = call;							                \
	if (res != cudaSuccess) {							            \
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);								                    \
	}										                        \
} while(0)

__global__ void elem_min_kernel(const double* a, const double* b, double* out, long long n) {
    long long idx = blockIdx.x * 1ll * blockDim.x + threadIdx.x;
    long long step = 1ll * gridDim.x * blockDim.x;
    for (long long i = idx; i < n; i += step) {
        double va = a[i], vb = b[i];
        out[i] = (va < vb) ? va : vb;
    }
}

int main() {
    long long n; std::scanf("%lld", &n);

    size_t bytes = (size_t)n * sizeof(double);
    double* h_a   = (double*)std::malloc(bytes);
    double* h_b   = (double*)std::malloc(bytes);
    double* h_out = (double*)std::malloc(bytes);

    for (long long i = 0; i < n; ++i) std::scanf("%lf", &h_a[i]);
    for (long long i = 0; i < n; ++i) std::scanf("%lf", &h_b[i]);

    double *d_a=nullptr, *d_b=nullptr, *d_out=nullptr;
    CSC(cudaMalloc(&d_a, bytes));
    CSC(cudaMalloc(&d_b, bytes));
    CSC(cudaMalloc(&d_out, bytes));

    CSC(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CSC(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    elem_min_kernel<<<BLOCKS, THREADS>>>(d_a, d_b, d_out, n);
    CSC(cudaGetLastError());
    CSC(cudaDeviceSynchronize());

    CSC(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    CSC(cudaFree(d_a)); CSC(cudaFree(d_b)); CSC(cudaFree(d_out));
    std::free(h_a); std::free(h_b);

    for (long long i = 0; i < n; ++i) { if (i) std::printf(" "); std::printf("%.10e", h_out[i]); }
    std::printf("\n");

    std::free(h_out);
    return 0;
}
