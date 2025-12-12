#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <iomanip>

#define CSC(call)  									                \
do {											                    \
	cudaError_t res = call;							                \
	if (res != cudaSuccess) {							            \
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);								                    \
	}										                        \
} while(0)

struct AbsLess {
    __host__ __device__ bool operator()(double a, double b) const {
        return fabs(a) < fabs(b);
    }
};

__device__ inline void swap_colmajor(double* a, int n, int row1, int row2, int col) {
    double t = a[col * n + row1];
    a[col * n + row1] = a[col * n + row2];
    a[col * n + row2] = t;
}

__global__ void row_swap_kernel(double* a, int n, int r1, int r2) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int step = blockDim.x * gridDim.x;
    for (int col = tid; col < n + 1; col += step) {
        swap_colmajor(a, n, r1, r2, col);
    }
}

__global__ void eliminate_kernel(double* a, int n, int col) {
    int j0 = blockIdx.x * blockDim.x + threadIdx.x; 
    int i0 = blockIdx.y * blockDim.y + threadIdx.y; 
    int sx = blockDim.x * gridDim.x;
    int sy = blockDim.y * gridDim.y;

    for (int j = col + 1 + j0; j < n; j += sx) {
        double piv = a[col * n + col];
        double k   = a[col * n + j] / piv;
        for (int i = col + 1 + i0; i < n + 1; i += sy) {
            a[i * n + j] -= k * a[i * n + col];
        }
    }
}

static void read_augmented_colmajor(double* a, int n) {
    for (int row = 0; row < n; ++row)
        for (int col = 0; col < n; ++col) {
            double v; std::cin >> v;
            a[col * n + row] = v;
        }

    for (int row = 0; row < n; ++row) {
        double v; std::cin >> v;
        a[n * n + row] = v;
    }
}

static void back_substitute_colmajor(const double* a, int n, double* x) {
    for (int i = n - 1; i >= 0; --i) {
        double s = a[n * n + i]; 
        for (int j = n - 1; j > i; --j)
            s -= x[j] * a[j * n + i]; 
        x[i] = s / a[i * n + i];      
    }
}

static void print_solution(const double* x, int n) {
    std::cout << std::fixed << std::setprecision(10);
    for (int i = 0; i < n; ++i) {
        std::cout << x[i] << (i + 1 == n ? '\n' : ' ');
    }
}

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    int n;
    if (!(std::cin >> n) || n <= 0) return 0;

    const size_t bytes = static_cast<size_t>(n) * (n + 1) * sizeof(double);
    double* hA = static_cast<double*>(std::malloc(bytes));
    if (!hA) return 0;

    read_augmented_colmajor(hA, n);

    double* dA = nullptr;
    CSC(cudaMalloc(&dA, bytes));
    CSC(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));

    thrust::device_ptr<double> base = thrust::device_pointer_cast(dA);
    AbsLess absLess;

    dim3 gridSwap(256, 1);
    dim3 blockSwap(256, 1);
    dim3 gridElim(32, 32);
    dim3 blockElim(16, 16);

    for (int col = 0; col < n - 1; ++col) {
        const int colOffset = col * n;
        const int pivotRow  = static_cast<int>(
            thrust::max_element(base + colOffset + col, base + colOffset + n, absLess)
            - (base + colOffset)
        );

        if (pivotRow != col) {
            row_swap_kernel<<<gridSwap, blockSwap>>>(dA, n, col, pivotRow);
            CSC(cudaGetLastError());
        }

        eliminate_kernel<<<gridElim, blockElim>>>(dA, n, col);
        CSC(cudaGetLastError());
    }
    CSC(cudaDeviceSynchronize());

    CSC(cudaMemcpy(hA, dA, bytes, cudaMemcpyDeviceToHost));
    CSC(cudaFree(dA));

    double* x = static_cast<double*>(std::malloc(n * sizeof(double)));
    if (!x) { std::free(hA); return 0; }

    back_substitute_colmajor(hA, n, x);
    print_solution(x, n);

    std::free(hA);
    std::free(x);
    return 0;
}
