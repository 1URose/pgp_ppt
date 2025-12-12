#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cmath>

struct Pixel { 
    uint8_t r, g, b, a; 
};

__constant__ int8_t Gx[9] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
__constant__ int8_t Gy[9] = { -1,-2,-1,  0, 0, 0,  1, 2, 1 };

__device__ __forceinline__ float luma(uchar4 p) {
    return 0.299f*p.x + 0.587f*p.y + 0.114f*p.z;
}

__global__ void sobel(cudaTextureObject_t tex, Pixel* out, int w, int h) {
    const int sx = blockIdx.x * blockDim.x + threadIdx.x;
    const int sy = blockIdx.y * blockDim.y + threadIdx.y;
    const int stepx = blockDim.x * gridDim.x;
    const int stepy = blockDim.y * gridDim.y;

    for (int y = sy; y < h; y += stepy)
        for (int x = sx; x < w; x += stepx) {
            float gx = 0.f, gy = 0.f; int k = 0;
            for (int j = -1; j <= 1; ++j)
                for (int i = -1; i <= 1; ++i, ++k) {
                    uchar4 c = tex2D<uchar4>(tex, float(x+i)+0.5f, float(y+j)+0.5f);
                    float Y = luma(c);
                    gx += float(Gx[k]) * Y;
                    gy += float(Gy[k]) * Y;
                }
            float m = sqrtf(gx*gx + gy*gy);
            if (m > 255.f) m = 255.f;
            unsigned char g = (unsigned char)(m + 0.5f);
            out[size_t(y)*size_t(w) + size_t(x)] = Pixel{g,g,g,0u};
        }
}

static inline void readData(const std::string& p, int& w, int& h, std::vector<Pixel>& a) {
    std::ifstream is(p, std::ios::binary);
    is.read((char*)&w, 4); is.read((char*)&h, 4);
    a.resize(size_t(w)*size_t(h));
    is.read((char*)a.data(), a.size()*sizeof(Pixel));
}

static inline void writeData(const std::string& p, int w, int h, const std::vector<Pixel>& a) {
    std::ofstream os(p, std::ios::binary | std::ios::trunc);
    os.write((const char*)&w, 4); os.write((const char*)&h, 4);
    os.write((const char*)a.data(), a.size()*sizeof(Pixel));
}

#define CSC(call)  									                \
do {											                    \
	cudaError_t res = call;							                \
	if (res != cudaSuccess) {							            \
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);								                    \
	}										                        \
} while(0)

int main() {
    std::string inPath, outPath; std::cin >> inPath >> outPath;

    int w=0, h=0; std::vector<Pixel> host;
    readData(inPath, w, h, host);

    cudaArray_t arr; auto desc = cudaCreateChannelDesc<uchar4>();
    CSC(cudaMallocArray(&arr, &desc, w, h));
    CSC(cudaMemcpy2DToArray(arr, 0, 0, host.data(), w*sizeof(Pixel), w*sizeof(Pixel), h, cudaMemcpyHostToDevice));

    cudaResourceDesc r{}; r.resType = cudaResourceTypeArray; r.res.array.array = arr;
    cudaTextureDesc  t{}; t.addressMode[0]=cudaAddressModeClamp; t.addressMode[1]=cudaAddressModeClamp;
    t.filterMode=cudaFilterModePoint; t.readMode=cudaReadModeElementType; t.normalizedCoords=0;
    cudaTextureObject_t tex=0; CSC(cudaCreateTextureObject(&tex, &r, &t, nullptr));

    Pixel* devOut=nullptr; size_t bytes = size_t(w)*size_t(h)*sizeof(Pixel);
    CSC(cudaMalloc(&devOut, bytes));

    dim3 block(16,16), grid(128,64);
    sobel<<<grid,block>>>(tex, devOut, w, h);
    CSC(cudaGetLastError()); CSC(cudaDeviceSynchronize());

    std::vector<Pixel> out(size_t(w)*size_t(h));
    CSC(cudaMemcpy(out.data(), devOut, bytes, cudaMemcpyDeviceToHost));

    writeData(outPath, w, h, out);

    cudaDestroyTextureObject(tex); cudaFreeArray(arr); cudaFree(devOut);
    return 0;
}
