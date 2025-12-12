#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>

#define CSC(call)  									                \
do {											                    \
	cudaError_t res = call;							                \
	if (res != cudaSuccess) {							            \
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);								                    \
	}										                        \
} while(0)

struct Pixel { uint8_t r,g,b,a; };
struct F3    { float   x,y,z; };
struct U3    { uint8_t x,y,z; };

__constant__ F3 CLASS_AVG_UNIT[32];
__constant__ U3 CLASS_AVG_RGB [32];

__global__ void classifyLinear(const Pixel* __restrict__ in,
                               Pixel*       __restrict__ out,
                               int n, int w, int nc)
{
    int i    = blockIdx.x * blockDim.x + threadIdx.x;
    int step = blockDim.x * gridDim.x;

    for (; i < n; i += step) {
        Pixel px = in[i];
        float rx = px.r, gy = px.g, bz = px.b;
        float lp = sqrtf(rx*rx + gy*gy + bz*bz);

        int   best    = 0;
        float bestVal = -1e30f;
        if (lp > 0.f) {
            #pragma unroll
            for (int c = 0; c < nc; ++c) {
                F3 a = CLASS_AVG_UNIT[c];
                float val = (rx*a.x + gy*a.y + bz*a.z) / lp;
                if (val > bestVal) { bestVal = val; best = c; }
            }
        }

        out[i] = Pixel{ px.r, px.g, px.b, (uint8_t)best };
    }
}

static void readData(const std::string& path, int& w, int& h, std::vector<Pixel>& img) {
    std::ifstream is(path, std::ios::binary);
    if (!is) { std::fprintf(stderr,"open fail\n"); std::exit(1); }
    is.read((char*)&w,4);
    is.read((char*)&h,4);
    img.resize((size_t)w*(size_t)h);
    is.read((char*)img.data(), img.size()*sizeof(Pixel));
}

static void writeData(const std::string& path, int w, int h, const std::vector<Pixel>& img) {
    std::ofstream os(path, std::ios::binary|std::ios::trunc);
    if (!os) { std::fprintf(stderr,"write fail\n"); std::exit(1); }
    os.write((const char*)&w,4);
    os.write((const char*)&h,4);
    os.write((const char*)img.data(), img.size()*sizeof(Pixel));
}

int main() {
    std::ios::sync_with_stdio(false);

    std::string inPath, outPath;
    if (!(std::cin >> inPath >> outPath)) return 0;

    int nc; 
    if (!(std::cin >> nc) || nc<=0 || nc>32) return 0;

    int w=0, h=0;
    std::vector<Pixel> host;
    readData(inPath, w, h, host);
    const int n = w*h;

    std::vector<F3> unit(nc);
    std::vector<U3> rgb (nc);

    for (int c=0; c<nc; ++c) {
        int np; std::cin >> np;
        double sx=0, sy=0, sz=0;
        for (int k=0; k<np; ++k) {
            int x,y; std::cin >> x >> y;
            const Pixel& p = host[(size_t)y*(size_t)w + (size_t)x];
            sx += p.r; sy += p.g; sz += p.b;
        }
        double inv = 1.0 / np;
        double mx = sx*inv, my = sy*inv, mz = sz*inv;

        int R = (int)std::lround(mx); if (R<0) R=0; if (R>255) R=255;
        int G = (int)std::lround(my); if (G<0) G=0; if (G>255) G=255;
        int B = (int)std::lround(mz); if (B<0) B=0; if (B>255) B=255;
        rgb[c] = U3{ (uint8_t)R, (uint8_t)G, (uint8_t)B };

        double L = std::sqrt(mx*mx + my*my + mz*mz);
        unit[c] = (L>0.0) ? F3{ float(mx/L), float(my/L), float(mz/L) } : F3{0,0,0};
    }

    CSC(cudaMemcpyToSymbol(CLASS_AVG_UNIT, unit.data(), nc*sizeof(F3)));
    CSC(cudaMemcpyToSymbol(CLASS_AVG_RGB , rgb .data(), nc*sizeof(U3)));

    Pixel *din=nullptr, *dout=nullptr;
    size_t bytes = (size_t)n * sizeof(Pixel);
    CSC(cudaMalloc(&din,  bytes));
    CSC(cudaMalloc(&dout, bytes));
    CSC(cudaMemcpy(din, host.data(), bytes, cudaMemcpyHostToDevice));

    const dim3 block(256,1,1);
    const dim3 grid (8192,1,1);
    classifyLinear<<<grid, block>>>(din, dout, n, w, nc);
    CSC(cudaGetLastError());
    CSC(cudaDeviceSynchronize());

    std::vector<Pixel> out(n);
    CSC(cudaMemcpy(out.data(), dout, bytes, cudaMemcpyDeviceToHost));
    writeData(outPath, w, h, out);

    cudaFree(din);
    cudaFree(dout);
    return 0;
}
