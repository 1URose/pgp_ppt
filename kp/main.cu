// main.cu
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define CUDA_OK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        std::exit(EXIT_FAILURE); \
    } \
} while(0)

namespace cfg {
    static constexpr int MAX_LIGHTS = 4;

    static constexpr double AMBIENT = 0.04;        
    static constexpr bool   USE_ATTENUATION = true;
    static constexpr double ATT_K = 0.02;         
    static constexpr bool   DRAW_EDGES = true;
    static constexpr double EDGE_EPS = 0.05;

    static constexpr int BLOCK_X = 16;
    static constexpr int BLOCK_Y = 16;
    static constexpr int GRID_X  = 8;
    static constexpr int GRID_Y  = 8;

    static constexpr double LIGHT_INTENSITY = 2.5;

    struct Config {
        int framesCount;
        const char* outPattern;
        int width;
        int height;
        double fovDeg;

        double r0c, z0c, phi0c, arc, azc, wrc, wzc, wphic, prc, pzc;

        double r0n, z0n, phi0n, arn, azn, wrn, wzn, wphin, prn, pzn;

        double b1x,b1y,b1z, b1r,b1g,b1b, b1rad, b1Refl, b1Trans; int b1EdgeLights;
        double b2x,b2y,b2z, b2r,b2g,b2b, b2rad, b2Refl, b2Trans; int b2EdgeLights;
        double b3x,b3y,b3z, b3r,b3g,b3b, b3rad, b3Refl, b3Trans; int b3EdgeLights;


        double fax,fay,faz, fbx,fby,fbz, fcx,fcy,fcz, fdx,fdy,fdz;
        const char* floorTexPath;
        double ftr, ftg, ftb;
        double floorRefl;        

        int lightsCount;
        double lx[MAX_LIGHTS], ly[MAX_LIGHTS], lz[MAX_LIGHTS];
        double lr[MAX_LIGHTS], lg[MAX_LIGHTS], lb[MAX_LIGHTS];

        int maxDepth;
        int sqrtRpp; 
    };

    static const Config CFG = {
        // frames
        120,
        "res/%d.data",
        800, 400,
        90.0,

        // camera position curve
        7., 3., 0.,   2., 1., 2., 6., 1., 0., 0.,
        // camera target curve
        2., 0., 0.,   0.5, 0.1, 1., 4., 1., 0., 0.,

        // body1 (cube)
        0., -2., 0.,   1.0, 0.78, 0.0,   1.0,   0.0, 0.0, 0,
        // body2 (octa)
        0.,  0., 0.,   0.0, 1.0,  0.7,   1.0,   0.0, 0.0, 0,
        // body3 (icosa)
        0.,  2., 0.,   0.47, 0.70, 1.0,  1.0,   0.0, 0.0, 0,

        // floor
        -5., -5., -1.,
        -5.,  5., -1.,
         5.,  5., -1.,
         5., -5., -1.,
        "floor.data",
        0.0, 1.0, 0.0,
        0.0,

        // lights
        1,                           
        {  8.0,  -6.0,  12.0, 0.0 }, 
        {  0.0,   0.0,   0.0, 0.0 }, 
        { 10.0,   0.0,   0.0, 0.0 }, 
        {  1.0,   0.0,   0.0, 0.0 }, 
        {  1.0,   0.0,   0.0, 0.0 }, 
        {  1.0,   0.0,   0.0, 0.0 }, 

        // limits
        10,
        4
    };

    inline void printDefault(std::ostream& out) {
        const auto& c = CFG;
        out << c.framesCount << "\n";
        out << c.outPattern << "\n";
        out << c.width << " " << c.height << " " << c.fovDeg << "\n";

        out << c.r0c << " " << c.z0c << " " << c.phi0c << " "
            << c.arc << " " << c.azc << " " << c.wrc << " "
            << c.wzc << " " << c.wphic << " " << c.prc << " " << c.pzc << "\n";

        out << c.r0n << " " << c.z0n << " " << c.phi0n << " "
            << c.arn << " " << c.azn << " " << c.wrn << " "
            << c.wzn << " " << c.wphin << " " << c.prn << " " << c.pzn << "\n";

        auto pb = [&](double x,double y,double z,double r,double g,double b,double rad,double refl,double tr,int edgeN) {
            out << x << " " << y << " " << z << " "
                << r << " " << g << " " << b << " "
                << rad << " "
                << refl << " " << tr << " " << edgeN << "\n";
        };
        pb(c.b1x,c.b1y,c.b1z,c.b1r,c.b1g,c.b1b,c.b1rad,c.b1Refl,c.b1Trans,c.b1EdgeLights);
        pb(c.b2x,c.b2y,c.b2z,c.b2r,c.b2g,c.b2b,c.b2rad,c.b2Refl,c.b2Trans,c.b2EdgeLights);
        pb(c.b3x,c.b3y,c.b3z,c.b3r,c.b3g,c.b3b,c.b3rad,c.b3Refl,c.b3Trans,c.b3EdgeLights);

        out << c.fax << " " << c.fay << " " << c.faz << " "
            << c.fbx << " " << c.fby << " " << c.fbz << " "
            << c.fcx << " " << c.fcy << " " << c.fcz << " "
            << c.fdx << " " << c.fdy << " " << c.fdz << " "
            << c.floorTexPath << " "
            << c.ftr << " " << c.ftg << " " << c.ftb << " "
            << c.floorRefl << "\n";

        out << c.lightsCount << "\n";
        for (int i = 0; i < c.lightsCount && i < MAX_LIGHTS; ++i) {
            out << c.lx[i] << " " << c.ly[i] << " " << c.lz[i] << " "
                << c.lr[i] << " " << c.lg[i] << " " << c.lb[i] << "\n";
        }

        out << c.maxDepth << " " << c.sqrtRpp;
    }
}
template <class T>
struct DeviceBuf {
    T* ptr = nullptr;
    size_t count = 0;

    DeviceBuf() = default;
    explicit DeviceBuf(size_t n) { alloc(n); }

    void alloc(size_t n) {
        free();
        count = n;
        CUDA_OK(cudaMalloc(&ptr, sizeof(T) * n));
    }
    void free() {
        if (ptr) CUDA_OK(cudaFree(ptr));
        ptr = nullptr;
        count = 0;
    }
    ~DeviceBuf() { free(); }

    DeviceBuf(const DeviceBuf&) = delete;
    DeviceBuf& operator=(const DeviceBuf&) = delete;

    DeviceBuf(DeviceBuf&& other) noexcept {
        ptr = other.ptr; count = other.count;
        other.ptr = nullptr; other.count = 0;
    }
    DeviceBuf& operator=(DeviceBuf&& other) noexcept {
        if (this != &other) {
            free();
            ptr = other.ptr; count = other.count;
            other.ptr = nullptr; other.count = 0;
        }
        return *this;
    }
};

struct Vec3 {
    double x = 0, y = 0, z = 0;

    __host__ __device__ Vec3() = default;
    __host__ __device__ Vec3(double xx, double yy, double zz) : x(xx), y(yy), z(zz) {}

    __host__ __device__ Vec3 operator+(const Vec3& r) const { return {x + r.x, y + r.y, z + r.z}; }
    __host__ __device__ Vec3 operator-(const Vec3& r) const { return {x - r.x, y - r.y, z - r.z}; }
    __host__ __device__ Vec3 operator*(double k) const { return {x * k, y * k, z * k}; }
};

__host__ __device__ inline double vdot(const Vec3& a, const Vec3& b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}
__host__ __device__ inline Vec3 vcross(const Vec3& a, const Vec3& b) {
    return Vec3(
        a.y*b.z - a.z*b.y,
        a.z*b.x - a.x*b.z,
        a.x*b.y - a.y*b.x
    );
}
__host__ __device__ inline double vlen(const Vec3& v) { return ::sqrt(vdot(v,v)); }
__host__ __device__ inline Vec3 vnorm(const Vec3& v) {
    double L = vlen(v);
    return (L > 0.0) ? (v * (1.0 / L)) : Vec3(0,0,0);
}
__host__ __device__ inline Vec3 basisMul(const Vec3& bx, const Vec3& by, const Vec3& bz, const Vec3& v) {
    return Vec3(
        bx.x*v.x + by.x*v.y + bz.x*v.z,
        bx.y*v.x + by.y*v.y + bz.y*v.z,
        bx.z*v.x + by.z*v.y + bz.z*v.z
    );
}
__host__ __device__ inline int clampi(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}
__host__ __device__ inline unsigned char toU8(int v) {
    return (unsigned char)clampi(v, 0, 255);
}
__host__ __device__ inline double clamp01(double x) {
    return x < 0.0 ? 0.0 : (x > 1.0 ? 1.0 : x);
}

__host__ __device__ inline uchar4 rgb01_to_uchar4(double r, double g, double b) {
    int R = (int)::llround(r * 255.0);
    int G = (int)::llround(g * 255.0);
    int B = (int)::llround(b * 255.0);
    return make_uchar4(toU8(R), toU8(G), toU8(B), 255);
}

struct Tri {
    Vec3 p0, p1, p2;
    uchar4 albedo;
    __host__ __device__ Tri() = default;
    __host__ __device__ Tri(const Vec3& a, const Vec3& b, const Vec3& c, uchar4 col)
        : p0(a), p1(b), p2(c), albedo(col) {}
};

struct Light {
    Vec3 pos;
    uchar4 color;
};

__host__ __device__ inline bool hitTriangle(
    const Vec3& ro, const Vec3& rd,
    const Tri& t,
    double& outT, double& outU, double& outV
) {
    const double EPS = 1e-10;
    Vec3 e1 = t.p1 - t.p0;
    Vec3 e2 = t.p2 - t.p0;

    Vec3 p = vcross(rd, e2);
    double det = vdot(e1, p);
    if (::fabs(det) < EPS) return false;

    double invDet = 1.0 / det;
    Vec3 s = ro - t.p0;
    double u = vdot(s, p) * invDet;
    if (u < 0.0 || u > 1.0) return false;

    Vec3 q = vcross(s, e1);
    double v = vdot(rd, q) * invDet;
    if (v < 0.0 || (u + v) > 1.0) return false;

    double tt = vdot(e2, q) * invDet;
    if (tt <= 0.0) return false;

    outT = tt; outU = u; outV = v;
    return true;
}

__host__ __device__ inline int findClosestHit(
    const Vec3& ro, const Vec3& rd,
    const Tri* tris, int n,
    double& tMin, double& uMin, double& vMin
) {
    int best = -1;
    tMin = 0; uMin = 0; vMin = 0;

    for (int i = 0; i < n; ++i) {
        double t, u, v;
        if (!hitTriangle(ro, rd, tris[i], t, u, v)) continue;
        if (best < 0 || t < tMin) {
            best = i;
            tMin = t;
            uMin = u;
            vMin = v;
        }
    }
    return best;
}

__host__ __device__ inline bool inShadow(
    const Vec3& p, const Vec3& toLightDir, double lightDist,
    const Tri* tris, int n,
    int ignoreIdx
) {
    const double bias = 1e-6;
    Vec3 ro = p + toLightDir * bias;

    for (int i = 0; i < n; ++i) {
        if (i == ignoreIdx) continue;
        double t, u, v;
        if (!hitTriangle(ro, toLightDir, tris[i], t, u, v)) continue;
        if (t > 0.0 && t < lightDist) return true;
    }
    return false;
}

__host__ __device__ inline uchar4 shadeDiffuse(
    const Vec3& ro, const Vec3& rd,
    const Light* lights, int lightCount,
    const Tri* tris, int nTris
) {
    double tMin, u, v;
    int idx = findClosestHit(ro, rd, tris, nTris, tMin, u, v);
    if (idx < 0) return make_uchar4(0, 0, 0, 255);

    const Tri& tr = tris[idx];
    Vec3 p = ro + rd * tMin;

    Vec3 e1 = tr.p1 - tr.p0;
    Vec3 e2 = tr.p2 - tr.p0;
    Vec3 nrm = vnorm(vcross(e1, e2));
    if (vdot(nrm, rd) > 0.0) nrm = nrm * -1.0;

    if (cfg::DRAW_EDGES) {
        double w = 1.0 - u - v;
        bool isEdge = (u < cfg::EDGE_EPS || v < cfg::EDGE_EPS || w < cfg::EDGE_EPS);
        if (idx >= 2 && isEdge) return make_uchar4(0, 0, 0, 255);
    }

    double sumR = 0.0, sumG = 0.0, sumB = 0.0;

    int L = lightCount;
    if (L > cfg::MAX_LIGHTS) L = cfg::MAX_LIGHTS;

    for (int i = 0; i < L; ++i) {
        Vec3 Lvec = lights[i].pos - p;
        double dist = vlen(Lvec);
        if (dist <= 0.0) continue;
        Vec3 ldir = Lvec * (1.0 / dist);

        if (inShadow(p, ldir, dist, tris, nTris, idx)) continue;

        double lam = vdot(nrm, ldir);
        if (lam <= 0.0) continue;

        double att = 1.0;
        if (cfg::USE_ATTENUATION) att = 1.0 / (1.0 + cfg::ATT_K * dist * dist);

        double k = cfg::LIGHT_INTENSITY * lam * att;

        sumR += k * ((double)lights[i].color.x / 255.0);
        sumG += k * ((double)lights[i].color.y / 255.0);
        sumB += k * ((double)lights[i].color.z / 255.0);
    }

    double Ir = clamp01(cfg::AMBIENT + sumR);
    double Ig = clamp01(cfg::AMBIENT + sumG);
    double Ib = clamp01(cfg::AMBIENT + sumB);

    int r = (int)::llround((double)tr.albedo.x * Ir);
    int g = (int)::llround((double)tr.albedo.y * Ig);
    int b = (int)::llround((double)tr.albedo.z * Ib);

    return make_uchar4(toU8(r), toU8(g), toU8(b), 255);
}

// ============================================================================
//                                    CAMERA
// ============================================================================
__host__ __device__ inline void buildCameraBasis(const Vec3& eye, const Vec3& at, Vec3& bx, Vec3& by, Vec3& bz) {
    bz = vnorm(at - eye);
    Vec3 worldUp(0.0, 0.0, 1.0);
    Vec3 right = vcross(bz, worldUp);
    if (vlen(right) < 1e-12) right = Vec3(1.0, 0.0, 0.0);
    bx = vnorm(right);
    by = vnorm(vcross(bx, bz));
}

__host__ __device__ inline Vec3 pixelRayDir(
    int px, int py, int W, int H,
    double fovDeg,
    const Vec3& bx, const Vec3& by, const Vec3& bz
) {
    double nx = (W == 1) ? 0.0 : (-1.0 + 2.0 * (double)px / (double)(W - 1));
    double ny = (H == 1) ? 0.0 : (-1.0 + 2.0 * (double)py / (double)(H - 1));
    double aspect = (double)H / (double)W;

    double z = 1.0 / ::tan(fovDeg * M_PI / 360.0);
    Vec3 v(nx, ny * aspect, z);

    Vec3 dir = basisMul(bx, by, bz, v);
    return vnorm(dir);
}

static inline Vec3 evalPolar(double R, double Phi, double Z) {
    return Vec3(R * ::cos(Phi), R * ::sin(Phi), Z);
}

// ============================================================================
//                           RENDER CPU / GPU + SSAA
// ============================================================================
__host__ __device__ void renderCPU(
    uchar4* hiRes,
    int W, int H,
    Vec3 eye, Vec3 at,
    double fovDeg,
    const Light* lights, int lightCount,
    const Tri* tris, int nTris
) {
    Vec3 bx, by, bz;
    buildCameraBasis(eye, at, bx, by, bz);

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            Vec3 rd = pixelRayDir(x, y, W, H, fovDeg, bx, by, bz);
            hiRes[(H - 1 - y) * W + x] = shadeDiffuse(eye, rd, lights, lightCount, tris, nTris);
        }
    }
}

__global__ void renderGPU(
    uchar4* hiRes,
    int W, int H,
    Vec3 eye, Vec3 at,
    double fovDeg,
    const Light* lights, int lightCount,
    const Tri* tris, int nTris
) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    int sx = gridDim.x * blockDim.x;
    int sy = gridDim.y * blockDim.y;

    Vec3 bx, by, bz;
    buildCameraBasis(eye, at, bx, by, bz);

    for (int y = ty; y < H; y += sy) {
        for (int x = tx; x < W; x += sx) {
            Vec3 rd = pixelRayDir(x, y, W, H, fovDeg, bx, by, bz);
            hiRes[(H - 1 - y) * W + x] = shadeDiffuse(eye, rd, lights, lightCount, tris, nTris);
        }
    }
}

__host__ __device__ void downsampleCPU(
    const uchar4* hiRes, uchar4* out,
    int w, int h, int sqrtRpp
) {
    int W = w * sqrtRpp;
    int rpp = sqrtRpp * sqrtRpp;

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            uint4 acc = make_uint4(0,0,0,0);
            int baseX = x * sqrtRpp;
            int baseY = y * sqrtRpp;

            for (int j = 0; j < sqrtRpp; ++j) {
                for (int i = 0; i < sqrtRpp; ++i) {
                    uchar4 p = hiRes[(baseY + j) * W + (baseX + i)];
                    acc.x += p.x; acc.y += p.y; acc.z += p.z;
                }
            }
            out[y*w + x] = make_uchar4(acc.x / rpp, acc.y / rpp, acc.z / rpp, 255);
        }
    }
}

__global__ void downsampleGPU(
    const uchar4* hiRes, uchar4* out,
    int w, int h, int sqrtRpp
) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    int sx = gridDim.x * blockDim.x;
    int sy = gridDim.y * blockDim.y;

    int W = w * sqrtRpp;
    int rpp = sqrtRpp * sqrtRpp;

    for (int y = ty; y < h; y += sy) {
        for (int x = tx; x < w; x += sx) {
            uint4 acc = make_uint4(0,0,0,0);
            int baseX = x * sqrtRpp;
            int baseY = y * sqrtRpp;

            for (int j = 0; j < sqrtRpp; ++j) {
                for (int i = 0; i < sqrtRpp; ++i) {
                    uchar4 p = hiRes[(baseY + j) * W + (baseX + i)];
                    acc.x += p.x; acc.y += p.y; acc.z += p.z;
                }
            }
            out[y*w + x] = make_uchar4(acc.x / rpp, acc.y / rpp, acc.z / rpp, 255);
        }
    }
}

// ============================================================================
//                               GEOMETRY BUILD
// ============================================================================
static void addFloor(std::vector<Tri>& tris, const cfg::Config& c) {
    uchar4 col = rgb01_to_uchar4(c.ftr, c.ftg, c.ftb);
    Vec3 a(c.fax, c.fay, c.faz);
    Vec3 b(c.fbx, c.fby, c.fbz);
    Vec3 cc(c.fcx, c.fcy, c.fcz);
    Vec3 d(c.fdx, c.fdy, c.fdz);
    tris.emplace_back(a, b, cc, col);
    tris.emplace_back(a, cc, d, col);
}

static void addCube(std::vector<Tri>& tris, double cx,double cy,double cz, double r,double g,double b, double radius) {
    Vec3 center(cx,cy,cz);
    double a = 2.0 * radius / ::sqrt(3.0);
    Vec3 o(center.x - a/2, center.y - a/2, center.z - a/2);

    Vec3 p[8] = {
        {o.x,     o.y,     o.z},
        {o.x,     o.y+a,   o.z},
        {o.x+a,   o.y+a,   o.z},
        {o.x+a,   o.y,     o.z},
        {o.x,     o.y,     o.z+a},
        {o.x,     o.y+a,   o.z+a},
        {o.x+a,   o.y+a,   o.z+a},
        {o.x+a,   o.y,     o.z+a},
    };

    uchar4 col = rgb01_to_uchar4(r,g,b);
    auto tri = [&](int ia, int ib, int ic) { tris.emplace_back(p[ia], p[ib], p[ic], col); };

    tri(0,1,2); tri(2,3,0);
    tri(6,7,3); tri(3,2,6);
    tri(2,1,5); tri(5,6,2);
    tri(4,5,1); tri(1,0,4);
    tri(3,7,4); tri(4,0,3);
    tri(6,5,4); tri(4,7,6);
}

static void addOctahedron(std::vector<Tri>& tris, double cx,double cy,double cz, double r,double g,double b, double rad) {
    Vec3 c(cx,cy,cz);

    Vec3 p[6] = {
        {c.x,     c.y - rad, c.z},
        {c.x - rad, c.y,     c.z},
        {c.x,     c.y + rad, c.z},
        {c.x + rad, c.y,     c.z},
        {c.x,     c.y,     c.z - rad},
        {c.x,     c.y,     c.z + rad},
    };

    uchar4 col = rgb01_to_uchar4(r,g,b);
    auto tri = [&](int ia, int ib, int ic) { tris.emplace_back(p[ia], p[ib], p[ic], col); };

    tri(0,1,4); tri(1,2,4); tri(2,3,4); tri(3,0,4);
    tri(0,5,1); tri(1,5,2); tri(2,5,3); tri(3,5,0);
}

static void addIcosahedron(std::vector<Tri>& tris, double cx,double cy,double cz, double r,double g,double b, double radius) {
    Vec3 center(cx,cy,cz);
    double phi = (1.0 + ::sqrt(5.0)) / 2.0;
    double s = radius / ::sqrt(1.0 + phi*phi);

    Vec3 v[12] = {
        {-1,  phi, 0}, { 1,  phi, 0}, {-1, -phi, 0}, { 1, -phi, 0},
        { 0, -1,  phi}, {0,  1,  phi}, {0, -1, -phi}, {0,  1, -phi},
        { phi, 0, -1}, {phi, 0,  1}, {-phi, 0, -1}, {-phi, 0,  1}
    };

    for (int i = 0; i < 12; ++i) {
        v[i].x = v[i].x * s + center.x;
        v[i].y = v[i].y * s + center.y;
        v[i].z = v[i].z * s + center.z;
    }

    const int F[20][3] = {
        {0,11,5},{0,5,1},{0,1,7},{0,7,10},{0,10,11},
        {1,5,9},{5,11,4},{11,10,2},{10,7,6},{7,1,8},
        {3,9,4},{3,4,2},{3,2,6},{3,6,8},{3,8,9},
        {4,9,5},{2,4,11},{6,2,10},{8,6,7},{9,8,1}
    };

    uchar4 col = rgb01_to_uchar4(r,g,b);
    for (int i = 0; i < 20; ++i) {
        tris.emplace_back(v[F[i][0]], v[F[i][1]], v[F[i][2]], col);
    }
}

// ============================================================================
//                                   APP
// ============================================================================
class RendererApp {
public:
    RendererApp(bool gpu, const cfg::Config& config,
                cfg::Config runtimeCfg,
                std::vector<Light> lights)
        : useGpu(gpu), defCfg(config), inCfg(runtimeCfg), lights(std::move(lights)) {}

    void run() {
        const int w = inCfg.width;
        const int h = inCfg.height;
        int sqrtRpp = inCfg.sqrtRpp;
        if (sqrtRpp < 1) sqrtRpp = 1;

        const int hiW = w * sqrtRpp;
        const int hiH = h * sqrtRpp;

        std::vector<Tri> hostTris;
        hostTris.reserve(80);

        addFloor(hostTris, inCfg);

        addCube(hostTris, inCfg.b1x,inCfg.b1y,inCfg.b1z, inCfg.b1r,inCfg.b1g,inCfg.b1b, inCfg.b1rad);
        addOctahedron(hostTris, inCfg.b2x,inCfg.b2y,inCfg.b2z, inCfg.b2r,inCfg.b2g,inCfg.b2b, inCfg.b2rad);
        addIcosahedron(hostTris, inCfg.b3x,inCfg.b3y,inCfg.b3z, inCfg.b3r,inCfg.b3g,inCfg.b3b, inCfg.b3rad);

        if (lights.empty()) {
            lights.push_back(Light{Vec3(10,0,15), make_uchar4(255,255,255,255)});
        }
        if ((int)lights.size() > cfg::MAX_LIGHTS) lights.resize(cfg::MAX_LIGHTS);

        std::vector<uchar4> hiRes((size_t)hiW * (size_t)hiH);
        std::vector<uchar4> out((size_t)w * (size_t)h);

        DeviceBuf<uchar4> dHi, dOut;
        DeviceBuf<Tri> dTris;
        DeviceBuf<Light> dLights;

        if (useGpu) {
            dHi.alloc((size_t)hiW * (size_t)hiH);
            dOut.alloc((size_t)w * (size_t)h);

            dTris.alloc(hostTris.size());
            CUDA_OK(cudaMemcpy(dTris.ptr, hostTris.data(), sizeof(Tri) * hostTris.size(), cudaMemcpyHostToDevice));

            dLights.alloc(lights.size());
            CUDA_OK(cudaMemcpy(dLights.ptr, lights.data(), sizeof(Light) * lights.size(), cudaMemcpyHostToDevice));
        }

        dim3 block(cfg::BLOCK_X, cfg::BLOCK_Y);
        dim3 grid(cfg::GRID_X, cfg::GRID_Y);

        for (int i = 0; i < inCfg.framesCount; ++i) {
            double t = 2.0 * M_PI * (double)i / (double)inCfg.framesCount;

            double Rc = inCfg.r0c + inCfg.arc * ::sin(inCfg.wrc * t + inCfg.prc);
            double Zc = inCfg.z0c + inCfg.azc * ::sin(inCfg.wzc * t + inCfg.pzc);
            double Phic = inCfg.phi0c + inCfg.wphic * t;
            Vec3 eye = evalPolar(Rc, Phic, Zc);

            double Rn = inCfg.r0n + inCfg.arn * ::sin(inCfg.wrn * t + inCfg.prn);
            double Zn = inCfg.z0n + inCfg.azn * ::sin(inCfg.wzn * t + inCfg.pzn);
            double Phin = inCfg.phi0n + inCfg.wphin * t;
            Vec3 at = evalPolar(Rn, Phin, Zn);

            cudaEvent_t ev0, ev1;
            CUDA_OK(cudaEventCreate(&ev0));
            CUDA_OK(cudaEventCreate(&ev1));
            CUDA_OK(cudaEventRecord(ev0));

            if (useGpu) {
                renderGPU<<<grid, block>>>(
                    dHi.ptr, hiW, hiH,
                    eye, at,
                    inCfg.fovDeg,
                    dLights.ptr, (int)lights.size(),
                    dTris.ptr, (int)hostTris.size()
                );
                CUDA_OK(cudaGetLastError());

                downsampleGPU<<<grid, block>>>(
                    dHi.ptr, dOut.ptr,
                    w, h, sqrtRpp
                );
                CUDA_OK(cudaGetLastError());

                CUDA_OK(cudaMemcpy(out.data(), dOut.ptr, sizeof(uchar4) * out.size(), cudaMemcpyDeviceToHost));
            } else {
                renderCPU(
                    hiRes.data(),
                    hiW, hiH,
                    eye, at,
                    inCfg.fovDeg,
                    lights.data(), (int)lights.size(),
                    hostTris.data(), (int)hostTris.size()
                );
                downsampleCPU(hiRes.data(), out.data(), w, h, sqrtRpp);
            }

            CUDA_OK(cudaEventRecord(ev1));
            CUDA_OK(cudaEventSynchronize(ev1));
            float ms = 0.f;
            CUDA_OK(cudaEventElapsedTime(&ms, ev0, ev1));
            CUDA_OK(cudaEventDestroy(ev0));
            CUDA_OK(cudaEventDestroy(ev1));

            char fileName[512];
            std::snprintf(fileName, sizeof(fileName), inCfg.outPattern, i);

            FILE* f = std::fopen(fileName, "wb");
            if (!f) {
                std::cerr << "Cannot open output: " << fileName << "\n";
                std::exit(EXIT_FAILURE);
            }
            std::fwrite(&w, sizeof(int), 1, f);
            std::fwrite(&h, sizeof(int), 1, f);
            std::fwrite(out.data(), sizeof(uchar4), out.size(), f);
            std::fclose(f);

            long long rays = 1LL * (long long)hiW * (long long)hiH;
            std::cout << (i + 1) << "\t" << ms << "\t" << rays << "\n";
        }
    }

private:
    bool useGpu;
    const cfg::Config& defCfg;
    cfg::Config inCfg;
    std::vector<Light> lights;
};

// ============================================================================
//                      INPUT READING IN ASSIGNMENT FORMAT
// ============================================================================
static bool readConfigFromStdin(cfg::Config& c) {
    // 1
    if (!(std::cin >> c.framesCount)) return false;

    // 2
    std::string pattern;
    std::cin >> pattern;
    // store pointer into owned static string? cannot.
    // We'll keep runtime config holding a std::string is messy.
    // simplest: copy into a static buffer per run.
    static char patternBuf[512];
    std::snprintf(patternBuf, sizeof(patternBuf), "%s", pattern.c_str());
    c.outPattern = patternBuf;

    // 3
    std::cin >> c.width >> c.height >> c.fovDeg;

    // 4
    std::cin >> c.r0c >> c.z0c >> c.phi0c >> c.arc >> c.azc >> c.wrc >> c.wzc >> c.wphic >> c.prc >> c.pzc;
    std::cin >> c.r0n >> c.z0n >> c.phi0n >> c.arn >> c.azn >> c.wrn >> c.wzn >> c.wphin >> c.prn >> c.pzn;

    // 5 bodies
    std::cin >> c.b1x >> c.b1y >> c.b1z >> c.b1r >> c.b1g >> c.b1b >> c.b1rad >> c.b1Refl >> c.b1Trans >> c.b1EdgeLights;
    std::cin >> c.b2x >> c.b2y >> c.b2z >> c.b2r >> c.b2g >> c.b2b >> c.b2rad >> c.b2Refl >> c.b2Trans >> c.b2EdgeLights;
    std::cin >> c.b3x >> c.b3y >> c.b3z >> c.b3r >> c.b3g >> c.b3b >> c.b3rad >> c.b3Refl >> c.b3Trans >> c.b3EdgeLights;

    // 6 floor
    std::cin >> c.fax >> c.fay >> c.faz
             >> c.fbx >> c.fby >> c.fbz
             >> c.fcx >> c.fcy >> c.fcz
             >> c.fdx >> c.fdy >> c.fdz;

    std::string tex;
    std::cin >> tex;
    static char texBuf[512];
    std::snprintf(texBuf, sizeof(texBuf), "%s", tex.c_str());
    c.floorTexPath = texBuf;

    std::cin >> c.ftr >> c.ftg >> c.ftb >> c.floorRefl;

    // 7 lights
    std::cin >> c.lightsCount;
    if (c.lightsCount < 0) c.lightsCount = 0;
    if (c.lightsCount > cfg::MAX_LIGHTS) c.lightsCount = cfg::MAX_LIGHTS;

    for (int i = 0; i < c.lightsCount; ++i) {
        std::cin >> c.lx[i] >> c.ly[i] >> c.lz[i]
                 >> c.lr[i] >> c.lg[i] >> c.lb[i];
    }

    // 8 limits
    std::cin >> c.maxDepth >> c.sqrtRpp;
    return true;
}

static std::vector<Light> lightsFromConfig(const cfg::Config& c) {
    std::vector<Light> out;
    int n = c.lightsCount;
    if (n < 0) n = 0;
    if (n > cfg::MAX_LIGHTS) n = cfg::MAX_LIGHTS;
    out.reserve((size_t)n);
    for (int i = 0; i < n; ++i) {
        out.push_back(Light{
            Vec3(c.lx[i], c.ly[i], c.lz[i]),
            rgb01_to_uchar4(c.lr[i], c.lg[i], c.lb[i])
        });
    }
    return out;
}

// ============================================================================
//                                    MAIN
// ============================================================================
int main(int argc, char** argv) {
    bool wantCpu = false, wantGpu = false, wantDefault = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--cpu") wantCpu = true;
        else if (a == "--gpu") wantGpu = true;
        else if (a == "--default") wantDefault = true;
    }

    bool useGpu = true;
    if (wantCpu && !wantGpu) useGpu = false;

    if (wantDefault) {
        cfg::printDefault(std::cout);
        std::cout << "\n";
        return 0;
    }

    cfg::Config runtime = cfg::CFG;
    if (!readConfigFromStdin(runtime)) {
        std::cerr << "Bad input\n";
        return 1;
    }

    std::vector<Light> lights = lightsFromConfig(runtime);

    RendererApp app(useGpu, cfg::CFG, runtime, std::move(lights));
    app.run();
    return 0;
}
