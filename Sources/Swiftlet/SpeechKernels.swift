// Metal kernels for SpeechEncoder, compiled from source at load, so the package needs no prebuilt metallib.
// GEMMs read the 4-bit palette weights in the lane-tiled layout written by Tools/export_model.py.

enum SpeechKernels {
    static let source = #"""
#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------------------------------------------
// 4-bit palette GEMM: Y[M,N] = epilogue(X[M,K] · dequant(W)[N,K]^T).
// W: indices packed two per byte (low nibble = even k); L: 16 fp16 values per output row.
// act: 0 none, 1 silu, 2 relu, 3 GLU over interleaved column pairs (writes N/2 columns: a * sigmoid(b)).
struct GemmParams { int M, N, K; int act; float alpha; int hasBias; int hasRes; int kLen; int partial; };

static inline void gemm_epilogue(float y0, float y1, int m, int n, constant GemmParams& p, device const half* bias, device const half* R, device half* Y) {
    if (p.hasBias) { y0 += float(bias[n]); y1 += float(bias[n + 1]); }
    if (p.act == 3) { Y[(long)m * (p.N / 2) + n / 2] = half(y0 / (1 + exp(-y1))); return; }
    if (p.act == 1) { y0 = y0 / (1 + exp(-y0)); y1 = y1 / (1 + exp(-y1)); }
    else if (p.act == 2) { y0 = max(y0, 0.0f); y1 = max(y1, 0.0f); }
    long o = (long)m * p.N + n;
    if (p.hasRes) { y0 = float(R[o]) + p.alpha * y0; y1 = float(R[o + 1]) + p.alpha * y1; }
    *(device half2*)(Y + o) = half2(y0, y1);
}


// Sums split-K partials [S][M][N] and applies the epilogue. One thread per column pair.
[[kernel]] void gemm_reduce(device const float* Part [[buffer(0)]], device const half* bias [[buffer(1)]], device const half* R [[buffer(2)]],
                            device half* Y [[buffer(3)]], constant GemmParams& p [[buffer(4)]], constant int& S [[buffer(5)]],
                            uint2 gid [[thread_position_in_grid]]) {
    const int n = gid.x * 2, m = gid.y;
    if (m >= p.M) return;
    float2 s = 0;
    for (int z = 0; z < S; z++) s += *(device const float2*)(Part + ((long)z * p.M + m) * p.N + n);
    gemm_epilogue(s.x, s.y, m, n, p, bias, R, Y);
}

// ---------------------------------------------------------------------------------------------------------------
// Batched dense GEMM for attention: C[b] = A[b] (M x K, lda) · B[b], B stored [N][K] (transB) or [K][N].
// Bounds-checked everywhere; C is fp32 or fp16.
struct BParams { int M, N, K, lda, ldb, ldc, transB, outHalf, sA, sB, sC; };

[[kernel]] void bgemm(device const half* A [[buffer(0)]], device const half* B [[buffer(1)]],
                      device float* Cf [[buffer(2)]], device half* Ch [[buffer(3)]], constant BParams& p [[buffer(4)]],
                      uint3 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                      ushort lane [[thread_index_in_simdgroup]], ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int BM = 32, BN = 32, BK = 16, PAD = 8;
    threadgroup half As[BM * (BK + PAD)];
    threadgroup half Bs[BK * (BN + PAD)];
    const int m0 = tg.y * BM, n0 = tg.x * BN;
    A += (long)tg.z * p.sA; B += (long)tg.z * p.sB;
    simdgroup_matrix<float, 8, 8> acc[2][2];
    for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++) acc[i][j] = simdgroup_matrix<float, 8, 8>(0);
    const int sm = sg / 2, sn = sg % 2;
    for (int k0 = 0; k0 < p.K; k0 += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int c = tid; c < BM * BK; c += 128) {
            int r = c / BK, k = c % BK, m = m0 + r, kk = k0 + k;
            As[r * (BK + PAD) + k] = (m < p.M && kk < p.K) ? A[(long)m * p.lda + kk] : half(0);
        }
        if (p.transB) {
            for (int c = tid; c < BN * BK; c += 128) {
                int n = c / BK, k = c % BK, nn = n0 + n, kk = k0 + k;
                Bs[k * (BN + PAD) + n] = (nn < p.N && kk < p.K) ? B[(long)nn * p.ldb + kk] : half(0);
            }
        } else {
            for (int c = tid; c < BN * BK; c += 128) {
                int k = c / BN, n = c % BN, nn = n0 + n, kk = k0 + k;
                Bs[k * (BN + PAD) + n] = (nn < p.N && kk < p.K) ? B[(long)kk * p.ldb + nn] : half(0);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int kk = 0; kk < BK; kk += 8) {
            simdgroup_matrix<half, 8, 8> a[2], b[2];
            for (int i = 0; i < 2; i++) simdgroup_load(a[i], As + (sm * 16 + i * 8) * (BK + PAD) + kk, BK + PAD);
            for (int j = 0; j < 2; j++) simdgroup_load(b[j], Bs + kk * (BN + PAD) + sn * 16 + j * 8, BN + PAD);
            for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
        }
    }
    const short qid = lane / 4;
    const short fm = (qid & 4) + ((lane / 2) % 4);
    const short fn = (qid & 2) * 2 + (lane % 2) * 2;
    for (int i = 0; i < 2; i++) {
        int m = m0 + sm * 16 + i * 8 + fm;
        if (m >= p.M) continue;
        for (int j = 0; j < 2; j++) {
            int n = n0 + sn * 16 + j * 8 + fn;
            auto e = acc[i][j].thread_elements();
            long o = (long)tg.z * p.sC + (long)m * p.ldc + n;
            for (int u = 0; u < 2; u++) {
                if (n + u >= p.N) break;
                if (p.outHalf) Ch[o + u] = half(e[u]); else Cf[o + u] = e[u];
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------
// LayerNorm over rows of C (multiple of 4), eps 1e-5, fp32 statistics. One 256-thread threadgroup per row.
[[kernel]] void layernorm(device const half* X [[buffer(0)]], device const half* g [[buffer(1)]],
                          device const half* b [[buffer(2)]], device half* Y [[buffer(3)]], constant int& C [[buffer(4)]],
                          uint row [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                          ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float part[8];
    device const half4* x = (device const half4*)(X + (long)row * C);
    float4 v[2];
    int n4 = C / 4, cnt = 0;
    float s = 0;
    for (int i = tid; i < n4; i += 256) { v[cnt] = float4(x[i]); s += v[cnt].x + v[cnt].y + v[cnt].z + v[cnt].w; cnt++; }
    s = simd_sum(s);
    if (lane == 0) part[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mean = 0;
    for (int i = 0; i < 8; i++) mean += part[i];
    mean /= C;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float q = 0;
    for (int i = 0; i < cnt; i++) { float4 d = v[i] - mean; q += dot(d, d); }
    q = simd_sum(q);
    if (lane == 0) part[sg] = q;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float var = 0;
    for (int i = 0; i < 8; i++) var += part[i];
    float inv = rsqrt(var / C + 1e-5f);
    device half4* y = (device half4*)(Y + (long)row * C);
    device const half4* g4 = (device const half4*)g; device const half4* b4 = (device const half4*)b;
    cnt = 0;
    for (int i = tid; i < n4; i += 256) { y[i] = half4((v[cnt] - mean) * inv * float4(g4[i]) + float4(b4[i])); cnt++; }
}

// ---------------------------------------------------------------------------------------------------------------
// Subsampling stage 1+2 fused: conv2d 1->256 3x3 s2 p1 + relu, then depthwise 3x3 s2 p1 + bias.
// mel: fp32 [128][melStride] (freq-major); out: [T2][32][256]. Kernel taps are (time, freq) = (dt, df).
struct SubParams { int T0, T1, T2, melStride; };
[[kernel]] void sub_conv01(device const float* mel [[buffer(0)]], device const half* w0 [[buffer(1)]],
                           device const half* b0 [[buffer(2)]], device const half* w1 [[buffer(3)]],
                           device const half* b1 [[buffer(4)]], device half* out [[buffer(5)]],
                           constant SubParams& p [[buffer(6)]], uint3 gid [[thread_position_in_grid]]) {
    const int c = gid.x, f2 = gid.y, t2 = gid.z;
    if (t2 >= p.T2) return;
    float wa[9];
    for (int i = 0; i < 9; i++) wa[i] = float(w0[c * 9 + i]);
    const float bias0 = float(b0[c]);
    float acc = float(b1[c]);
    for (int dt = 0; dt < 3; dt++) {
        int t1 = 2 * t2 - 1 + dt;
        if (t1 < 0 || t1 >= p.T1) continue;
        for (int df = 0; df < 3; df++) {
            int f1 = 2 * f2 - 1 + df;
            if (f1 < 0 || f1 >= 64) continue;
            float v = bias0;
            for (int a = 0; a < 3; a++) {
                int t0 = 2 * t1 - 1 + a;
                if (t0 < 0 || t0 >= p.T0) continue;
                for (int b = 0; b < 3; b++) {
                    int f0 = 2 * f1 - 1 + b;
                    if (f0 < 0 || f0 >= 128) continue;
                    v += wa[a * 3 + b] * mel[f0 * p.melStride + t0];
                }
            }
            acc += float(w1[c * 9 + dt * 3 + df]) * max(v, 0.0f);
        }
    }
    out[((long)t2 * 32 + f2) * 256 + c] = half(acc);
}

// Depthwise 3x3 stride 2 pad 1 over [Ti][Fi][256] -> [To][Fo][256], plus bias.
struct DwParams { int Ti, Fi, To, Fo; };
[[kernel]] void sub_dw(device const half* x [[buffer(0)]], device const half* w [[buffer(1)]], device const half* b [[buffer(2)]],
                       device half* out [[buffer(3)]], constant DwParams& p [[buffer(4)]], uint3 gid [[thread_position_in_grid]]) {
    const int c = gid.x, fo = gid.y, to = gid.z;
    if (to >= p.To) return;
    float acc = float(b[c]);
    for (int dt = 0; dt < 3; dt++) {
        int ti = 2 * to - 1 + dt;
        if (ti < 0 || ti >= p.Ti) continue;
        for (int df = 0; df < 3; df++) {
            int fi = 2 * fo - 1 + df;
            if (fi < 0 || fi >= p.Fi) continue;
            acc += float(w[c * 9 + dt * 3 + df]) * float(x[((long)ti * p.Fi + fi) * 256 + c]);
        }
    }
    out[((long)to * p.Fo + fo) * 256 + c] = half(acc);
}

// ---------------------------------------------------------------------------------------------------------------
// Depthwise conv over time, same padding: x [T][C], w [C][k]; act 1 silu, 2 exact gelu.
static inline float erf_approx(float x) {  // Abramowitz-Stegun 7.1.26, |err| < 1.5e-7
    float s = sign(x); x = fabs(x);
    float t = 1.0f / (1.0f + 0.3275911f * x);
    float y = 1.0f - (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t * exp(-x * x);
    return s * y;
}
struct Dw1Params { int T, C, k, act; };
[[kernel]] void dwconv_time(device const half* x [[buffer(0)]], device const half* w [[buffer(1)]], device const half* b [[buffer(2)]],
                            device half* y [[buffer(3)]], constant Dw1Params& p [[buffer(4)]], uint2 gid [[thread_position_in_grid]]) {
    const int c = gid.x, t = gid.y;
    if (t >= p.T) return;
    const int pad = p.k / 2;
    float acc = float(b[c]);
    for (int i = 0; i < p.k; i++) {
        int tt = t - pad + i;
        if (tt < 0 || tt >= p.T) continue;
        acc += float(w[c * p.k + i]) * float(x[(long)tt * p.C + c]);
    }
    if (p.act == 1) acc = acc / (1 + exp(-acc));
    else if (p.act == 2) acc = 0.5f * acc * (1 + erf_approx(acc * 0.70710678f));
    y[(long)t * p.C + c] = half(acc);
}

// ---------------------------------------------------------------------------------------------------------------
// Attention helpers. qkv [T][3072]; qu = q + bias_u, qv = q + bias_v.
[[kernel]] void att_bias(device const half* qkv [[buffer(0)]], device const half* bu [[buffer(1)]], device const half* bv [[buffer(2)]],
                         device half* qu [[buffer(3)]], device half* qv [[buffer(4)]], uint2 gid [[thread_position_in_grid]]) {
    const int c = gid.x, t = gid.y;
    float q = float(qkv[(long)t * 3072 + c]);
    qu[(long)t * 1024 + c] = half(q + float(bu[c]));
    qv[(long)t * 1024 + c] = half(q + float(bv[c]));
}

// Softmax of (ac + rel-shifted bd) * scale per (head, query row). S1 [H][T][T], S2 [H][T][2T-1] -> P [H][T][T] fp16.
// One simdgroup per row.
[[kernel]] void att_softmax(device const float* S1 [[buffer(0)]], device const float* S2 [[buffer(1)]], device half* P [[buffer(2)]],
                            constant int& T [[buffer(3)]], uint gsg [[threadgroup_position_in_grid]],
                            ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    const int row = gsg * 8 + sg;  // row = h * T + i
    if (row >= 8 * T) return;
    const int h = row / T, i = row % T;
    device const float* a = S1 + (long)row * T;
    device const float* d = S2 + ((long)h * T + i) * (2 * T - 1) + (T - 1 - i);
    const float scale = 0.08838834764831845f;
    float mx = -INFINITY;
    for (int j = lane; j < T; j += 32) mx = max(mx, (a[j] + d[j]) * scale);
    mx = simd_max(mx);
    float sum = 0;
    for (int j = lane; j < T; j += 32) sum += exp((a[j] + d[j]) * scale - mx);
    sum = simd_sum(sum);
    const float inv = 1.0f / sum;
    device half* out = P + (long)row * T;
    for (int j = lane; j < T; j += 32) out[j] = half(exp((a[j] + d[j]) * scale - mx) * inv);
}

// Convert mel MLMultiArray-style fp32 is consumed directly; this copies fp16 rows for the final output.
[[kernel]] void to_float(device const half* x [[buffer(0)]], device float* y [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    y[i] = float(x[i]);
}
// ---------------------------------------------------------------------------------------------------------------
// X-stationary variant: X tile staged in threadgroup memory (A operand), W fragments (B operand, k x n) built per
// lane in registers from a lane-tiled layout: [N/8][K/32][32 lanes] uint; for lane l, byte s holds
// idx(n = 8*nb + fn(l), k = 32*kb + 8*s + fm(l)) | idx(n + 1, same k) << 4.
template <int MF, int NF, int SM, int SN>
[[kernel]] void wq2_gemm(device const half* X [[buffer(0)]], device const uint* W [[buffer(1)]],
                         device const half* L [[buffer(2)]], device const half* bias [[buffer(3)]],
                         device const half* R [[buffer(4)]], device half* Y [[buffer(5)]],
                         constant GemmParams& p [[buffer(6)]], device float* Part [[buffer(7)]],
                         uint3 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                         ushort lane [[thread_index_in_simdgroup]], ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int THREADS = SM * SN * 32, BM = SM * MF * 8, BN = SN * NF * 8, BK = 32, PAD = 8, XS = BM * (BK + PAD);
    threadgroup half Xs[2 * XS];
    threadgroup half Ls[BN * 16];
    const int m0 = tg.x * BM, n0 = tg.y * BN;
    const int K = p.K, M = p.M, KB = K / 32;
    for (int i = tid; i < BN * 16; i += THREADS) Ls[i] = L[(long)n0 * 16 + i];

    const int sm = sg / SN, sn = sg % SN;
    const short qid = lane / 4;
    const short fm = (qid & 4) + ((lane / 2) % 4);
    const short fn = (qid & 2) * 2 + (lane % 2) * 2;
    const int nS = n0 + sn * NF * 8;
    device const uint* wp = W + (long)(nS / 8) * KB * 32 + lane;

    simdgroup_matrix<float, 8, 8> acc[MF][NF];
    for (int i = 0; i < MF; i++) for (int j = 0; j < NF; j++) acc[i][j] = simdgroup_matrix<float, 8, 8>(0);

    constexpr int XCH = BM * BK / 8;
    constexpr int XPT = (XCH + THREADS - 1) / THREADS;
    const int kbBeg = tg.z * (p.kLen / 32), kbEnd = min(KB, kbBeg + p.kLen / 32);
    half4 xr[XPT][2];
    auto fetch = [&](int kb) {
        for (int t = 0; t < XPT; t++) {
            int c = tid + t * THREADS;
            if (XCH % THREADS && c >= XCH) break;
            int r = c / (BK / 8), kc = (c % (BK / 8)) * 8, m = m0 + r;
            if (m < M) { device const half4* s = (device const half4*)(X + (long)m * K + kb * 32 + kc); xr[t][0] = s[0]; xr[t][1] = s[1]; }
            else { xr[t][0] = 0; xr[t][1] = 0; }
        }
    };
    auto store = [&](int buf) {
        for (int t = 0; t < XPT; t++) {
            int c = tid + t * THREADS;
            if (XCH % THREADS && c >= XCH) break;
            int r = c / (BK / 8), kc = (c % (BK / 8)) * 8;
            threadgroup half4* d = (threadgroup half4*)(Xs + buf * XS + r * (BK + PAD) + kc);
            d[0] = xr[t][0]; d[1] = xr[t][1];
        }
    };
    fetch(kbBeg);
    int buf = 0;
    for (int kb = kbBeg; kb < kbEnd; kb++) {
        store(buf);
        uint w[NF];
        for (int f = 0; f < NF; f++) w[f] = wp[((long)f * KB + kb) * 32];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (kb + 1 < kbEnd) fetch(kb + 1);
        threadgroup const half* xs = Xs + buf * XS + (sm * MF * 8) * (BK + PAD);
        for (int s = 0; s < 4; s++) {
            simdgroup_matrix<half, 8, 8> a[MF];
            for (int i = 0; i < MF; i++) simdgroup_load(a[i], xs + i * 8 * (BK + PAD) + s * 8, BK + PAD);
            for (int f = 0; f < NF; f++) {
                uint byte = (w[f] >> (8 * s)) & 0xff;
                threadgroup const half* lt = Ls + (sn * NF * 8 + f * 8 + fn) * 16;
                simdgroup_matrix<half, 8, 8> b;
                b.thread_elements()[0] = lt[byte & 15];
                b.thread_elements()[1] = lt[16 + (byte >> 4)];
                for (int i = 0; i < MF; i++) simdgroup_multiply_accumulate(acc[i][f], a[i], b, acc[i][f]);
            }
        }
        buf ^= 1;
    }
    for (int i = 0; i < MF; i++) {
        int m = m0 + sm * MF * 8 + i * 8 + fm;
        if (m >= M) continue;
        for (int f = 0; f < NF; f++) {
            int n = nS + f * 8 + fn;
            auto e = acc[i][f].thread_elements();
            if (p.partial) { *(device float2*)(Part + ((long)tg.z * M + m) * p.N + n) = float2(e[0], e[1]); continue; }
            gemm_epilogue(e[0], e[1], m, n, p, bias, R, Y);
        }
    }
}
#define WQ2_INST(NAME, MF, NF, SM, SN) template [[host_name(NAME)]] [[kernel]] void wq2_gemm<MF, NF, SM, SN>(device const half*, device const uint*, device const half*, device const half*, device const half*, device half*, constant GemmParams&, device float*, uint3, ushort, ushort, ushort);
WQ2_INST("wq2_16x64", 2, 2, 1, 4)

// Same as wq2 plus a one-step W prefetch; DB = 0 uses one X buffer (and one extra barrier) to save threadgroup memory.
template <int MF, int NF, int SM, int SN, int DB>
[[kernel]] void wq3_gemm(device const half* X [[buffer(0)]], device const uint* W [[buffer(1)]],
                         device const half* L [[buffer(2)]], device const half* bias [[buffer(3)]],
                         device const half* R [[buffer(4)]], device half* Y [[buffer(5)]],
                         constant GemmParams& p [[buffer(6)]], device float* Part [[buffer(7)]],
                         uint3 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                         ushort lane [[thread_index_in_simdgroup]], ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int THREADS = SM * SN * 32, BM = SM * MF * 8, BN = SN * NF * 8, BK = 32, PAD = 8, XS = BM * (BK + PAD);
    threadgroup half Xs[DB ? 2 * XS : XS];
    threadgroup half Ls[BN * 16];
    const int m0 = tg.x * BM, n0 = tg.y * BN;
    const int K = p.K, M = p.M, KB = K / 32;
    for (int i = tid; i < BN * 16; i += THREADS) Ls[i] = L[(long)n0 * 16 + i];

    const int sm = sg / SN, sn = sg % SN;
    const short qid = lane / 4;
    const short fm = (qid & 4) + ((lane / 2) % 4);
    const short fn = (qid & 2) * 2 + (lane % 2) * 2;
    const int nS = n0 + sn * NF * 8;
    device const uint* wp = W + (long)(nS / 8) * KB * 32 + lane;

    simdgroup_matrix<float, 8, 8> acc[MF][NF];
    for (int i = 0; i < MF; i++) for (int j = 0; j < NF; j++) acc[i][j] = simdgroup_matrix<float, 8, 8>(0);

    constexpr int XCH = BM * BK / 8;
    constexpr int XPT = (XCH + THREADS - 1) / THREADS;
    const int kbBeg = tg.z * (p.kLen / 32), kbEnd = min(KB, kbBeg + p.kLen / 32);
    half4 xr[XPT][2];
    auto fetch = [&](int kb) {
        for (int t = 0; t < XPT; t++) {
            int c = tid + t * THREADS;
            if (XCH % THREADS && c >= XCH) break;
            int r = c / (BK / 8), kc = (c % (BK / 8)) * 8, m = m0 + r;
            if (m < M) { device const half4* s = (device const half4*)(X + (long)m * K + kb * 32 + kc); xr[t][0] = s[0]; xr[t][1] = s[1]; }
            else { xr[t][0] = 0; xr[t][1] = 0; }
        }
    };
    auto store = [&](int buf) {
        for (int t = 0; t < XPT; t++) {
            int c = tid + t * THREADS;
            if (XCH % THREADS && c >= XCH) break;
            int r = c / (BK / 8), kc = (c % (BK / 8)) * 8;
            threadgroup half4* d = (threadgroup half4*)(Xs + buf * XS + r * (BK + PAD) + kc);
            d[0] = xr[t][0]; d[1] = xr[t][1];
        }
    };
    fetch(kbBeg);
    uint wn[NF];
    for (int f = 0; f < NF; f++) wn[f] = wp[((long)f * KB + kbBeg) * 32];
    int buf = 0;
    for (int kb = kbBeg; kb < kbEnd; kb++) {
        if (!DB) threadgroup_barrier(mem_flags::mem_threadgroup);
        store(buf);
        uint w[NF];
        for (int f = 0; f < NF; f++) w[f] = wn[f];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (kb + 1 < kbEnd) { fetch(kb + 1); for (int f = 0; f < NF; f++) wn[f] = wp[((long)f * KB + kb + 1) * 32]; }
        threadgroup const half* xs = Xs + buf * XS + (sm * MF * 8) * (BK + PAD);
        for (int s = 0; s < 4; s++) {
            simdgroup_matrix<half, 8, 8> a[MF];
            for (int i = 0; i < MF; i++) simdgroup_load(a[i], xs + i * 8 * (BK + PAD) + s * 8, BK + PAD);
            for (int f = 0; f < NF; f++) {
                uint byte = (w[f] >> (8 * s)) & 0xff;
                threadgroup const half* lt = Ls + (sn * NF * 8 + f * 8 + fn) * 16;
                simdgroup_matrix<half, 8, 8> b;
                b.thread_elements()[0] = lt[byte & 15];
                b.thread_elements()[1] = lt[16 + (byte >> 4)];
                for (int i = 0; i < MF; i++) simdgroup_multiply_accumulate(acc[i][f], a[i], b, acc[i][f]);
            }
        }
        if (DB) buf ^= 1;
    }
    for (int i = 0; i < MF; i++) {
        int m = m0 + sm * MF * 8 + i * 8 + fm;
        if (m >= M) continue;
        for (int f = 0; f < NF; f++) {
            int n = nS + f * 8 + fn;
            auto e = acc[i][f].thread_elements();
            if (p.partial) { *(device float2*)(Part + ((long)tg.z * M + m) * p.N + n) = float2(e[0], e[1]); continue; }
            gemm_epilogue(e[0], e[1], m, n, p, bias, R, Y);
        }
    }
}
#define WQ3_INST(NAME, MF, NF, SM, SN, DB) template [[host_name(NAME)]] [[kernel]] void wq3_gemm<MF, NF, SM, SN, DB>(device const half*, device const uint*, device const half*, device const half*, device const half*, device half*, constant GemmParams&, device float*, uint3, ushort, ushort, ushort);
WQ3_INST("wq3_m8n2_1x4", 8, 2, 1, 4, 0)
"""#
}
