#ifndef SwiftletNEON_h
#define SwiftletNEON_h

/// y[r] = bias[r] + W[r] · x for rows r0 ..< r1. W is fp16 row-major [rows][K], bias fp16 (or NULL), x and y fp32,
/// K a multiple of 8 and at most 2048. Products and sums are fp32 (NEON FMLAL).
void swiftlet_matvec_f16(const void *W, const void *bias, const float *x, float *y, int K, int r0, int r1);

/// Starts the helper threads for swiftlet_pool_matvec once per process; later calls do nothing.
void swiftlet_pool_start(int threads);

/// swiftlet_matvec_f16 over all rows, split across the calling thread and the helpers. Callers are serialized.
void swiftlet_pool_matvec(const void *W, const void *bias, const float *x, float *y, int K, int rows);

#endif
