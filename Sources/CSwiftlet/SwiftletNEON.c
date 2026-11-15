#include "SwiftletNEON.h"

// Apple Silicon only. The app is also built for Intel, where SpeechEngine refuses to load and these are no-ops.
#if defined(__aarch64__)
#include <arm_neon.h>
#include <pthread.h>
#include <stdatomic.h>

void swiftlet_matvec_f16(const void *weights, const void *biases, const float *x, float *y, int K, int r0, int r1) {
    const __fp16 *W = weights, *bias = biases;
    // x goes to fp16 only as the FMLAL operand; every product and sum is fp32.
    float16x8_t xh[256];
    for (int k = 0; k < K; k += 8)
        xh[k / 8] = vcombine_f16(vcvt_f16_f32(vld1q_f32(x + k)), vcvt_f16_f32(vld1q_f32(x + k + 4)));
    int r = r0;
    for (; r + 4 <= r1; r += 4) {
        const __fp16 *p0 = W + (long)r * K, *p1 = p0 + K, *p2 = p1 + K, *p3 = p2 + K;
        float32x4_t a0 = vdupq_n_f32(0), a1 = a0, a2 = a0, a3 = a0, b0 = a0, b1 = a0, b2 = a0, b3 = a0;
        for (int k = 0; k < K; k += 8) {
            float16x8_t xv = xh[k / 8];
            float16x8_t w0 = vld1q_f16(p0 + k), w1 = vld1q_f16(p1 + k), w2 = vld1q_f16(p2 + k), w3 = vld1q_f16(p3 + k);
            a0 = vfmlalq_low_f16(a0, w0, xv); b0 = vfmlalq_high_f16(b0, w0, xv);
            a1 = vfmlalq_low_f16(a1, w1, xv); b1 = vfmlalq_high_f16(b1, w1, xv);
            a2 = vfmlalq_low_f16(a2, w2, xv); b2 = vfmlalq_high_f16(b2, w2, xv);
            a3 = vfmlalq_low_f16(a3, w3, xv); b3 = vfmlalq_high_f16(b3, w3, xv);
        }
        y[r] = vaddvq_f32(vaddq_f32(a0, b0)) + (bias ? (float)bias[r] : 0);
        y[r + 1] = vaddvq_f32(vaddq_f32(a1, b1)) + (bias ? (float)bias[r + 1] : 0);
        y[r + 2] = vaddvq_f32(vaddq_f32(a2, b2)) + (bias ? (float)bias[r + 2] : 0);
        y[r + 3] = vaddvq_f32(vaddq_f32(a3, b3)) + (bias ? (float)bias[r + 3] : 0);
    }
    for (; r < r1; r++) {
        const __fp16 *p0 = W + (long)r * K;
        float32x4_t a0 = vdupq_n_f32(0), b0 = a0;
        for (int k = 0; k < K; k += 8) {
            float16x8_t w0 = vld1q_f16(p0 + k);
            a0 = vfmlalq_low_f16(a0, w0, xh[k / 8]); b0 = vfmlalq_high_f16(b0, w0, xh[k / 8]);
        }
        y[r] = vaddvq_f32(vaddq_f32(a0, b0)) + (bias ? (float)bias[r] : 0);
    }
}

// The decode loop issues thousands of small matvecs back to back, so helpers spin between jobs (a GCD wake-up
// costs about as much as a job) and only sleep after a few milliseconds without work.
#define MAX_THREADS 16
static struct { const void *W, *bias; const float *x; float *y; int K, rows, chunk; } job;
static _Atomic unsigned generation;
static _Atomic int pending;
static int poolSize = 1;
static pthread_once_t once = PTHREAD_ONCE_INIT;
static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t wake = PTHREAD_COND_INITIALIZER;
static pthread_mutex_t callers = PTHREAD_MUTEX_INITIALIZER;  // one job at a time, whoever calls

static void runSlice(int i) {
    int r0 = i * job.chunk, r1 = r0 + job.chunk < job.rows ? r0 + job.chunk : job.rows;
    if (r0 < r1) swiftlet_matvec_f16(job.W, job.bias, job.x, job.y, job.K, r0, r1);
}

static void *helper(void *argument) {
    int index = (int)(long)argument;
    unsigned seen = 0;
    for (;;) {
        unsigned current;
        long spins = 0;
        while ((current = atomic_load_explicit(&generation, memory_order_acquire)) == seen) {
            if (++spins < 200000) { __asm__ volatile("yield"); continue; }
            pthread_mutex_lock(&mutex);
            while (atomic_load(&generation) == seen) pthread_cond_wait(&wake, &mutex);
            pthread_mutex_unlock(&mutex);
            spins = 0;
        }
        seen = current;
        runSlice(index);
        atomic_fetch_sub_explicit(&pending, 1, memory_order_acq_rel);
    }
    return 0;
}

static int requested = 1;
static void startHelpers(void) {
    poolSize = requested < 1 ? 1 : requested > MAX_THREADS ? MAX_THREADS : requested;
    for (int i = 1; i < poolSize; i++) {
        pthread_t thread;
        pthread_attr_t attributes;
        pthread_attr_init(&attributes);
        pthread_attr_set_qos_class_np(&attributes, QOS_CLASS_USER_INITIATED, 0);
        if (pthread_create(&thread, &attributes, helper, (void *)(long)i) != 0) { poolSize = i; break; }
        pthread_detach(thread);
        pthread_attr_destroy(&attributes);
    }
}

void swiftlet_pool_start(int threads) {
    requested = threads;
    pthread_once(&once, startHelpers);
}

void swiftlet_pool_matvec(const void *W, const void *bias, const float *x, float *y, int K, int rows) {
    if (poolSize <= 1) { swiftlet_matvec_f16(W, bias, x, y, K, 0, rows); return; }
    pthread_mutex_lock(&callers);
    job.W = W; job.bias = bias; job.x = x; job.y = y; job.K = K; job.rows = rows;
    job.chunk = ((rows + poolSize - 1) / poolSize + 3) / 4 * 4;
    atomic_store(&pending, poolSize - 1);
    pthread_mutex_lock(&mutex);
    atomic_fetch_add_explicit(&generation, 1, memory_order_release);
    pthread_cond_broadcast(&wake);
    pthread_mutex_unlock(&mutex);
    runSlice(0);
    while (atomic_load_explicit(&pending, memory_order_acquire) > 0) __asm__ volatile("yield");
    pthread_mutex_unlock(&callers);
}

#else
void swiftlet_matvec_f16(const void *W, const void *bias, const float *x, float *y, int K, int r0, int r1) {}
void swiftlet_pool_start(int threads) {}
void swiftlet_pool_matvec(const void *W, const void *bias, const float *x, float *y, int K, int rows) {}
#endif
