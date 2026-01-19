/*
 * FLUX Metal Acceleration - Optimized Implementation
 *
 * Uses Metal Performance Shaders (MPS) for GPU-accelerated matrix operations.
 * Optimizations:
 * - Weight buffer caching (weights stay on GPU)
 * - Shared memory buffers (zero-copy on Apple Silicon unified memory)
 * - Buffer pooling for activations
 * - Batched command buffer execution
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include "flux_metal.h"
#include <stdio.h>
#include <string.h>
#include <pthread.h>

/* Global Metal state */
static id<MTLDevice> g_device = nil;
static id<MTLCommandQueue> g_queue = nil;
static int g_initialized = 0;

/* ========================================================================
 * Batch Execution State
 * When in batch mode, operations are encoded but not executed until
 * flux_metal_end_batch() is called.
 * ======================================================================== */

#define MAX_BATCH_OUTPUTS 256

typedef struct {
    id<MTLBuffer> buffer;
    float *cpu_ptr;
    size_t size;
} pending_output_t;

static id<MTLCommandBuffer> g_batch_cmd = nil;
static int g_in_batch = 0;
static pending_output_t g_pending_outputs[MAX_BATCH_OUTPUTS];
static int g_pending_count = 0;

/* ========================================================================
 * Weight Buffer Cache
 * Cache GPU buffers for weight matrices to avoid repeated allocations.
 * Weights are identified by their CPU pointer address.
 * ======================================================================== */

#define WEIGHT_CACHE_SIZE 512

typedef struct {
    const void *cpu_ptr;      /* CPU pointer (key) */
    id<MTLBuffer> gpu_buffer; /* Cached GPU buffer */
    size_t size;              /* Buffer size */
} weight_cache_entry_t;

static weight_cache_entry_t g_weight_cache[WEIGHT_CACHE_SIZE];
static int g_weight_cache_count = 0;
static pthread_mutex_t g_cache_mutex = PTHREAD_MUTEX_INITIALIZER;

static id<MTLBuffer> get_cached_weight_buffer(const float *weights, size_t size) {
    pthread_mutex_lock(&g_cache_mutex);

    /* Look for existing entry */
    for (int i = 0; i < g_weight_cache_count; i++) {
        if (g_weight_cache[i].cpu_ptr == weights && g_weight_cache[i].size == size) {
            id<MTLBuffer> buf = g_weight_cache[i].gpu_buffer;
            pthread_mutex_unlock(&g_cache_mutex);
            return buf;
        }
    }

    /* Not found - create new buffer */
    if (g_weight_cache_count >= WEIGHT_CACHE_SIZE) {
        /* Cache full - just create without caching */
        pthread_mutex_unlock(&g_cache_mutex);
        return [g_device newBufferWithBytes:weights
                                     length:size
                                    options:MTLResourceStorageModeShared];
    }

    /* Create and cache */
    id<MTLBuffer> buf = [g_device newBufferWithBytes:weights
                                              length:size
                                             options:MTLResourceStorageModeShared];
    g_weight_cache[g_weight_cache_count].cpu_ptr = weights;
    g_weight_cache[g_weight_cache_count].gpu_buffer = buf;
    g_weight_cache[g_weight_cache_count].size = size;
    g_weight_cache_count++;

    pthread_mutex_unlock(&g_cache_mutex);
    return buf;
}

static void clear_weight_cache(void) {
    pthread_mutex_lock(&g_cache_mutex);
    for (int i = 0; i < g_weight_cache_count; i++) {
        g_weight_cache[i].gpu_buffer = nil;
        g_weight_cache[i].cpu_ptr = NULL;
    }
    g_weight_cache_count = 0;
    pthread_mutex_unlock(&g_cache_mutex);
}

/* ========================================================================
 * Activation Buffer Pool
 * Reusable GPU buffers for activation tensors to avoid per-operation allocation.
 * Buffers use shared memory mode for zero-copy CPU access on Apple Silicon.
 * ======================================================================== */

#define ACTIVATION_POOL_SIZE 64

typedef struct {
    id<MTLBuffer> buffer;
    size_t size;
    int in_use;
} pool_buffer_t;

static pool_buffer_t g_activation_pool[ACTIVATION_POOL_SIZE];
static int g_pool_count = 0;
static pthread_mutex_t g_pool_mutex = PTHREAD_MUTEX_INITIALIZER;

/* Get a buffer from pool, or create new one if needed */
static id<MTLBuffer> pool_get_buffer(size_t size) {
    pthread_mutex_lock(&g_pool_mutex);

    /* Look for existing buffer of sufficient size */
    for (int i = 0; i < g_pool_count; i++) {
        if (!g_activation_pool[i].in_use && g_activation_pool[i].size >= size) {
            g_activation_pool[i].in_use = 1;
            id<MTLBuffer> buf = g_activation_pool[i].buffer;
            pthread_mutex_unlock(&g_pool_mutex);
            return buf;
        }
    }

    /* No suitable buffer found - create new one */
    if (g_pool_count < ACTIVATION_POOL_SIZE) {
        /* Round up size to reduce fragmentation */
        size_t alloc_size = size;
        if (alloc_size < 1024 * 1024) {
            alloc_size = ((alloc_size + 65535) / 65536) * 65536;  /* 64KB alignment */
        } else {
            alloc_size = ((alloc_size + 1048575) / 1048576) * 1048576;  /* 1MB alignment */
        }

        id<MTLBuffer> buf = [g_device newBufferWithLength:alloc_size
                                                  options:MTLResourceStorageModeShared];
        if (buf) {
            g_activation_pool[g_pool_count].buffer = buf;
            g_activation_pool[g_pool_count].size = alloc_size;
            g_activation_pool[g_pool_count].in_use = 1;
            g_pool_count++;
            pthread_mutex_unlock(&g_pool_mutex);
            return buf;
        }
    }

    pthread_mutex_unlock(&g_pool_mutex);

    /* Pool full or allocation failed - create temporary buffer */
    return [g_device newBufferWithLength:size options:MTLResourceStorageModeShared];
}

/* Return buffer to pool (mark as available) */
static void pool_release_buffer(id<MTLBuffer> buffer) {
    if (!buffer) return;

    pthread_mutex_lock(&g_pool_mutex);
    for (int i = 0; i < g_pool_count; i++) {
        if (g_activation_pool[i].buffer == buffer) {
            g_activation_pool[i].in_use = 0;
            break;
        }
    }
    pthread_mutex_unlock(&g_pool_mutex);
}

/* Release all buffers back to pool (call at end of batch) */
static void pool_release_all(void) {
    pthread_mutex_lock(&g_pool_mutex);
    for (int i = 0; i < g_pool_count; i++) {
        g_activation_pool[i].in_use = 0;
    }
    pthread_mutex_unlock(&g_pool_mutex);
}

/* Clear the entire pool */
static void clear_activation_pool(void) {
    pthread_mutex_lock(&g_pool_mutex);
    for (int i = 0; i < g_pool_count; i++) {
        g_activation_pool[i].buffer = nil;
        g_activation_pool[i].in_use = 0;
        g_activation_pool[i].size = 0;
    }
    g_pool_count = 0;
    pthread_mutex_unlock(&g_pool_mutex);
}


/* ========================================================================
 * Metal Initialization
 * ======================================================================== */

int flux_metal_init(void) {
    if (g_initialized) return 1;

    @autoreleasepool {
        /* Get default Metal device */
        g_device = MTLCreateSystemDefaultDevice();
        if (!g_device) {
            return 0;
        }

        /* Check if this is Apple Silicon */
        if (![g_device supportsFamily:MTLGPUFamilyApple7]) {
            if (![g_device supportsFamily:MTLGPUFamilyApple6]) {
                g_device = nil;
                return 0;
            }
        }

        /* Create command queue */
        g_queue = [g_device newCommandQueue];
        if (!g_queue) {
            g_device = nil;
            return 0;
        }

        /* Initialize weight cache */
        memset(g_weight_cache, 0, sizeof(g_weight_cache));

        g_initialized = 1;
        fprintf(stderr, "Metal: GPU acceleration enabled (%s)\n",
                [[g_device name] UTF8String]);

        /* Load compute shaders (high thresholds keep them dormant for small ops) */
        flux_metal_init_shaders();
    }

    return 1;
}

int flux_metal_available(void) {
    return g_initialized;
}

void flux_metal_cleanup(void) {
    if (!g_initialized) return;

    @autoreleasepool {
        /* End any pending batch */
        if (g_in_batch) {
            flux_metal_end_batch();
        }
        clear_weight_cache();
        clear_activation_pool();
        g_queue = nil;
        g_device = nil;
        g_initialized = 0;
    }
}

/* ========================================================================
 * Batch Execution Functions
 * ======================================================================== */

void flux_metal_begin_batch(void) {
    if (!g_initialized || g_in_batch) return;

    @autoreleasepool {
        g_batch_cmd = [g_queue commandBuffer];
        g_in_batch = 1;
        g_pending_count = 0;
    }
}

void flux_metal_end_batch(void) {
    if (!g_initialized || !g_in_batch) return;

    @autoreleasepool {
        if (g_batch_cmd) {
            [g_batch_cmd commit];
            [g_batch_cmd waitUntilCompleted];

            /* Copy all pending outputs back to CPU */
            for (int i = 0; i < g_pending_count; i++) {
                memcpy(g_pending_outputs[i].cpu_ptr,
                       [g_pending_outputs[i].buffer contents],
                       g_pending_outputs[i].size);
                /* Don't nil the buffer - it's from the pool */
            }

            g_batch_cmd = nil;
        }
        g_in_batch = 0;
        g_pending_count = 0;

        /* Release all pooled buffers back to pool */
        pool_release_all();
    }
}

int flux_metal_in_batch(void) {
    return g_in_batch;
}

/* ========================================================================
 * Optimized Matrix Multiplication
 * ======================================================================== */

void flux_metal_sgemm(int transpose_a, int transpose_b,
                      int M, int N, int K,
                      float alpha,
                      const float *A, int lda,
                      const float *B, int ldb,
                      float beta,
                      float *C, int ldc) {
    if (!g_initialized) return;

    @autoreleasepool {
        /* Compute dimensions */
        int rowsA = transpose_a ? K : M;
        int colsA = transpose_a ? M : K;
        int rowsB = transpose_b ? N : K;
        int colsB = transpose_b ? K : N;

        size_t sizeA = (size_t)rowsA * lda * sizeof(float);
        size_t sizeB = (size_t)rowsB * ldb * sizeof(float);
        size_t sizeC = (size_t)M * ldc * sizeof(float);

        /* Get or create buffers
         * - B (weights) uses cache (likely reused across calls)
         * - A (input) and C (output) use pooled buffers to avoid allocation overhead
         */
        id<MTLBuffer> bufferB = get_cached_weight_buffer(B, sizeB);

        /* Use pooled buffers for activations */
        id<MTLBuffer> bufferA = pool_get_buffer(sizeA);
        id<MTLBuffer> bufferC = pool_get_buffer(sizeC);

        if (!bufferA || !bufferB || !bufferC) {
            /* Fallback if buffer creation fails */
            if (bufferA) pool_release_buffer(bufferA);
            if (bufferC) pool_release_buffer(bufferC);
            return;
        }

        /* Copy input A to GPU buffer */
        memcpy([bufferA contents], A, sizeA);

        /* Initialize C if beta != 0 */
        if (beta != 0.0f) {
            memcpy([bufferC contents], C, sizeC);
        }

        /* Create matrix descriptors */
        MPSMatrixDescriptor *descA = [MPSMatrixDescriptor
            matrixDescriptorWithRows:rowsA
                             columns:colsA
                            rowBytes:lda * sizeof(float)
                            dataType:MPSDataTypeFloat32];

        MPSMatrixDescriptor *descB = [MPSMatrixDescriptor
            matrixDescriptorWithRows:rowsB
                             columns:colsB
                            rowBytes:ldb * sizeof(float)
                            dataType:MPSDataTypeFloat32];

        MPSMatrixDescriptor *descC = [MPSMatrixDescriptor
            matrixDescriptorWithRows:M
                             columns:N
                            rowBytes:ldc * sizeof(float)
                            dataType:MPSDataTypeFloat32];

        /* Create MPS matrices */
        MPSMatrix *matrixA = [[MPSMatrix alloc] initWithBuffer:bufferA descriptor:descA];
        MPSMatrix *matrixB = [[MPSMatrix alloc] initWithBuffer:bufferB descriptor:descB];
        MPSMatrix *matrixC = [[MPSMatrix alloc] initWithBuffer:bufferC descriptor:descC];

        /* Create and configure matrix multiplication */
        MPSMatrixMultiplication *matmul = [[MPSMatrixMultiplication alloc]
            initWithDevice:g_device
               transposeLeft:transpose_a ? YES : NO
              transposeRight:transpose_b ? YES : NO
                  resultRows:M
               resultColumns:N
             interiorColumns:K
                       alpha:alpha
                        beta:beta];

        /* Use batch command buffer if in batch mode, otherwise create new one */
        id<MTLCommandBuffer> cmdBuffer = g_in_batch ? g_batch_cmd : [g_queue commandBuffer];

        [matmul encodeToCommandBuffer:cmdBuffer
                           leftMatrix:matrixA
                          rightMatrix:matrixB
                         resultMatrix:matrixC];

        if (g_in_batch) {
            /* In batch mode: defer result copy until end_batch */
            if (g_pending_count < MAX_BATCH_OUTPUTS) {
                g_pending_outputs[g_pending_count].buffer = bufferC;
                g_pending_outputs[g_pending_count].cpu_ptr = C;
                g_pending_outputs[g_pending_count].size = sizeC;
                g_pending_count++;
                /* bufferA can be released immediately after encoding */
                pool_release_buffer(bufferA);
            } else {
                /* Too many pending outputs - fall back to immediate sync */
                [cmdBuffer commit];
                [cmdBuffer waitUntilCompleted];
                memcpy(C, [bufferC contents], sizeC);
                pool_release_buffer(bufferA);
                pool_release_buffer(bufferC);
            }
        } else {
            /* Not in batch mode: execute immediately */
            [cmdBuffer commit];
            [cmdBuffer waitUntilCompleted];
            memcpy(C, [bufferC contents], sizeC);

            /* Release pooled buffers */
            pool_release_buffer(bufferA);
            pool_release_buffer(bufferC);
        }
    }
}

/* Convert bf16 to f16 for MPS compatibility
 * MPS only supports mixed precision with f16, not bf16 */
static inline uint16_t bf16_to_f16(uint16_t bf16) {
    /* bf16: sign(1) + exp(8) + mant(7)
     * f16:  sign(1) + exp(5) + mant(10)
     * We need to rebias the exponent and handle special cases */
    uint32_t sign = (bf16 >> 15) & 0x1;
    int32_t exp = (bf16 >> 7) & 0xFF;  /* bf16 exponent (bias 127) */
    uint32_t mant = bf16 & 0x7F;       /* bf16 mantissa (7 bits) */

    if (exp == 0) {
        /* Zero or denormal -> zero in f16 (denormals too small) */
        return (uint16_t)(sign << 15);
    } else if (exp == 0xFF) {
        /* Inf or NaN */
        return (uint16_t)((sign << 15) | 0x7C00 | (mant ? 0x200 : 0));
    }

    /* Rebias: bf16 bias=127, f16 bias=15 */
    int32_t new_exp = exp - 127 + 15;

    if (new_exp <= 0) {
        /* Underflow to zero */
        return (uint16_t)(sign << 15);
    } else if (new_exp >= 31) {
        /* Overflow to infinity */
        return (uint16_t)((sign << 15) | 0x7C00);
    }

    /* Normal case: shift mantissa from 7 bits to 10 bits */
    uint32_t new_mant = mant << 3;  /* 7 -> 10 bits */
    return (uint16_t)((sign << 15) | (new_exp << 10) | new_mant);
}

/* F16 weight cache (stores bf16 weights converted to f16 for MPS) */
#define F16_WEIGHT_CACHE_SIZE 512

typedef struct {
    const void *cpu_ptr;
    id<MTLBuffer> gpu_buffer;
    size_t size;
} f16_cache_entry_t;

static f16_cache_entry_t g_f16_cache[F16_WEIGHT_CACHE_SIZE];
static int g_f16_cache_count = 0;
static pthread_mutex_t g_f16_cache_mutex = PTHREAD_MUTEX_INITIALIZER;

/* Get bf16 weights as f16 buffer for MPS */
static id<MTLBuffer> get_cached_bf16_as_f16_buffer(const uint16_t *weights, size_t num_elements) {
    pthread_mutex_lock(&g_f16_cache_mutex);

    /* Look for existing entry */
    for (int i = 0; i < g_f16_cache_count; i++) {
        if (g_f16_cache[i].cpu_ptr == weights) {
            id<MTLBuffer> buf = g_f16_cache[i].gpu_buffer;
            pthread_mutex_unlock(&g_f16_cache_mutex);
            return buf;
        }
    }

    /* Convert bf16 to f16 */
    uint16_t *f16_data = malloc(num_elements * sizeof(uint16_t));
    if (!f16_data) {
        pthread_mutex_unlock(&g_f16_cache_mutex);
        return nil;
    }
    for (size_t i = 0; i < num_elements; i++) {
        f16_data[i] = bf16_to_f16(weights[i]);
    }

    size_t size = num_elements * sizeof(uint16_t);

    /* Cache is full - just create buffer without caching */
    if (g_f16_cache_count >= F16_WEIGHT_CACHE_SIZE) {
        id<MTLBuffer> buf = [g_device newBufferWithBytes:f16_data
                                                  length:size
                                                 options:MTLResourceStorageModeShared];
        free(f16_data);
        pthread_mutex_unlock(&g_f16_cache_mutex);
        return buf;
    }

    /* Create and cache */
    id<MTLBuffer> buf = [g_device newBufferWithBytes:f16_data
                                              length:size
                                             options:MTLResourceStorageModeShared];
    free(f16_data);

    g_f16_cache[g_f16_cache_count].cpu_ptr = weights;
    g_f16_cache[g_f16_cache_count].gpu_buffer = buf;
    g_f16_cache[g_f16_cache_count].size = size;
    g_f16_cache_count++;

    pthread_mutex_unlock(&g_f16_cache_mutex);
    return buf;
}

/*
 * BF16 matrix multiplication: C = alpha * A @ B + beta * C
 * A is f32, B is bf16 (weights, converted to f16 for MPS), C is f32
 * This provides 2x memory bandwidth for weights.
 * Note: bf16 is converted to f16 because MPS only supports mixed f32/f16 matmul.
 */
void flux_metal_sgemm_bf16(int transpose_a, int transpose_b,
                           int M, int N, int K,
                           float alpha,
                           const float *A, int lda,
                           const uint16_t *B_bf16, int ldb,
                           float beta,
                           float *C, int ldc) {
    if (!g_initialized) return;

    @autoreleasepool {
        int rowsA = transpose_a ? K : M;
        int colsA = transpose_a ? M : K;
        int rowsB = transpose_b ? N : K;
        int colsB = transpose_b ? K : N;

        size_t sizeA = (size_t)rowsA * lda * sizeof(float);
        size_t numB = (size_t)rowsB * ldb;  /* Number of bf16 elements */
        size_t sizeC = (size_t)M * ldc * sizeof(float);

        /* Get cached f16 weight buffer (bf16 converted to f16) */
        id<MTLBuffer> bufferB = get_cached_bf16_as_f16_buffer(B_bf16, numB);

        /* Use pooled buffers for activations */
        id<MTLBuffer> bufferA = pool_get_buffer(sizeA);
        id<MTLBuffer> bufferC = pool_get_buffer(sizeC);

        if (!bufferA || !bufferB || !bufferC) {
            if (bufferA) pool_release_buffer(bufferA);
            if (bufferC) pool_release_buffer(bufferC);
            return;
        }

        memcpy([bufferA contents], A, sizeA);
        if (beta != 0.0f) {
            memcpy([bufferC contents], C, sizeC);
        }

        /* Create matrix descriptors - B uses Float16 (converted from bf16) */
        MPSMatrixDescriptor *descA = [MPSMatrixDescriptor
            matrixDescriptorWithRows:rowsA columns:colsA
                            rowBytes:lda * sizeof(float)
                            dataType:MPSDataTypeFloat32];

        MPSMatrixDescriptor *descB = [MPSMatrixDescriptor
            matrixDescriptorWithRows:rowsB columns:colsB
                            rowBytes:ldb * sizeof(uint16_t)
                            dataType:MPSDataTypeFloat16];

        MPSMatrixDescriptor *descC = [MPSMatrixDescriptor
            matrixDescriptorWithRows:M columns:N
                            rowBytes:ldc * sizeof(float)
                            dataType:MPSDataTypeFloat32];

        MPSMatrix *matrixA = [[MPSMatrix alloc] initWithBuffer:bufferA descriptor:descA];
        MPSMatrix *matrixB = [[MPSMatrix alloc] initWithBuffer:bufferB descriptor:descB];
        MPSMatrix *matrixC = [[MPSMatrix alloc] initWithBuffer:bufferC descriptor:descC];

        MPSMatrixMultiplication *matmul = [[MPSMatrixMultiplication alloc]
            initWithDevice:g_device
               transposeLeft:transpose_a ? YES : NO
              transposeRight:transpose_b ? YES : NO
                  resultRows:M
               resultColumns:N
             interiorColumns:K
                       alpha:alpha
                        beta:beta];

        id<MTLCommandBuffer> cmdBuffer = g_in_batch ? g_batch_cmd : [g_queue commandBuffer];

        [matmul encodeToCommandBuffer:cmdBuffer
                           leftMatrix:matrixA
                          rightMatrix:matrixB
                         resultMatrix:matrixC];

        if (g_in_batch) {
            if (g_pending_count < MAX_BATCH_OUTPUTS) {
                g_pending_outputs[g_pending_count].buffer = bufferC;
                g_pending_outputs[g_pending_count].cpu_ptr = C;
                g_pending_outputs[g_pending_count].size = sizeC;
                g_pending_count++;
                pool_release_buffer(bufferA);
            } else {
                [cmdBuffer commit];
                [cmdBuffer waitUntilCompleted];
                memcpy(C, [bufferC contents], sizeC);
                pool_release_buffer(bufferA);
                pool_release_buffer(bufferC);
            }
        } else {
            [cmdBuffer commit];
            [cmdBuffer waitUntilCompleted];
            memcpy(C, [bufferC contents], sizeC);
            pool_release_buffer(bufferA);
            pool_release_buffer(bufferC);
        }
    }
}

void flux_metal_sgemm_batch(int transpose_a, int transpose_b,
                            int M, int N, int K,
                            float alpha,
                            const float *A, int lda, int stride_a,
                            const float *B, int ldb, int stride_b,
                            float beta,
                            float *C, int ldc, int stride_c,
                            int batch_count) {
    if (!g_initialized || batch_count <= 0) return;

    @autoreleasepool {
        /* For batched ops, encode all into single command buffer */
        id<MTLCommandBuffer> cmdBuffer = [g_queue commandBuffer];

        int rowsA = transpose_a ? K : M;
        int colsA = transpose_a ? M : K;
        int rowsB = transpose_b ? N : K;
        int colsB = transpose_b ? K : N;

        /* Create descriptors once */
        MPSMatrixDescriptor *descA = [MPSMatrixDescriptor
            matrixDescriptorWithRows:rowsA columns:colsA
                            rowBytes:lda * sizeof(float)
                            dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor *descB = [MPSMatrixDescriptor
            matrixDescriptorWithRows:rowsB columns:colsB
                            rowBytes:ldb * sizeof(float)
                            dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor *descC = [MPSMatrixDescriptor
            matrixDescriptorWithRows:M columns:N
                            rowBytes:ldc * sizeof(float)
                            dataType:MPSDataTypeFloat32];

        /* Create kernel once */
        MPSMatrixMultiplication *matmul = [[MPSMatrixMultiplication alloc]
            initWithDevice:g_device
               transposeLeft:transpose_a ? YES : NO
              transposeRight:transpose_b ? YES : NO
                  resultRows:M resultColumns:N interiorColumns:K
                       alpha:alpha beta:beta];

        size_t sizeA_elem = (size_t)rowsA * lda * sizeof(float);
        size_t sizeB_elem = (size_t)rowsB * ldb * sizeof(float);
        size_t sizeC_elem = (size_t)M * ldc * sizeof(float);

        /* Store C buffers so we can copy results back after GPU completes */
        __strong id<MTLBuffer> *cBuffers = (__strong id<MTLBuffer> *)calloc(batch_count, sizeof(id<MTLBuffer>));
        float **cPtrs = (float **)malloc(batch_count * sizeof(float *));

        for (int i = 0; i < batch_count; i++) {
            const float *Ai = A + i * stride_a;
            const float *Bi = B + i * stride_b;
            float *Ci = C + i * stride_c;

            /* Use copy-based buffers to avoid alignment issues */
            id<MTLBuffer> bufA = [g_device newBufferWithBytes:Ai
                                                       length:sizeA_elem
                                                      options:MTLResourceStorageModeShared];
            id<MTLBuffer> bufB = get_cached_weight_buffer(Bi, sizeB_elem);
            id<MTLBuffer> bufC = [g_device newBufferWithLength:sizeC_elem
                                                       options:MTLResourceStorageModeShared];

            cBuffers[i] = bufC;
            cPtrs[i] = Ci;

            MPSMatrix *matA = [[MPSMatrix alloc] initWithBuffer:bufA descriptor:descA];
            MPSMatrix *matB = [[MPSMatrix alloc] initWithBuffer:bufB descriptor:descB];
            MPSMatrix *matC = [[MPSMatrix alloc] initWithBuffer:bufC descriptor:descC];

            [matmul encodeToCommandBuffer:cmdBuffer
                               leftMatrix:matA
                              rightMatrix:matB
                             resultMatrix:matC];
        }

        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];

        /* Copy results back and release buffers */
        for (int i = 0; i < batch_count; i++) {
            memcpy(cPtrs[i], [cBuffers[i] contents], sizeC_elem);
            cBuffers[i] = nil;  /* Release under ARC */
        }

        free(cBuffers);
        free(cPtrs);
    }
}

void flux_metal_sync(void) {
    /* All operations are currently synchronous */
}

size_t flux_metal_memory_used(void) {
    if (!g_initialized || !g_device) return 0;
    return [g_device currentAllocatedSize];
}

/* External softmax function from flux_kernels.c */
extern void flux_softmax(float *x, int rows, int cols);

/*
 * GPU-accelerated attention with batched heads.
 * Does: out = softmax(Q @ K^T * scale) @ V for all heads in parallel.
 *
 * Q: [heads, seq_q, head_dim]
 * K: [heads, seq_k, head_dim]
 * V: [heads, seq_k, head_dim]
 * scores_scratch: [heads * seq_q * seq_k]
 * out: [heads, seq_q, head_dim]
 */
void flux_metal_attention(float *out,
                          const float *Q, const float *K, const float *V,
                          float *scores_scratch,
                          int heads, int seq_q, int seq_k, int head_dim,
                          float scale) {
    if (!g_initialized || heads <= 0) return;

    @autoreleasepool {
        size_t q_stride = (size_t)seq_q * head_dim;
        size_t k_stride = (size_t)seq_k * head_dim;
        size_t v_stride = (size_t)seq_k * head_dim;
        size_t scores_stride = (size_t)seq_q * seq_k;
        size_t out_stride = (size_t)seq_q * head_dim;

        size_t sizeQ = heads * q_stride * sizeof(float);
        size_t sizeK = heads * k_stride * sizeof(float);
        size_t sizeV = heads * v_stride * sizeof(float);
        size_t sizeScores = heads * scores_stride * sizeof(float);
        size_t sizeOut = heads * out_stride * sizeof(float);

        /* Create GPU buffers using shared memory (zero-copy on Apple Silicon) */
        id<MTLBuffer> bufQ = [g_device newBufferWithBytes:Q length:sizeQ
                                                  options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufK = [g_device newBufferWithBytes:K length:sizeK
                                                  options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufV = [g_device newBufferWithBytes:V length:sizeV
                                                  options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufScores = [g_device newBufferWithLength:sizeScores
                                                        options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufOut = [g_device newBufferWithLength:sizeOut
                                                     options:MTLResourceStorageModeShared];

        if (!bufQ || !bufK || !bufV || !bufScores || !bufOut) {
            return;
        }

        /* === Phase 1: Batched Q @ K^T === */
        {
            id<MTLCommandBuffer> cmdBuffer = [g_queue commandBuffer];

            /* Descriptors for single head matrices */
            MPSMatrixDescriptor *descQ = [MPSMatrixDescriptor
                matrixDescriptorWithRows:seq_q columns:head_dim
                                rowBytes:head_dim * sizeof(float)
                                dataType:MPSDataTypeFloat32];
            MPSMatrixDescriptor *descK = [MPSMatrixDescriptor
                matrixDescriptorWithRows:seq_k columns:head_dim
                                rowBytes:head_dim * sizeof(float)
                                dataType:MPSDataTypeFloat32];
            MPSMatrixDescriptor *descScores = [MPSMatrixDescriptor
                matrixDescriptorWithRows:seq_q columns:seq_k
                                rowBytes:seq_k * sizeof(float)
                                dataType:MPSDataTypeFloat32];

            /* Create matmul kernel: scores = scale * Q @ K^T */
            MPSMatrixMultiplication *matmul_qk = [[MPSMatrixMultiplication alloc]
                initWithDevice:g_device
                   transposeLeft:NO
                  transposeRight:YES
                      resultRows:seq_q
                   resultColumns:seq_k
                 interiorColumns:head_dim
                           alpha:scale
                            beta:0.0f];

            /* Encode all heads */
            for (int h = 0; h < heads; h++) {
                size_t offsetQ = h * q_stride * sizeof(float);
                size_t offsetK = h * k_stride * sizeof(float);
                size_t offsetS = h * scores_stride * sizeof(float);

                MPSMatrix *matQ = [[MPSMatrix alloc]
                    initWithBuffer:bufQ offset:offsetQ descriptor:descQ];
                MPSMatrix *matK = [[MPSMatrix alloc]
                    initWithBuffer:bufK offset:offsetK descriptor:descK];
                MPSMatrix *matScores = [[MPSMatrix alloc]
                    initWithBuffer:bufScores offset:offsetS descriptor:descScores];

                [matmul_qk encodeToCommandBuffer:cmdBuffer
                                      leftMatrix:matQ
                                     rightMatrix:matK
                                    resultMatrix:matScores];
            }

            [cmdBuffer commit];
            [cmdBuffer waitUntilCompleted];
        }

        /* Copy scores to CPU scratch buffer */
        memcpy(scores_scratch, [bufScores contents], sizeScores);

        /* === Phase 2: Softmax on CPU (per head, per row) === */
        for (int h = 0; h < heads; h++) {
            flux_softmax(scores_scratch + h * scores_stride, seq_q, seq_k);
        }

        /* Copy softmax results back to GPU */
        memcpy([bufScores contents], scores_scratch, sizeScores);

        /* === Phase 3: Batched scores @ V === */
        {
            id<MTLCommandBuffer> cmdBuffer = [g_queue commandBuffer];

            MPSMatrixDescriptor *descScores = [MPSMatrixDescriptor
                matrixDescriptorWithRows:seq_q columns:seq_k
                                rowBytes:seq_k * sizeof(float)
                                dataType:MPSDataTypeFloat32];
            MPSMatrixDescriptor *descV = [MPSMatrixDescriptor
                matrixDescriptorWithRows:seq_k columns:head_dim
                                rowBytes:head_dim * sizeof(float)
                                dataType:MPSDataTypeFloat32];
            MPSMatrixDescriptor *descOut = [MPSMatrixDescriptor
                matrixDescriptorWithRows:seq_q columns:head_dim
                                rowBytes:head_dim * sizeof(float)
                                dataType:MPSDataTypeFloat32];

            /* Create matmul kernel: out = scores @ V */
            MPSMatrixMultiplication *matmul_sv = [[MPSMatrixMultiplication alloc]
                initWithDevice:g_device
                   transposeLeft:NO
                  transposeRight:NO
                      resultRows:seq_q
                   resultColumns:head_dim
                 interiorColumns:seq_k
                           alpha:1.0f
                            beta:0.0f];

            /* Encode all heads */
            for (int h = 0; h < heads; h++) {
                size_t offsetS = h * scores_stride * sizeof(float);
                size_t offsetV = h * v_stride * sizeof(float);
                size_t offsetO = h * out_stride * sizeof(float);

                MPSMatrix *matScores = [[MPSMatrix alloc]
                    initWithBuffer:bufScores offset:offsetS descriptor:descScores];
                MPSMatrix *matV = [[MPSMatrix alloc]
                    initWithBuffer:bufV offset:offsetV descriptor:descV];
                MPSMatrix *matOut = [[MPSMatrix alloc]
                    initWithBuffer:bufOut offset:offsetO descriptor:descOut];

                [matmul_sv encodeToCommandBuffer:cmdBuffer
                                      leftMatrix:matScores
                                     rightMatrix:matV
                                    resultMatrix:matOut];
            }

            [cmdBuffer commit];
            [cmdBuffer waitUntilCompleted];
        }

        /* Copy output back to CPU */
        memcpy(out, [bufOut contents], sizeOut);
    }
}

/* ========================================================================
 * GPU Tensor API Implementation
 * ======================================================================== */

/* Internal tensor structure */
struct flux_gpu_tensor {
    id<MTLBuffer> buffer;
    size_t num_elements;
    int has_pending_work;  /* Flag to track if GPU work is pending */
};

/* Pending command buffer for batched operations */
static id<MTLCommandBuffer> g_tensor_cmd = nil;
static int g_tensor_batch_mode = 0;

flux_gpu_tensor_t flux_gpu_tensor_create(const float *data, size_t num_elements) {
    if (!g_initialized || !data || num_elements == 0) return NULL;

    @autoreleasepool {
        size_t size = num_elements * sizeof(float);

        /* Get buffer from pool */
        id<MTLBuffer> buf = pool_get_buffer(size);
        if (!buf) return NULL;

        /* Copy data to buffer (shared memory - this is fast on Apple Silicon) */
        memcpy([buf contents], data, size);

        /* Allocate tensor structure */
        flux_gpu_tensor_t tensor = (flux_gpu_tensor_t)malloc(sizeof(struct flux_gpu_tensor));
        if (!tensor) {
            pool_release_buffer(buf);
            return NULL;
        }

        tensor->buffer = buf;
        tensor->num_elements = num_elements;
        tensor->has_pending_work = 0;

        return tensor;
    }
}

flux_gpu_tensor_t flux_gpu_tensor_alloc(size_t num_elements) {
    if (!g_initialized || num_elements == 0) return NULL;

    @autoreleasepool {
        size_t size = num_elements * sizeof(float);

        /* Get buffer from pool */
        id<MTLBuffer> buf = pool_get_buffer(size);
        if (!buf) return NULL;

        /* Allocate tensor structure */
        flux_gpu_tensor_t tensor = (flux_gpu_tensor_t)malloc(sizeof(struct flux_gpu_tensor));
        if (!tensor) {
            pool_release_buffer(buf);
            return NULL;
        }

        tensor->buffer = buf;
        tensor->num_elements = num_elements;
        tensor->has_pending_work = 0;

        return tensor;
    }
}

void flux_gpu_tensor_read(flux_gpu_tensor_t tensor, float *out) {
    if (!tensor || !out) return;

    /* If there's pending work, sync first */
    if (tensor->has_pending_work) {
        flux_gpu_sync();
        tensor->has_pending_work = 0;
    }

    /* Copy from shared memory buffer */
    size_t size = tensor->num_elements * sizeof(float);
    memcpy(out, [tensor->buffer contents], size);
}

float *flux_gpu_tensor_data(flux_gpu_tensor_t tensor) {
    if (!tensor) return NULL;
    return (float *)[tensor->buffer contents];
}

void flux_gpu_tensor_free(flux_gpu_tensor_t tensor) {
    if (!tensor) return;

    /* Release buffer back to pool */
    pool_release_buffer(tensor->buffer);
    tensor->buffer = nil;

    free(tensor);
}

size_t flux_gpu_tensor_size(flux_gpu_tensor_t tensor) {
    if (!tensor) return 0;
    return tensor->num_elements;
}

void flux_gpu_sync(void) {
    if (!g_initialized) return;

    @autoreleasepool {
        if (g_tensor_cmd) {
            [g_tensor_cmd commit];
            [g_tensor_cmd waitUntilCompleted];
            g_tensor_cmd = nil;
        }
    }
}

void flux_gpu_batch_begin(void) {
    if (!g_initialized || g_tensor_batch_mode) return;

    @autoreleasepool {
        g_tensor_cmd = [g_queue commandBuffer];
        g_tensor_batch_mode = 1;
    }
}

void flux_gpu_batch_end(void) {
    if (!g_initialized || !g_tensor_batch_mode) return;

    @autoreleasepool {
        if (g_tensor_cmd) {
            [g_tensor_cmd commit];
            [g_tensor_cmd waitUntilCompleted];
            g_tensor_cmd = nil;
        }
        g_tensor_batch_mode = 0;
    }
}

/* Get or create command buffer for tensor operations */
static id<MTLCommandBuffer> get_tensor_cmd(void) {
    if (g_tensor_batch_mode && g_tensor_cmd) {
        return g_tensor_cmd;
    }
    return [g_queue commandBuffer];
}

flux_gpu_tensor_t flux_gpu_linear(flux_gpu_tensor_t x,
                                   const float *W, const float *b,
                                   int seq_len, int in_dim, int out_dim) {
    if (!g_initialized || !x || !W) return NULL;

    @autoreleasepool {
        size_t out_elements = (size_t)seq_len * out_dim;
        flux_gpu_tensor_t out = flux_gpu_tensor_alloc(out_elements);
        if (!out) return NULL;

        /* Get weight buffer (cached) */
        size_t sizeW = (size_t)out_dim * in_dim * sizeof(float);
        id<MTLBuffer> bufW = get_cached_weight_buffer(W, sizeW);
        if (!bufW) {
            flux_gpu_tensor_free(out);
            return NULL;
        }

        /* Create matrix descriptors
         * x: [seq_len, in_dim]
         * W: [out_dim, in_dim] (need to transpose for x @ W^T)
         * out: [seq_len, out_dim]
         */
        MPSMatrixDescriptor *descX = [MPSMatrixDescriptor
            matrixDescriptorWithRows:seq_len columns:in_dim
                            rowBytes:in_dim * sizeof(float)
                            dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor *descW = [MPSMatrixDescriptor
            matrixDescriptorWithRows:out_dim columns:in_dim
                            rowBytes:in_dim * sizeof(float)
                            dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor *descOut = [MPSMatrixDescriptor
            matrixDescriptorWithRows:seq_len columns:out_dim
                            rowBytes:out_dim * sizeof(float)
                            dataType:MPSDataTypeFloat32];

        MPSMatrix *matX = [[MPSMatrix alloc] initWithBuffer:x->buffer descriptor:descX];
        MPSMatrix *matW = [[MPSMatrix alloc] initWithBuffer:bufW descriptor:descW];
        MPSMatrix *matOut = [[MPSMatrix alloc] initWithBuffer:out->buffer descriptor:descOut];

        /* Create matmul: out = x @ W^T */
        MPSMatrixMultiplication *matmul = [[MPSMatrixMultiplication alloc]
            initWithDevice:g_device
               transposeLeft:NO
              transposeRight:YES
                  resultRows:seq_len
               resultColumns:out_dim
             interiorColumns:in_dim
                       alpha:1.0f
                        beta:0.0f];

        id<MTLCommandBuffer> cmdBuffer = get_tensor_cmd();
        [matmul encodeToCommandBuffer:cmdBuffer
                           leftMatrix:matX
                          rightMatrix:matW
                         resultMatrix:matOut];

        /* Add bias if present */
        if (b != NULL) {
            /* For now, sync and add bias on CPU (can optimize later with compute shader) */
            [cmdBuffer commit];
            [cmdBuffer waitUntilCompleted];

            float *out_data = (float *)[out->buffer contents];
            for (int i = 0; i < seq_len; i++) {
                for (int j = 0; j < out_dim; j++) {
                    out_data[i * out_dim + j] += b[j];
                }
            }
            out->has_pending_work = 0;
        } else {
            /* Mark output as having pending work */
            out->has_pending_work = 1;

            if (!g_tensor_batch_mode) {
                /* Not in batch mode - sync immediately */
                [cmdBuffer commit];
                [cmdBuffer waitUntilCompleted];
                out->has_pending_work = 0;
            }
        }

        /* Mark input as having pending work if in batch mode */
        if (g_tensor_batch_mode) {
            x->has_pending_work = 1;
        }

        return out;
    }
}

/* ========================================================================
 * Compute Shader Support
 * Custom Metal compute shaders for element-wise operations.
 * ======================================================================== */

/* Compute pipeline states */
static id<MTLLibrary> g_shader_library = nil;
static id<MTLComputePipelineState> g_rms_norm_pipeline = nil;
static id<MTLComputePipelineState> g_qk_rms_norm_pipeline = nil;
static id<MTLComputePipelineState> g_adaln_norm_pipeline = nil;
static id<MTLComputePipelineState> g_silu_pipeline = nil;
static id<MTLComputePipelineState> g_silu_mul_pipeline = nil;
static id<MTLComputePipelineState> g_softmax_pipeline = nil;
static id<MTLComputePipelineState> g_rope_2d_pipeline = nil;
static int g_shaders_initialized = 0;

int flux_metal_shaders_available(void) {
    return g_shaders_initialized;
}

int flux_metal_init_shaders(void) {
    if (g_shaders_initialized) return 1;
    if (!g_initialized) return 0;

    @autoreleasepool {
        NSError *error = nil;

        /* Try to find the shader file in various locations */
        NSString *shaderPath = nil;
        NSArray *searchPaths = @[
            @"flux_shaders.metal",
            @"./flux_shaders.metal",
            [[NSBundle mainBundle] pathForResource:@"flux_shaders" ofType:@"metal"],
        ];

        for (NSString *path in searchPaths) {
            if (path && [[NSFileManager defaultManager] fileExistsAtPath:path]) {
                shaderPath = path;
                break;
            }
        }

        if (!shaderPath) {
            /* Try executable directory */
            NSString *execPath = [[NSBundle mainBundle] executablePath];
            if (execPath) {
                NSString *execDir = [execPath stringByDeletingLastPathComponent];
                NSString *path = [execDir stringByAppendingPathComponent:@"flux_shaders.metal"];
                if ([[NSFileManager defaultManager] fileExistsAtPath:path]) {
                    shaderPath = path;
                }
            }
        }

        if (!shaderPath) {
            fprintf(stderr, "Metal shaders: flux_shaders.metal not found\n");
            return 0;
        }

        /* Load shader source */
        NSString *shaderSource = [NSString stringWithContentsOfFile:shaderPath
                                                           encoding:NSUTF8StringEncoding
                                                              error:&error];
        if (!shaderSource) {
            fprintf(stderr, "Metal shaders: failed to read %s: %s\n",
                    [shaderPath UTF8String], [[error localizedDescription] UTF8String]);
            return 0;
        }

        /* Compile shader library */
        MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
        options.fastMathEnabled = YES;

        g_shader_library = [g_device newLibraryWithSource:shaderSource
                                                  options:options
                                                    error:&error];
        if (!g_shader_library) {
            fprintf(stderr, "Metal shaders: compilation failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            return 0;
        }

        /* Create compute pipeline states for each kernel */
        id<MTLFunction> func;

        func = [g_shader_library newFunctionWithName:@"rms_norm"];
        if (func) {
            g_rms_norm_pipeline = [g_device newComputePipelineStateWithFunction:func error:&error];
            if (!g_rms_norm_pipeline) {
                fprintf(stderr, "Metal shaders: rms_norm pipeline failed: %s\n",
                        [[error localizedDescription] UTF8String]);
            }
        }

        func = [g_shader_library newFunctionWithName:@"qk_rms_norm"];
        if (func) {
            g_qk_rms_norm_pipeline = [g_device newComputePipelineStateWithFunction:func error:&error];
            if (!g_qk_rms_norm_pipeline) {
                fprintf(stderr, "Metal shaders: qk_rms_norm pipeline failed: %s\n",
                        [[error localizedDescription] UTF8String]);
            }
        }

        func = [g_shader_library newFunctionWithName:@"adaln_norm"];
        if (func) {
            g_adaln_norm_pipeline = [g_device newComputePipelineStateWithFunction:func error:&error];
            if (!g_adaln_norm_pipeline) {
                fprintf(stderr, "Metal shaders: adaln_norm pipeline failed: %s\n",
                        [[error localizedDescription] UTF8String]);
            }
        }

        func = [g_shader_library newFunctionWithName:@"silu"];
        if (func) {
            g_silu_pipeline = [g_device newComputePipelineStateWithFunction:func error:&error];
            if (!g_silu_pipeline) {
                fprintf(stderr, "Metal shaders: silu pipeline failed: %s\n",
                        [[error localizedDescription] UTF8String]);
            }
        }

        func = [g_shader_library newFunctionWithName:@"silu_mul"];
        if (func) {
            g_silu_mul_pipeline = [g_device newComputePipelineStateWithFunction:func error:&error];
            if (!g_silu_mul_pipeline) {
                fprintf(stderr, "Metal shaders: silu_mul pipeline failed: %s\n",
                        [[error localizedDescription] UTF8String]);
            }
        }

        func = [g_shader_library newFunctionWithName:@"softmax"];
        if (func) {
            g_softmax_pipeline = [g_device newComputePipelineStateWithFunction:func error:&error];
            if (!g_softmax_pipeline) {
                fprintf(stderr, "Metal shaders: softmax pipeline failed: %s\n",
                        [[error localizedDescription] UTF8String]);
            }
        }

        func = [g_shader_library newFunctionWithName:@"apply_rope_2d"];
        if (func) {
            g_rope_2d_pipeline = [g_device newComputePipelineStateWithFunction:func error:&error];
            if (!g_rope_2d_pipeline) {
                fprintf(stderr, "Metal shaders: apply_rope_2d pipeline failed: %s\n",
                        [[error localizedDescription] UTF8String]);
            }
        }

        g_shaders_initialized = 1;
        fprintf(stderr, "Metal shaders: compute kernels loaded\n");
        return 1;
    }
}

/* Helper to encode a compute shader with common setup */
static id<MTLCommandBuffer> encode_compute_shader(void) {
    return g_in_batch ? g_batch_cmd : [g_queue commandBuffer];
}

void flux_metal_rms_norm(float *out, const float *x, const float *weight,
                         int seq_len, int hidden, float eps) {
    if (!g_shaders_initialized || !g_rms_norm_pipeline) return;

    @autoreleasepool {
        size_t data_size = (size_t)seq_len * hidden * sizeof(float);
        size_t weight_size = (size_t)hidden * sizeof(float);

        /* Create buffers */
        id<MTLBuffer> bufX = pool_get_buffer(data_size);
        id<MTLBuffer> bufWeight = get_cached_weight_buffer(weight, weight_size);
        id<MTLBuffer> bufOut = pool_get_buffer(data_size);

        if (!bufX || !bufWeight || !bufOut) {
            if (bufX) pool_release_buffer(bufX);
            if (bufOut) pool_release_buffer(bufOut);
            return;
        }

        memcpy([bufX contents], x, data_size);

        id<MTLCommandBuffer> cmdBuffer = encode_compute_shader();
        id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

        [encoder setComputePipelineState:g_rms_norm_pipeline];
        [encoder setBuffer:bufX offset:0 atIndex:0];
        [encoder setBuffer:bufWeight offset:0 atIndex:1];
        [encoder setBuffer:bufOut offset:0 atIndex:2];
        [encoder setBytes:&hidden length:sizeof(int) atIndex:3];
        [encoder setBytes:&eps length:sizeof(float) atIndex:4];

        /* One threadgroup per row, 256 threads per group */
        NSUInteger threadsPerGroup = MIN(256, (NSUInteger)hidden);
        [encoder dispatchThreadgroups:MTLSizeMake(seq_len, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(threadsPerGroup, 1, 1)];

        [encoder endEncoding];

        if (!g_in_batch) {
            [cmdBuffer commit];
            [cmdBuffer waitUntilCompleted];
            memcpy(out, [bufOut contents], data_size);
            pool_release_buffer(bufX);
            pool_release_buffer(bufOut);
        } else {
            if (g_pending_count < MAX_BATCH_OUTPUTS) {
                g_pending_outputs[g_pending_count].buffer = bufOut;
                g_pending_outputs[g_pending_count].cpu_ptr = out;
                g_pending_outputs[g_pending_count].size = data_size;
                g_pending_count++;
            }
            pool_release_buffer(bufX);
        }
    }
}

void flux_metal_qk_rms_norm(float *q, float *k,
                            const float *q_weight, const float *k_weight,
                            int seq, int heads, int head_dim, float eps) {
    if (!g_shaders_initialized || !g_qk_rms_norm_pipeline) return;

    @autoreleasepool {
        size_t data_size = (size_t)seq * heads * head_dim * sizeof(float);
        size_t weight_size = (size_t)head_dim * sizeof(float);

        /* Create buffers - Q and K are modified in-place */
        id<MTLBuffer> bufQ = pool_get_buffer(data_size);
        id<MTLBuffer> bufK = pool_get_buffer(data_size);
        id<MTLBuffer> bufQWeight = get_cached_weight_buffer(q_weight, weight_size);
        id<MTLBuffer> bufKWeight = get_cached_weight_buffer(k_weight, weight_size);

        if (!bufQ || !bufK || !bufQWeight || !bufKWeight) {
            if (bufQ) pool_release_buffer(bufQ);
            if (bufK) pool_release_buffer(bufK);
            return;
        }

        memcpy([bufQ contents], q, data_size);
        memcpy([bufK contents], k, data_size);

        id<MTLCommandBuffer> cmdBuffer = encode_compute_shader();
        id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

        [encoder setComputePipelineState:g_qk_rms_norm_pipeline];
        [encoder setBuffer:bufQ offset:0 atIndex:0];
        [encoder setBuffer:bufK offset:0 atIndex:1];
        [encoder setBuffer:bufQWeight offset:0 atIndex:2];
        [encoder setBuffer:bufKWeight offset:0 atIndex:3];
        [encoder setBytes:&heads length:sizeof(int) atIndex:4];
        [encoder setBytes:&head_dim length:sizeof(int) atIndex:5];
        [encoder setBytes:&eps length:sizeof(float) atIndex:6];

        /* One thread per (seq_idx, head_idx) pair */
        [encoder dispatchThreads:MTLSizeMake(seq, heads, 1)
           threadsPerThreadgroup:MTLSizeMake(1, MIN(heads, 64), 1)];

        [encoder endEncoding];

        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];

        memcpy(q, [bufQ contents], data_size);
        memcpy(k, [bufK contents], data_size);

        pool_release_buffer(bufQ);
        pool_release_buffer(bufK);
    }
}

void flux_metal_adaln_norm(float *out, const float *x,
                           const float *shift, const float *scale,
                           int seq_len, int hidden, float eps) {
    if (!g_shaders_initialized || !g_adaln_norm_pipeline) return;

    @autoreleasepool {
        size_t data_size = (size_t)seq_len * hidden * sizeof(float);
        size_t param_size = (size_t)hidden * sizeof(float);

        id<MTLBuffer> bufX = pool_get_buffer(data_size);
        id<MTLBuffer> bufShift = pool_get_buffer(param_size);
        id<MTLBuffer> bufScale = pool_get_buffer(param_size);
        id<MTLBuffer> bufOut = pool_get_buffer(data_size);

        if (!bufX || !bufShift || !bufScale || !bufOut) {
            if (bufX) pool_release_buffer(bufX);
            if (bufShift) pool_release_buffer(bufShift);
            if (bufScale) pool_release_buffer(bufScale);
            if (bufOut) pool_release_buffer(bufOut);
            return;
        }

        memcpy([bufX contents], x, data_size);
        memcpy([bufShift contents], shift, param_size);
        memcpy([bufScale contents], scale, param_size);

        id<MTLCommandBuffer> cmdBuffer = encode_compute_shader();
        id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

        [encoder setComputePipelineState:g_adaln_norm_pipeline];
        [encoder setBuffer:bufX offset:0 atIndex:0];
        [encoder setBuffer:bufShift offset:0 atIndex:1];
        [encoder setBuffer:bufScale offset:0 atIndex:2];
        [encoder setBuffer:bufOut offset:0 atIndex:3];
        [encoder setBytes:&hidden length:sizeof(int) atIndex:4];
        [encoder setBytes:&eps length:sizeof(float) atIndex:5];

        NSUInteger threadsPerGroup = MIN(256, (NSUInteger)hidden);
        [encoder dispatchThreadgroups:MTLSizeMake(seq_len, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(threadsPerGroup, 1, 1)];

        [encoder endEncoding];

        if (!g_in_batch) {
            [cmdBuffer commit];
            [cmdBuffer waitUntilCompleted];
            memcpy(out, [bufOut contents], data_size);
            pool_release_buffer(bufX);
            pool_release_buffer(bufShift);
            pool_release_buffer(bufScale);
            pool_release_buffer(bufOut);
        } else {
            if (g_pending_count < MAX_BATCH_OUTPUTS) {
                g_pending_outputs[g_pending_count].buffer = bufOut;
                g_pending_outputs[g_pending_count].cpu_ptr = out;
                g_pending_outputs[g_pending_count].size = data_size;
                g_pending_count++;
            }
            pool_release_buffer(bufX);
            pool_release_buffer(bufShift);
            pool_release_buffer(bufScale);
        }
    }
}

void flux_metal_silu(float *x, int n) {
    if (!g_shaders_initialized || !g_silu_pipeline || n <= 0) return;

    @autoreleasepool {
        size_t data_size = (size_t)n * sizeof(float);

        id<MTLBuffer> bufX = pool_get_buffer(data_size);
        if (!bufX) return;

        memcpy([bufX contents], x, data_size);

        id<MTLCommandBuffer> cmdBuffer = encode_compute_shader();
        id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

        [encoder setComputePipelineState:g_silu_pipeline];
        [encoder setBuffer:bufX offset:0 atIndex:0];
        [encoder setBytes:&n length:sizeof(int) atIndex:1];

        NSUInteger threadsPerGroup = [g_silu_pipeline maxTotalThreadsPerThreadgroup];
        NSUInteger threadGroups = (n + threadsPerGroup - 1) / threadsPerGroup;
        [encoder dispatchThreadgroups:MTLSizeMake(threadGroups, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(threadsPerGroup, 1, 1)];

        [encoder endEncoding];

        if (!g_in_batch) {
            [cmdBuffer commit];
            [cmdBuffer waitUntilCompleted];
            memcpy(x, [bufX contents], data_size);
            pool_release_buffer(bufX);
        } else {
            if (g_pending_count < MAX_BATCH_OUTPUTS) {
                g_pending_outputs[g_pending_count].buffer = bufX;
                g_pending_outputs[g_pending_count].cpu_ptr = x;
                g_pending_outputs[g_pending_count].size = data_size;
                g_pending_count++;
            }
        }
    }
}

void flux_metal_silu_mul(float *gate, const float *up, int n) {
    if (!g_shaders_initialized || !g_silu_mul_pipeline || n <= 0) return;

    @autoreleasepool {
        size_t data_size = (size_t)n * sizeof(float);

        id<MTLBuffer> bufGate = pool_get_buffer(data_size);
        id<MTLBuffer> bufUp = pool_get_buffer(data_size);
        if (!bufGate || !bufUp) {
            if (bufGate) pool_release_buffer(bufGate);
            if (bufUp) pool_release_buffer(bufUp);
            return;
        }

        memcpy([bufGate contents], gate, data_size);
        memcpy([bufUp contents], up, data_size);

        id<MTLCommandBuffer> cmdBuffer = encode_compute_shader();
        id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

        [encoder setComputePipelineState:g_silu_mul_pipeline];
        [encoder setBuffer:bufGate offset:0 atIndex:0];
        [encoder setBuffer:bufUp offset:0 atIndex:1];
        [encoder setBytes:&n length:sizeof(int) atIndex:2];

        NSUInteger threadsPerGroup = [g_silu_mul_pipeline maxTotalThreadsPerThreadgroup];
        NSUInteger threadGroups = (n + threadsPerGroup - 1) / threadsPerGroup;
        [encoder dispatchThreadgroups:MTLSizeMake(threadGroups, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(threadsPerGroup, 1, 1)];

        [encoder endEncoding];

        if (!g_in_batch) {
            [cmdBuffer commit];
            [cmdBuffer waitUntilCompleted];
            memcpy(gate, [bufGate contents], data_size);
            pool_release_buffer(bufGate);
            pool_release_buffer(bufUp);
        } else {
            if (g_pending_count < MAX_BATCH_OUTPUTS) {
                g_pending_outputs[g_pending_count].buffer = bufGate;
                g_pending_outputs[g_pending_count].cpu_ptr = gate;
                g_pending_outputs[g_pending_count].size = data_size;
                g_pending_count++;
            }
            pool_release_buffer(bufUp);
        }
    }
}

void flux_metal_softmax(float *x, int rows, int cols) {
    if (!g_shaders_initialized || !g_softmax_pipeline || rows <= 0 || cols <= 0) return;

    @autoreleasepool {
        size_t data_size = (size_t)rows * cols * sizeof(float);

        id<MTLBuffer> bufX = pool_get_buffer(data_size);
        if (!bufX) return;

        memcpy([bufX contents], x, data_size);

        id<MTLCommandBuffer> cmdBuffer = encode_compute_shader();
        id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

        [encoder setComputePipelineState:g_softmax_pipeline];
        [encoder setBuffer:bufX offset:0 atIndex:0];
        [encoder setBytes:&rows length:sizeof(int) atIndex:1];
        [encoder setBytes:&cols length:sizeof(int) atIndex:2];

        /* One threadgroup per row */
        NSUInteger threadsPerGroup = MIN(256, (NSUInteger)cols);
        [encoder dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(threadsPerGroup, 1, 1)];

        [encoder endEncoding];

        if (!g_in_batch) {
            [cmdBuffer commit];
            [cmdBuffer waitUntilCompleted];
            memcpy(x, [bufX contents], data_size);
            pool_release_buffer(bufX);
        } else {
            if (g_pending_count < MAX_BATCH_OUTPUTS) {
                g_pending_outputs[g_pending_count].buffer = bufX;
                g_pending_outputs[g_pending_count].cpu_ptr = x;
                g_pending_outputs[g_pending_count].size = data_size;
                g_pending_count++;
            }
        }
    }
}

void flux_metal_rope_2d(float *x, const float *cos_freq, const float *sin_freq,
                        int seq, int heads, int head_dim, int axis_dim) {
    if (!g_shaders_initialized || !g_rope_2d_pipeline) return;

    @autoreleasepool {
        size_t data_size = (size_t)seq * heads * head_dim * sizeof(float);
        size_t freq_size = (size_t)seq * head_dim * sizeof(float);

        id<MTLBuffer> bufX = pool_get_buffer(data_size);
        id<MTLBuffer> bufCos = pool_get_buffer(freq_size);
        id<MTLBuffer> bufSin = pool_get_buffer(freq_size);

        if (!bufX || !bufCos || !bufSin) {
            if (bufX) pool_release_buffer(bufX);
            if (bufCos) pool_release_buffer(bufCos);
            if (bufSin) pool_release_buffer(bufSin);
            return;
        }

        memcpy([bufX contents], x, data_size);
        memcpy([bufCos contents], cos_freq, freq_size);
        memcpy([bufSin contents], sin_freq, freq_size);

        id<MTLCommandBuffer> cmdBuffer = encode_compute_shader();
        id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

        [encoder setComputePipelineState:g_rope_2d_pipeline];
        [encoder setBuffer:bufX offset:0 atIndex:0];
        [encoder setBuffer:bufCos offset:0 atIndex:1];
        [encoder setBuffer:bufSin offset:0 atIndex:2];
        [encoder setBytes:&seq length:sizeof(int) atIndex:3];
        [encoder setBytes:&heads length:sizeof(int) atIndex:4];
        [encoder setBytes:&head_dim length:sizeof(int) atIndex:5];
        [encoder setBytes:&axis_dim length:sizeof(int) atIndex:6];

        /* One thread per (seq, head) pair */
        [encoder dispatchThreads:MTLSizeMake(seq, heads, 1)
           threadsPerThreadgroup:MTLSizeMake(1, MIN(heads, 64), 1)];

        [encoder endEncoding];

        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];

        memcpy(x, [bufX contents], data_size);

        pool_release_buffer(bufX);
        pool_release_buffer(bufCos);
        pool_release_buffer(bufSin);
    }
}
