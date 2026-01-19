/*
 * flux_shaders.metal - Metal compute shaders for FLUX inference
 *
 * These kernels accelerate operations that run on CPU otherwise:
 * - RMSNorm (used in QK normalization)
 * - LayerNorm + AdaLN modulation
 * - SiLU activation
 * - Softmax (row-wise)
 */

#include <metal_stdlib>
using namespace metal;

/* ========================================================================
 * RMSNorm: out[i] = x[i] * rsqrt(mean(x^2) + eps) * weight[i]
 * ======================================================================== */

/* RMSNorm kernel - processes one row per threadgroup
 * x: [seq, hidden], weight: [hidden], out: [seq, hidden]
 */
kernel void rms_norm(
    device const float *x [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant int &hidden [[buffer(3)]],
    constant float &eps [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]
) {
    threadgroup float shared_sum[256];  // For reduction

    device const float *x_row = x + row * hidden;
    device float *out_row = out + row * hidden;

    // Compute partial sum of squares
    float local_sum = 0.0f;
    for (int i = tid; i < hidden; i += threads) {
        float val = x_row[i];
        local_sum += val * val;
    }
    shared_sum[tid] = local_sum;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction for sum
    for (uint stride = threads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Compute RMS inverse
    float rms_inv = rsqrt(shared_sum[0] / float(hidden) + eps);

    // Apply normalization with weight
    for (int i = tid; i < hidden; i += threads) {
        out_row[i] = x_row[i] * rms_inv * weight[i];
    }
}

/* QK RMSNorm - processes Q and K for all heads in a sequence position
 * q: [seq, heads*head_dim], k: [seq, heads*head_dim]
 * q_weight, k_weight: [head_dim]
 */
kernel void qk_rms_norm(
    device float *q [[buffer(0)]],
    device float *k [[buffer(1)]],
    device const float *q_weight [[buffer(2)]],
    device const float *k_weight [[buffer(3)]],
    constant int &heads [[buffer(4)]],
    constant int &head_dim [[buffer(5)]],
    constant float &eps [[buffer(6)]],
    uint2 pos [[thread_position_in_grid]]  // (seq_idx, head_idx)
) {
    uint seq_idx = pos.x;
    uint head_idx = pos.y;

    uint hidden = heads * head_dim;
    uint offset = seq_idx * hidden + head_idx * head_dim;

    // RMSNorm for Q
    float sum_sq = 0.0f;
    for (int d = 0; d < head_dim; d++) {
        float val = q[offset + d];
        sum_sq += val * val;
    }
    float rms_inv = rsqrt(sum_sq / float(head_dim) + eps);
    for (int d = 0; d < head_dim; d++) {
        q[offset + d] = q[offset + d] * rms_inv * q_weight[d];
    }

    // RMSNorm for K
    sum_sq = 0.0f;
    for (int d = 0; d < head_dim; d++) {
        float val = k[offset + d];
        sum_sq += val * val;
    }
    rms_inv = rsqrt(sum_sq / float(head_dim) + eps);
    for (int d = 0; d < head_dim; d++) {
        k[offset + d] = k[offset + d] * rms_inv * k_weight[d];
    }
}

/* ========================================================================
 * LayerNorm + AdaLN modulation
 * out = (1 + scale) * norm(x) + shift
 * where norm(x) = (x - mean) / sqrt(var + eps)
 * ======================================================================== */

kernel void adaln_norm(
    device const float *x [[buffer(0)]],
    device const float *shift [[buffer(1)]],
    device const float *scale [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant int &hidden [[buffer(4)]],
    constant float &eps [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]
) {
    threadgroup float shared_sum[256];
    threadgroup float shared_sum_sq[256];

    device const float *x_row = x + row * hidden;
    device float *out_row = out + row * hidden;

    // Compute partial sums for mean and variance
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    for (int i = tid; i < hidden; i += threads) {
        float val = x_row[i];
        local_sum += val;
        local_sum_sq += val * val;
    }
    shared_sum[tid] = local_sum;
    shared_sum_sq[tid] = local_sum_sq;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction
    for (uint stride = threads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
            shared_sum_sq[tid] += shared_sum_sq[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Compute mean and std_inv
    float mean = shared_sum[0] / float(hidden);
    float var = shared_sum_sq[0] / float(hidden) - mean * mean;
    float std_inv = rsqrt(var + eps);

    // Apply LayerNorm + AdaLN modulation
    for (int i = tid; i < hidden; i += threads) {
        float norm = (x_row[i] - mean) * std_inv;
        out_row[i] = (1.0f + scale[i]) * norm + shift[i];
    }
}

/* ========================================================================
 * SiLU activation: x * sigmoid(x) = x / (1 + exp(-x))
 * ======================================================================== */

kernel void silu(
    device float *x [[buffer(0)]],
    constant int &n [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < uint(n)) {
        float val = x[gid];
        x[gid] = val / (1.0f + exp(-val));
    }
}

/* SiLU with multiply: gate = silu(gate) * up (SwiGLU style) */
kernel void silu_mul(
    device float *gate [[buffer(0)]],
    device const float *up [[buffer(1)]],
    constant int &n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < uint(n)) {
        float g = gate[gid];
        float silu_g = g / (1.0f + exp(-g));
        gate[gid] = silu_g * up[gid];
    }
}

/* ========================================================================
 * Softmax (row-wise): out[i] = exp(x[i] - max) / sum(exp(x - max))
 * ======================================================================== */

kernel void softmax(
    device float *x [[buffer(0)]],
    constant int &rows [[buffer(1)]],
    constant int &cols [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]
) {
    threadgroup float shared_max[256];
    threadgroup float shared_sum[256];

    device float *row_ptr = x + row * cols;

    // Find max (for numerical stability)
    float local_max = -INFINITY;
    for (int i = tid; i < cols; i += threads) {
        local_max = max(local_max, row_ptr[i]);
    }
    shared_max[tid] = local_max;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce to find global max
    for (uint stride = threads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_max[tid] = max(shared_max[tid], shared_max[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float max_val = shared_max[0];

    // Compute exp and sum
    float local_sum = 0.0f;
    for (int i = tid; i < cols; i += threads) {
        float e = exp(row_ptr[i] - max_val);
        row_ptr[i] = e;  // Store exp temporarily
        local_sum += e;
    }
    shared_sum[tid] = local_sum;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce to find total sum
    for (uint stride = threads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float sum = shared_sum[0];

    // Normalize
    float inv_sum = 1.0f / sum;
    for (int i = tid; i < cols; i += threads) {
        row_ptr[i] *= inv_sum;
    }
}

/* ========================================================================
 * RoPE (Rotary Position Embeddings)
 * ======================================================================== */

/* Apply 2D RoPE to Q or K tensor
 * x: [seq, heads*head_dim]
 * cos, sin: [seq, head_dim]  (precomputed frequencies)
 */
kernel void apply_rope_2d(
    device float *x [[buffer(0)]],
    device const float *cos_freq [[buffer(1)]],
    device const float *sin_freq [[buffer(2)]],
    constant int &seq [[buffer(3)]],
    constant int &heads [[buffer(4)]],
    constant int &head_dim [[buffer(5)]],
    constant int &axis_dim [[buffer(6)]],  // 32 for FLUX
    uint2 pos [[thread_position_in_grid]]  // (seq_idx, head_idx)
) {
    uint seq_idx = pos.x;
    uint head_idx = pos.y;

    if (seq_idx >= uint(seq) || head_idx >= uint(heads)) return;

    uint hidden = heads * head_dim;
    device float *vec = x + seq_idx * hidden + head_idx * head_dim;
    device const float *cos_row = cos_freq + seq_idx * head_dim;
    device const float *sin_row = sin_freq + seq_idx * head_dim;

    // RoPE rotation for each axis (4 axes of 32 dims each = 128)
    int half_axis = axis_dim / 2;  // 16

    for (int axis = 0; axis < 4; axis++) {
        int axis_offset = axis * axis_dim;
        for (int d = 0; d < half_axis; d++) {
            int i0 = axis_offset + d;
            int i1 = axis_offset + half_axis + d;

            float c = cos_row[i0];
            float s = sin_row[i0];

            float x0 = vec[i0];
            float x1 = vec[i1];

            vec[i0] = x0 * c - x1 * s;
            vec[i1] = x0 * s + x1 * c;
        }
    }
}
