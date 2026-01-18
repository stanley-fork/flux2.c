/*
 * FLUX Diffusion Transformer Implementation
 *
 * Flux2Transformer2DModel - Rectified Flow Transformer for image generation.
 *
 * Architecture (klein 4B):
 * - 5 double-stream blocks (MM-DiT: separate image/text processing, joint attention)
 * - 20 single-stream blocks (parallel DiT: fused QKV+FFN)
 * - 24 attention heads, 128 dim per head (3072 hidden)
 * - No bias parameters
 * - SwiGLU activation
 * - RoPE positional embeddings
 * - Shared AdaLN-Zero modulation
 */

#include "flux.h"
#include "flux_kernels.h"
#include "flux_safetensors.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ========================================================================
 * Transformer Data Structures
 * ======================================================================== */

/* AdaLN-Zero modulation parameters (shared across blocks) */
typedef struct {
    float *mod_weight;      /* [hidden * 6] for double, [hidden * 3] for single */
} adaln_t;

/* Double-stream block (MM-DiT style) */
typedef struct {
    /* Image stream - separate Q, K, V weights */
    float *img_q_weight;            /* [hidden, hidden] */
    float *img_k_weight;            /* [hidden, hidden] */
    float *img_v_weight;            /* [hidden, hidden] */
    float *img_norm_q_weight;       /* [head_dim] - QK norm on Q */
    float *img_norm_k_weight;       /* [head_dim] - QK norm on K */
    float *img_proj_weight;         /* [hidden, hidden] */
    float *img_mlp_gate_weight;     /* [mlp_hidden, hidden] */
    float *img_mlp_up_weight;       /* [mlp_hidden, hidden] */
    float *img_mlp_down_weight;     /* [hidden, mlp_hidden] */

    /* Text stream - separate Q, K, V weights */
    float *txt_q_weight;            /* [hidden, hidden] */
    float *txt_k_weight;            /* [hidden, hidden] */
    float *txt_v_weight;            /* [hidden, hidden] */
    float *txt_norm_q_weight;       /* [head_dim] - QK norm on Q */
    float *txt_norm_k_weight;       /* [head_dim] - QK norm on K */
    float *txt_proj_weight;         /* [hidden, hidden] */
    float *txt_mlp_gate_weight;     /* [mlp_hidden, hidden] */
    float *txt_mlp_up_weight;       /* [mlp_hidden, hidden] */
    float *txt_mlp_down_weight;     /* [hidden, mlp_hidden] */
} double_block_t;

/* Single-stream block (Parallel DiT style, fused) */
typedef struct {
    /* Fused QKV + FFN input projection */
    float *qkv_mlp_weight;          /* [hidden*3 + mlp_hidden*2, hidden] */
    /* QK normalization */
    float *norm_q_weight;           /* [head_dim] */
    float *norm_k_weight;           /* [head_dim] */
    /* Fused attention out + FFN down projection */
    float *proj_mlp_weight;         /* [hidden, hidden + mlp_hidden] */
} single_block_t;

/* Timestep embedding MLP
 * FLUX.2-klein uses 256-dim sinusoidal embedding (128 frequencies)
 * linear_1: [hidden, 256] - projects sinusoidal to hidden
 * linear_2: [hidden, hidden] - another linear layer
 */
typedef struct {
    float *fc1_weight;              /* [hidden, 256] */
    float *fc2_weight;              /* [hidden, hidden] */
    int sincos_dim;                 /* 256 for FLUX.2-klein */
} time_embed_t;

/* Full transformer context */
typedef struct flux_transformer {
    /* Configuration */
    int hidden_size;        /* 3072 */
    int num_heads;          /* 24 */
    int head_dim;           /* 128 */
    int mlp_hidden;         /* hidden * 3 = 9216 */
    int num_double_layers;  /* 5 */
    int num_single_layers;  /* 20 */
    int text_dim;           /* 7680 */
    int latent_channels;    /* 128 */
    float rope_theta;       /* 2000 */
    int rope_dim;           /* 128 */

    /* Input projections */
    float *img_in_weight;   /* [hidden, latent_channels] */
    float *txt_in_weight;   /* [hidden, text_dim] */

    /* Timestep embedding */
    time_embed_t time_embed;
    float *time_freq;       /* [hidden/2] - precomputed sinusoidal frequencies */

    /* Shared AdaLN modulation */
    float *adaln_double_img_weight;  /* [hidden * 6, hidden] for double block img stream */
    float *adaln_double_txt_weight;  /* [hidden * 6, hidden] for double block txt stream */
    float *adaln_single_weight;      /* [hidden * 3, hidden] for single block */

    /* Transformer blocks */
    double_block_t *double_blocks;
    single_block_t *single_blocks;

    /* Final layer */
    float *final_norm_weight;       /* [hidden] */
    float *final_proj_weight;       /* [latent_channels, hidden] */

    /* RoPE frequencies (precomputed) */
    float *rope_freqs;              /* [max_seq, head_dim/2, 2] - legacy 1D */
    float *rope_cos;                /* [max_seq, axis_dim] - 2D cos frequencies */
    float *rope_sin;                /* [max_seq, axis_dim] - 2D sin frequencies */
    int max_seq_len;
    int axis_dim;                   /* 32 for FLUX (128 head_dim / 4 axes) */

    /* Working memory */
    float *img_hidden;              /* [max_img_seq, hidden] */
    float *txt_hidden;              /* [max_txt_seq, hidden] */
    float *q, *k, *v;               /* [max_seq, hidden] */
    float *attn_out;                /* [max_seq, hidden] */
    float *mlp_buffer;              /* [max_seq, mlp_hidden] */
    float *work1, *work2;
    size_t work_size;

    /* Pre-allocated attention workspaces to avoid malloc in hot path */
    float *attn_q_t;                /* [max_seq, hidden] transposed Q */
    float *attn_k_t;                /* [max_seq, hidden] transposed K */
    float *attn_v_t;                /* [max_seq, hidden] transposed V */
    float *attn_out_t;              /* [max_seq, hidden] transposed output */
    float *attn_scores;             /* [max_seq, max_seq] attention scores */
    float *attn_cat_k;              /* [max_seq, hidden] concatenated K */
    float *attn_cat_v;              /* [max_seq, hidden] concatenated V */
} flux_transformer_t;

/* Forward declarations */
void flux_transformer_free(flux_transformer_t *tf);

/* ========================================================================
 * RoPE (Rotary Position Embeddings)
 * ======================================================================== */

/* Precompute RoPE frequencies for given positions (1D version) */
static void compute_rope_freqs(float *freqs, int max_seq, int dim, float theta) {
    int half_dim = dim / 2;

    for (int pos = 0; pos < max_seq; pos++) {
        for (int d = 0; d < half_dim; d++) {
            float freq = 1.0f / powf(theta, (float)(2 * d) / (float)dim);
            float angle = (float)pos * freq;
            freqs[pos * half_dim * 2 + d * 2] = cosf(angle);
            freqs[pos * half_dim * 2 + d * 2 + 1] = sinf(angle);
        }
    }
}

/* Compute 2D RoPE frequencies for image tokens (h, w positions)
 * FLUX uses axes_dims_rope: [32, 32, 32, 32] = 128 total
 * Position IDs format: (T, H, W, L) where:
 * - Axis 0 (dims 0-31): T position (always 0 for images)
 * - Axis 1 (dims 32-63): H position (y/height coordinate)
 * - Axis 2 (dims 64-95): W position (x/width coordinate)
 * - Axis 3 (dims 96-127): L position (always 0 for images)
 */
static void compute_rope_2d(float *cos_out, float *sin_out,
                            int patch_h, int patch_w, int axis_dim, float theta) {
    int half_axis = axis_dim / 2;  /* 16 dims per half-axis */
    int seq = patch_h * patch_w;
    (void)seq;  /* Unused but kept for documentation */

    /* Precompute base frequencies: omega = 1 / (theta^(2d/dim)) for d = 0..15 */
    float *base_freqs = (float *)malloc(half_axis * sizeof(float));
    for (int d = 0; d < half_axis; d++) {
        base_freqs[d] = 1.0f / powf(theta, (float)(2 * d) / (float)axis_dim);
    }

    for (int hy = 0; hy < patch_h; hy++) {
        for (int wx = 0; wx < patch_w; wx++) {
            int pos = hy * patch_w + wx;
            float *cos_p = cos_out + pos * axis_dim * 4;  /* 4 axes * 32 dims each = 128 */
            float *sin_p = sin_out + pos * axis_dim * 4;

            /* Axis 0 (dims 0-31): T position = 0, so cos=1, sin=0 */
            for (int d = 0; d < axis_dim; d++) {
                cos_p[d] = 1.0f;
                sin_p[d] = 0.0f;
            }

            /* Axis 1 (dims 32-63): H position (y/height)
             * Python RoPE stacks [cos, -sin, sin, cos] as 2x2 matrix per freq.
             * For apply_rope: out = [[cos, -sin], [sin, cos]] @ [x0, x1]
             * We store cos/sin per pair and apply_rope_2d handles the rotation.
             */
            for (int d = 0; d < half_axis; d++) {
                float angle_h = (float)hy * base_freqs[d];
                float cos_h = cosf(angle_h);
                float sin_h = sinf(angle_h);
                /* Each frequency contributes to a pair of dimensions */
                cos_p[axis_dim + d * 2] = cos_h;
                cos_p[axis_dim + d * 2 + 1] = cos_h;
                sin_p[axis_dim + d * 2] = sin_h;
                sin_p[axis_dim + d * 2 + 1] = sin_h;
            }

            /* Axis 2 (dims 64-95): W position (x/width) */
            for (int d = 0; d < half_axis; d++) {
                float angle_w = (float)wx * base_freqs[d];
                float cos_w = cosf(angle_w);
                float sin_w = sinf(angle_w);
                cos_p[axis_dim * 2 + d * 2] = cos_w;
                cos_p[axis_dim * 2 + d * 2 + 1] = cos_w;
                sin_p[axis_dim * 2 + d * 2] = sin_w;
                sin_p[axis_dim * 2 + d * 2 + 1] = sin_w;
            }

            /* Axis 3 (dims 96-127): L position = 0, so cos=1, sin=0 */
            for (int d = 0; d < axis_dim; d++) {
                cos_p[axis_dim * 3 + d] = 1.0f;
                sin_p[axis_dim * 3 + d] = 0.0f;
            }
        }
    }
    free(base_freqs);
}

/* Apply 2D RoPE to image Q/K: x shape [seq, heads * head_dim]
 * Matches diffusers apply_rotary_emb with use_real=True, use_real_unbind_dim=-1
 * For each pair (i, i+1): out[i] = x[i]*cos - x[i+1]*sin, out[i+1] = x[i+1]*cos + x[i]*sin
 * cos/sin have 128 dims per position (4 axes * 32 dims)
 */
static void apply_rope_2d(float *x, const float *cos_freq, const float *sin_freq,
                          int seq, int heads, int head_dim, int axis_dim) {
    (void)axis_dim;  /* head_dim = 128 = 4 * axis_dim (axis_dim = 32) */
    for (int s = 0; s < seq; s++) {
        const float *cos_s = cos_freq + s * head_dim;  /* [128] */
        const float *sin_s = sin_freq + s * head_dim;

        for (int h = 0; h < heads; h++) {
            float *vec = x + (s * heads + h) * head_dim;

            /* Apply rotation to all 128 dims in pairs (0,1), (2,3), ... (126,127) */
            for (int d = 0; d < head_dim; d += 2) {
                float cos_val = cos_s[d];  /* cos[d] == cos[d+1] due to repeat_interleave */
                float sin_val = sin_s[d];
                float x0 = vec[d];
                float x1 = vec[d + 1];
                /* Complex rotation: (x0 + i*x1) * (cos + i*sin) */
                vec[d] = x0 * cos_val - x1 * sin_val;
                vec[d + 1] = x1 * cos_val + x0 * sin_val;
            }
        }
    }
}

/* Compute text RoPE frequencies for axis 3 (L dimension)
 * Text tokens have position IDs (T=0, H=0, W=0, L=seq_idx) where L = 0..seq-1
 * So axes 0-2 are identity, and axis 3 has the sequence position
 */
static void compute_rope_text(float *cos_out, float *sin_out,
                              int txt_seq, int axis_dim, float theta) {
    int half_axis = axis_dim / 2;  /* 16 */
    int head_dim = axis_dim * 4;   /* 128 */

    /* Precompute base frequencies */
    float *base_freqs = (float *)malloc(half_axis * sizeof(float));
    for (int d = 0; d < half_axis; d++) {
        base_freqs[d] = 1.0f / powf(theta, (float)(2 * d) / (float)axis_dim);
    }

    for (int s = 0; s < txt_seq; s++) {
        float *cos_p = cos_out + s * head_dim;  /* 128 dims */
        float *sin_p = sin_out + s * head_dim;

        /* Axes 0, 1, 2 (dims 0-95): T=H=W=0, so identity */
        for (int d = 0; d < axis_dim * 3; d++) {
            cos_p[d] = 1.0f;
            sin_p[d] = 0.0f;
        }

        /* Axis 3 (dims 96-127): L position = s (sequence index) */
        for (int d = 0; d < half_axis; d++) {
            float angle = (float)s * base_freqs[d];
            float cos_l = cosf(angle);
            float sin_l = sinf(angle);
            cos_p[axis_dim * 3 + d * 2] = cos_l;
            cos_p[axis_dim * 3 + d * 2 + 1] = cos_l;
            sin_p[axis_dim * 3 + d * 2] = sin_l;
            sin_p[axis_dim * 3 + d * 2 + 1] = sin_l;
        }
    }
    free(base_freqs);
}

/* ========================================================================
 * Timestep Embedding
 * ======================================================================== */

/* Sinusoidal timestep embedding
 * Matches diffusers get_timestep_embedding with:
 *   flip_sin_to_cos=True, downscale_freq_shift=0
 * Output format: [cos(all freqs), sin(all freqs)]
 */
static void get_timestep_embedding(float *out, float t, int dim, float max_period) {
    int half = dim / 2;
    float log_max = logf(max_period);

    for (int i = 0; i < half; i++) {
        /* freq = exp(-log(max_period) * i / half_dim) */
        float freq = expf(-log_max * (float)i / (float)half);
        float angle = t * freq;
        out[i] = cosf(angle);           /* cos part first (flip_sin_to_cos=True) */
        out[i + half] = sinf(angle);    /* sin part second */
    }
}

/* Forward through time embedding MLP */
static void time_embed_forward(float *out, const float *t_sincos,
                               const time_embed_t *te, int hidden) {
    /* MLP: fc1 (256->hidden) -> SiLU -> fc2 (hidden->hidden) */
    int sincos_dim = te->sincos_dim;

    float *h = (float *)malloc(hidden * sizeof(float));

    /* fc1: [sincos_dim] -> [hidden] */
    flux_linear_nobias(h, t_sincos, te->fc1_weight, 1, sincos_dim, hidden);

    /* SiLU */
    flux_silu(h, hidden);

    /* fc2: [hidden] -> [hidden] */
    flux_linear_nobias(out, h, te->fc2_weight, 1, hidden, hidden);

    free(h);
}

/* ========================================================================
 * AdaLN-Zero Modulation
 * ======================================================================== */

/* Apply AdaLN: out = (1 + scale) * LayerNorm(x) + shift
 * This is the standard DiT/FLUX formulation where scale is centered at 0
 * FLUX2 uses LayerNorm (not RMSNorm) with elementwise_affine=False before modulation
 */
static void apply_adaln(float *out, const float *x,
                        const float *shift, const float *scale,
                        int seq, int hidden, float eps) {
    /* Layer Norm (subtract mean, divide by std) + AdaLN modulation
     * Note: Flux2 uses LayerNorm with elementwise_affine=False (no learned weights)
     */
    for (int s = 0; s < seq; s++) {
        const float *x_row = x + s * hidden;
        float *out_row = out + s * hidden;

        /* Compute mean */
        float sum = 0.0f;
        for (int i = 0; i < hidden; i++) {
            sum += x_row[i];
        }
        float mean = sum / hidden;

        /* Compute variance */
        float var_sum = 0.0f;
        for (int i = 0; i < hidden; i++) {
            float diff = x_row[i] - mean;
            var_sum += diff * diff;
        }
        float var = var_sum / hidden;
        float std_inv = 1.0f / sqrtf(var + eps);

        /* Apply Layer Norm + AdaLN modulation */
        for (int i = 0; i < hidden; i++) {
            float norm = (x_row[i] - mean) * std_inv;
            out_row[i] = (1.0f + scale[i]) * norm + shift[i];
        }
    }
}

/* Apply QK normalization (RMSNorm per head) */
static void apply_qk_norm(float *q, float *k,
                          const float *q_weight, const float *k_weight,
                          int seq, int heads, int head_dim, float eps) {
    /* Apply RMSNorm to each head of Q and K */
    for (int s = 0; s < seq; s++) {
        for (int h = 0; h < heads; h++) {
            /* Q normalization */
            float *qh = q + s * heads * head_dim + h * head_dim;
            float sum_sq = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                sum_sq += qh[d] * qh[d];
            }
            float rms_inv = 1.0f / sqrtf(sum_sq / head_dim + eps);
            for (int d = 0; d < head_dim; d++) {
                qh[d] = qh[d] * rms_inv * q_weight[d];
            }

            /* K normalization */
            float *kh = k + s * heads * head_dim + h * head_dim;
            sum_sq = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                sum_sq += kh[d] * kh[d];
            }
            rms_inv = 1.0f / sqrtf(sum_sq / head_dim + eps);
            for (int d = 0; d < head_dim; d++) {
                kh[d] = kh[d] * rms_inv * k_weight[d];
            }
        }
    }
}

/* ========================================================================
 * Attention Layer
 * ======================================================================== */

/* Multi-head self-attention */
/* Transpose from [seq, heads, head_dim] to [heads, seq, head_dim] */
static void transpose_shd_to_hsd(float *out, const float *in,
                                  int seq, int heads, int head_dim) {
    for (int s = 0; s < seq; s++) {
        for (int h = 0; h < heads; h++) {
            const float *src = in + (s * heads + h) * head_dim;
            float *dst = out + (h * seq + s) * head_dim;
            memcpy(dst, src, head_dim * sizeof(float));
        }
    }
}

/* Transpose from [heads, seq, head_dim] to [seq, heads, head_dim] */
static void transpose_hsd_to_shd(float *out, const float *in,
                                  int seq, int heads, int head_dim) {
    for (int h = 0; h < heads; h++) {
        for (int s = 0; s < seq; s++) {
            const float *src = in + (h * seq + s) * head_dim;
            float *dst = out + (s * heads + h) * head_dim;
            memcpy(dst, src, head_dim * sizeof(float));
        }
    }
}

/* Multi-head attention with BLAS optimization
 * Uses pre-allocated workspace buffers from transformer struct
 */
static void mha_forward(float *out, const float *q, const float *k, const float *v,
                        int seq, int heads, int head_dim, flux_transformer_t *tf) {
    float scale = 1.0f / sqrtf((float)head_dim);
    (void)heads; /* hidden = heads * head_dim, but we use tf->hidden_size */

    /* Use pre-allocated buffers from transformer */
    float *q_t = tf->attn_q_t;
    float *k_t = tf->attn_k_t;
    float *v_t = tf->attn_v_t;
    float *out_t = tf->attn_out_t;
    float *scores = tf->attn_scores;

    /* Transpose to [heads, seq, head_dim] for efficient BLAS operations */
    transpose_shd_to_hsd(q_t, q, seq, tf->num_heads, head_dim);
    transpose_shd_to_hsd(k_t, k, seq, tf->num_heads, head_dim);
    transpose_shd_to_hsd(v_t, v, seq, tf->num_heads, head_dim);

    /* Process each head with BLAS */
    for (int h = 0; h < tf->num_heads; h++) {
        float *qh = q_t + h * seq * head_dim;
        float *kh = k_t + h * seq * head_dim;
        float *vh = v_t + h * seq * head_dim;
        float *oh = out_t + h * seq * head_dim;

        /* scores = Q @ K^T using BLAS */
        /* Q[seq, head_dim] @ K[seq, head_dim]^T = scores[seq, seq] */
#if defined(USE_ACCELERATE) || defined(USE_OPENBLAS)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    seq, seq, head_dim,
                    scale, qh, head_dim, kh, head_dim,
                    0.0f, scores, seq);
#else
        for (int i = 0; i < seq; i++) {
            for (int j = 0; j < seq; j++) {
                float dot = 0.0f;
                const float *qh_row = qh + i * head_dim;
                const float *kh_row = kh + j * head_dim;
                for (int d = 0; d < head_dim; d++) {
                    dot += qh_row[d] * kh_row[d];
                }
                scores[i * seq + j] = dot * scale;
            }
        }
#endif

        /* Softmax */
        flux_softmax(scores, seq, seq);

        /* out = scores @ V using BLAS */
        /* scores[seq, seq] @ V[seq, head_dim] = out[seq, head_dim] */
#if defined(USE_ACCELERATE) || defined(USE_OPENBLAS)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    seq, head_dim, seq,
                    1.0f, scores, seq, vh, head_dim,
                    0.0f, oh, head_dim);
#else
        for (int i = 0; i < seq; i++) {
            float *oh_row = oh + i * head_dim;
            const float *score_row = scores + i * seq;
            for (int d = 0; d < head_dim; d++) {
                oh_row[d] = 0.0f;
            }
            for (int j = 0; j < seq; j++) {
                float s_val = score_row[j];
                const float *v_row = vh + j * head_dim;
                for (int d = 0; d < head_dim; d++) {
                    oh_row[d] += s_val * v_row[d];
                }
            }
        }
#endif
    }

    /* Transpose output back to [seq, heads, head_dim] */
    transpose_hsd_to_shd(out, out_t, seq, tf->num_heads, head_dim);

    /* No free - using pre-allocated buffers */
}

/* Joint attention (for double blocks) - image and text attend to each other
 * Uses pre-allocated workspace buffers from transformer struct
 */
static void joint_attention(float *img_out, float *txt_out,
                            const float *img_q, const float *img_k, const float *img_v,
                            const float *txt_q, const float *txt_k, const float *txt_v,
                            int img_seq, int txt_seq, int heads, int head_dim,
                            flux_transformer_t *tf) {
    int total_seq = img_seq + txt_seq;
    int hidden = heads * head_dim;
    float scale = 1.0f / sqrtf((float)head_dim);

    /* Use pre-allocated buffers for K/V concatenation */
    float *cat_k = tf->attn_cat_k;
    float *cat_v = tf->attn_cat_v;

    /* Concatenate K, V from both streams in [seq, heads, head_dim] format
     * IMPORTANT: Python (official Flux2) concatenates as [TEXT, IMAGE]
     */
    memcpy(cat_k, txt_k, txt_seq * hidden * sizeof(float));
    memcpy(cat_v, txt_v, txt_seq * hidden * sizeof(float));
    memcpy(cat_k + txt_seq * hidden, img_k, img_seq * hidden * sizeof(float));
    memcpy(cat_v + txt_seq * hidden, img_v, img_seq * hidden * sizeof(float));

    /* Use pre-allocated buffers for transposed data
     * Layout: attn_q_t holds [img_q_t, txt_q_t], attn_k_t holds cat_k_t, etc.
     */
    float *img_q_t = tf->attn_q_t;
    float *txt_q_t = tf->attn_q_t + img_seq * hidden;  /* After img_q_t */
    float *cat_k_t = tf->attn_k_t;
    float *cat_v_t = tf->attn_v_t;
    float *img_out_t = tf->attn_out_t;
    float *txt_out_t = tf->attn_out_t + img_seq * hidden;  /* After img_out_t */

    /* Transpose to [heads, seq, head_dim] for BLAS */
    transpose_shd_to_hsd(img_q_t, img_q, img_seq, heads, head_dim);
    transpose_shd_to_hsd(txt_q_t, txt_q, txt_seq, heads, head_dim);
    transpose_shd_to_hsd(cat_k_t, cat_k, total_seq, heads, head_dim);
    transpose_shd_to_hsd(cat_v_t, cat_v, total_seq, heads, head_dim);

    /* Use attn_scores buffer, split for img and txt */
    float *img_scores = tf->attn_scores;
    float *txt_scores = tf->attn_scores + img_seq * total_seq;

    for (int h = 0; h < heads; h++) {
        float *qh_img = img_q_t + h * img_seq * head_dim;
        float *qh_txt = txt_q_t + h * txt_seq * head_dim;
        float *kh = cat_k_t + h * total_seq * head_dim;
        float *vh = cat_v_t + h * total_seq * head_dim;
        float *oh_img = img_out_t + h * img_seq * head_dim;
        float *oh_txt = txt_out_t + h * txt_seq * head_dim;

        /* Image Q @ K^T */
#if defined(USE_ACCELERATE) || defined(USE_OPENBLAS)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    img_seq, total_seq, head_dim,
                    scale, qh_img, head_dim, kh, head_dim,
                    0.0f, img_scores, total_seq);
#else
        for (int i = 0; i < img_seq; i++) {
            for (int j = 0; j < total_seq; j++) {
                float dot = 0.0f;
                const float *qh_row = qh_img + i * head_dim;
                const float *kh_row = kh + j * head_dim;
                for (int d = 0; d < head_dim; d++) {
                    dot += qh_row[d] * kh_row[d];
                }
                img_scores[i * total_seq + j] = dot * scale;
            }
        }
#endif

#ifdef DEBUG_ATTENTION
        if (h == 0) {
            fprintf(stderr, "[ATTN] Image query 0, head 0 scores:\n");
            fprintf(stderr, "  first 5 txt keys: ");
            for (int j = 0; j < 5; j++) fprintf(stderr, "%.6f ", img_scores[j]);
            fprintf(stderr, "\n  first 5 img keys (pos %d): ", txt_seq);
            for (int j = txt_seq; j < txt_seq + 5; j++) fprintf(stderr, "%.6f ", img_scores[j]);
            fprintf(stderr, "\n");
        }
#endif

        flux_softmax(img_scores, img_seq, total_seq);

        /* Image scores @ V */
#if defined(USE_ACCELERATE) || defined(USE_OPENBLAS)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    img_seq, head_dim, total_seq,
                    1.0f, img_scores, total_seq, vh, head_dim,
                    0.0f, oh_img, head_dim);
#else
        for (int i = 0; i < img_seq; i++) {
            float *oh_row = oh_img + i * head_dim;
            const float *score_row = img_scores + i * total_seq;
            for (int d = 0; d < head_dim; d++) {
                oh_row[d] = 0.0f;
            }
            for (int j = 0; j < total_seq; j++) {
                float s_val = score_row[j];
                const float *v_row = vh + j * head_dim;
                for (int d = 0; d < head_dim; d++) {
                    oh_row[d] += s_val * v_row[d];
                }
            }
        }
#endif

        /* Text Q @ K^T */
#if defined(USE_ACCELERATE) || defined(USE_OPENBLAS)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    txt_seq, total_seq, head_dim,
                    scale, qh_txt, head_dim, kh, head_dim,
                    0.0f, txt_scores, total_seq);
#else
        for (int i = 0; i < txt_seq; i++) {
            for (int j = 0; j < total_seq; j++) {
                float dot = 0.0f;
                const float *qh_row = qh_txt + i * head_dim;
                const float *kh_row = kh + j * head_dim;
                for (int d = 0; d < head_dim; d++) {
                    dot += qh_row[d] * kh_row[d];
                }
                txt_scores[i * total_seq + j] = dot * scale;
            }
        }
#endif

        flux_softmax(txt_scores, txt_seq, total_seq);

        /* Text scores @ V */
#if defined(USE_ACCELERATE) || defined(USE_OPENBLAS)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    txt_seq, head_dim, total_seq,
                    1.0f, txt_scores, total_seq, vh, head_dim,
                    0.0f, oh_txt, head_dim);
#else
        for (int i = 0; i < txt_seq; i++) {
            float *oh_row = oh_txt + i * head_dim;
            const float *score_row = txt_scores + i * total_seq;
            for (int d = 0; d < head_dim; d++) {
                oh_row[d] = 0.0f;
            }
            for (int j = 0; j < total_seq; j++) {
                float s_val = score_row[j];
                const float *v_row = vh + j * head_dim;
                for (int d = 0; d < head_dim; d++) {
                    oh_row[d] += s_val * v_row[d];
                }
            }
        }
#endif
    }

    /* Transpose outputs back */
    transpose_hsd_to_shd(img_out, img_out_t, img_seq, heads, head_dim);
    transpose_hsd_to_shd(txt_out, txt_out_t, txt_seq, heads, head_dim);

    /* No free - using pre-allocated buffers */
}

/* ========================================================================
 * SwiGLU FFN
 * ======================================================================== */

static void swiglu_ffn(float *out, const float *x,
                       const float *gate_weight, const float *up_weight,
                       const float *down_weight,
                       int seq, int hidden, int mlp_hidden) {
    float *gate = (float *)malloc(seq * mlp_hidden * sizeof(float));
    float *up = (float *)malloc(seq * mlp_hidden * sizeof(float));

    /* Gate and up projections */
    flux_linear_nobias(gate, x, gate_weight, seq, hidden, mlp_hidden);
    flux_linear_nobias(up, x, up_weight, seq, hidden, mlp_hidden);

    /* SiLU(gate) * up */
    flux_silu(gate, seq * mlp_hidden);
    flux_mul_inplace(gate, up, seq * mlp_hidden);

    /* Down projection */
    flux_linear_nobias(out, gate, down_weight, seq, mlp_hidden, hidden);

    free(gate);
    free(up);
}

/* ========================================================================
 * Double-Stream Block (MM-DiT)
 * ======================================================================== */

static void double_block_forward(float *img_hidden, float *txt_hidden,
                                 const double_block_t *block,
                                 const float *t_emb,
                                 const float *img_adaln_weight,
                                 const float *txt_adaln_weight,
                                 const float *img_rope_cos, const float *img_rope_sin,
                                 const float *txt_rope_cos, const float *txt_rope_sin,
                                 int img_seq, int txt_seq,
                                 flux_transformer_t *tf) {
    int hidden = tf->hidden_size;
    int heads = tf->num_heads;
    int head_dim = tf->head_dim;
    int mlp_hidden = tf->mlp_hidden;
    float eps = 1e-6f;

    /* Compute AdaLN parameters (6 per stream: shift1, scale1, gate1, shift2, scale2, gate2)
     * adaln_weight is [hidden*6, hidden], t_emb is [hidden]
     * Output: 6 * hidden parameters per stream
     * FLUX applies SiLU to t_emb before the modulation projection
     */
    int mod_size = hidden * 6;

    /* Apply SiLU to t_emb for modulation (FLUX architecture requirement) */
    float *t_emb_silu = (float *)malloc(hidden * sizeof(float));
    for (int i = 0; i < hidden; i++) {
        float x = t_emb[i];
        t_emb_silu[i] = x / (1.0f + expf(-x));  /* SiLU = x * sigmoid(x) */
    }

    /* Image stream modulation */
    float *img_mod = (float *)malloc(mod_size * sizeof(float));
    flux_linear_nobias(img_mod, t_emb_silu, img_adaln_weight, 1, hidden, mod_size);

    float *img_shift1 = img_mod;
    float *img_scale1 = img_mod + hidden;
    float *img_gate1 = img_mod + hidden * 2;
    float *img_shift2 = img_mod + hidden * 3;
    float *img_scale2 = img_mod + hidden * 4;
    float *img_gate2 = img_mod + hidden * 5;

    /* Text stream modulation */
    float *txt_mod = (float *)malloc(mod_size * sizeof(float));
    flux_linear_nobias(txt_mod, t_emb_silu, txt_adaln_weight, 1, hidden, mod_size);

    free(t_emb_silu);

    float *txt_shift1 = txt_mod;
    float *txt_scale1 = txt_mod + hidden;
    float *txt_gate1 = txt_mod + hidden * 2;
    float *txt_shift2 = txt_mod + hidden * 3;
    float *txt_scale2 = txt_mod + hidden * 4;
    float *txt_gate2 = txt_mod + hidden * 5;

    /* Image stream: AdaLN -> QKV -> QK-norm -> RoPE */
    float *img_norm = tf->work1;
    apply_adaln(img_norm, img_hidden, img_shift1, img_scale1, img_seq, hidden, eps);

#ifdef DEBUG_DOUBLE_BLOCK
    static int block_idx = 0;
    if (block_idx == 0) {
        fprintf(stderr, "\n[DBL] img_mod[0,:10] (shift1): ");
        for (int i = 0; i < 10; i++) fprintf(stderr, "%.6f ", img_shift1[i]);
        fprintf(stderr, "\n[DBL] img_mod[0,3072:3082] (scale1): ");
        for (int i = 0; i < 10; i++) fprintf(stderr, "%.6f ", img_scale1[i]);
        fprintf(stderr, "\n[DBL] After AdaLN img_norm[0,0,:10]: ");
        for (int i = 0; i < 10; i++) fprintf(stderr, "%.6f ", img_norm[i]);
        fprintf(stderr, "\n");
    }
#endif

    /* Separate Q, K, V projections (fixes interleaved output bug) */
    float *img_q = tf->work2;
    float *img_k = img_q + img_seq * hidden;
    float *img_v = img_k + img_seq * hidden;

    flux_linear_nobias(img_q, img_norm, block->img_q_weight, img_seq, hidden, hidden);
    flux_linear_nobias(img_k, img_norm, block->img_k_weight, img_seq, hidden, hidden);
    flux_linear_nobias(img_v, img_norm, block->img_v_weight, img_seq, hidden, hidden);

    /* Apply QK normalization (per-head RMSNorm) */
    apply_qk_norm(img_q, img_k, block->img_norm_q_weight, block->img_norm_k_weight,
                  img_seq, heads, head_dim, eps);

    /* Apply 2D RoPE to image Q, K (using h, w positions) */
    int axis_dim = 32;
    apply_rope_2d(img_q, img_rope_cos, img_rope_sin, img_seq, heads, head_dim, axis_dim);
    apply_rope_2d(img_k, img_rope_cos, img_rope_sin, img_seq, heads, head_dim, axis_dim);

#ifdef DEBUG_DOUBLE_BLOCK
    if (block_idx == 0) {
        fprintf(stderr, "[DBL] After Q proj img_q[0,0,:5]: ");
        for (int i = 0; i < 5; i++) fprintf(stderr, "%.6f ", img_q[i]);
        fprintf(stderr, "\n[DBL] After RoPE img_q[0,0,:5]: ");
        for (int i = 0; i < 5; i++) fprintf(stderr, "%.6f ", img_q[i]);
        fprintf(stderr, "\n");
    }
#endif

    /* Text stream: AdaLN -> QKV -> QK-norm -> RoPE */
    float *txt_norm = img_norm + img_seq * hidden;
    apply_adaln(txt_norm, txt_hidden, txt_shift1, txt_scale1, txt_seq, hidden, eps);

    /* Separate Q, K, V projections for text */
    float *txt_q = img_v + img_seq * hidden;  /* After img_v */
    float *txt_k = txt_q + txt_seq * hidden;
    float *txt_v = txt_k + txt_seq * hidden;

    flux_linear_nobias(txt_q, txt_norm, block->txt_q_weight, txt_seq, hidden, hidden);
    flux_linear_nobias(txt_k, txt_norm, block->txt_k_weight, txt_seq, hidden, hidden);
    flux_linear_nobias(txt_v, txt_norm, block->txt_v_weight, txt_seq, hidden, hidden);

    /* Apply QK normalization */
    apply_qk_norm(txt_q, txt_k, block->txt_norm_q_weight, block->txt_norm_k_weight,
                  txt_seq, heads, head_dim, eps);

    /* Apply text RoPE - text tokens have position IDs (0, 0, 0, L) where L is sequence index
     * This applies rotation in axis 3 (dims 96-127)
     */
    apply_rope_2d(txt_q, txt_rope_cos, txt_rope_sin, txt_seq, heads, head_dim, axis_dim);
    apply_rope_2d(txt_k, txt_rope_cos, txt_rope_sin, txt_seq, heads, head_dim, axis_dim);

    /* Joint attention */
    float *img_attn_out = (float *)malloc(img_seq * hidden * sizeof(float));
    float *txt_attn_out = (float *)malloc(txt_seq * hidden * sizeof(float));

    joint_attention(img_attn_out, txt_attn_out,
                    img_q, img_k, img_v,
                    txt_q, txt_k, txt_v,
                    img_seq, txt_seq, heads, head_dim, tf);

#ifdef DEBUG_DOUBLE_BLOCK
    if (block_idx == 0) {
        fprintf(stderr, "[DBL] After attn img_attn_out[0,0,:5]: ");
        for (int i = 0; i < 5; i++) fprintf(stderr, "%.6f ", img_attn_out[i]);
        fprintf(stderr, "\n");
    }
#endif

    /* Project attention output */
    float *img_proj = tf->work1;
    float *txt_proj = img_proj + img_seq * hidden;

    flux_linear_nobias(img_proj, img_attn_out, block->img_proj_weight,
                       img_seq, hidden, hidden);
    flux_linear_nobias(txt_proj, txt_attn_out, block->txt_proj_weight,
                       txt_seq, hidden, hidden);

#ifdef DEBUG_DOUBLE_BLOCK
    if (block_idx == 0) {
        fprintf(stderr, "[DBL] After proj img_proj[0,0,:5]: ");
        for (int i = 0; i < 5; i++) fprintf(stderr, "%.6f ", img_proj[i]);
        fprintf(stderr, "\n");
        fprintf(stderr, "[DBL] gate1[0:5]: ");
        for (int i = 0; i < 5; i++) fprintf(stderr, "%.6f ", img_gate1[i]);
        fprintf(stderr, "\n");
    }
#endif

    /* Apply gate and add residual */
    for (int i = 0; i < img_seq * hidden; i++) {
        img_hidden[i] += img_gate1[i % hidden] * img_proj[i];
    }
    for (int i = 0; i < txt_seq * hidden; i++) {
        txt_hidden[i] += txt_gate1[i % hidden] * txt_proj[i];
    }

#ifdef DEBUG_DOUBLE_BLOCK
    if (block_idx == 0) {
        fprintf(stderr, "[DBL] After attn residual img_hidden[0,0,:5]: ");
        for (int i = 0; i < 5; i++) fprintf(stderr, "%.6f ", img_hidden[i]);
        fprintf(stderr, "\n");
    }
#endif

    /* FFN for image */
    apply_adaln(img_norm, img_hidden, img_shift2, img_scale2, img_seq, hidden, eps);

#ifdef DEBUG_DOUBLE_BLOCK
    if (block_idx == 0) {
        fprintf(stderr, "[DBL] FFN input (after AdaLN) img_norm[0,0,:5]: ");
        for (int i = 0; i < 5; i++) fprintf(stderr, "%.6f ", img_norm[i]);
        fprintf(stderr, "\n");
    }
#endif

    swiglu_ffn(img_proj, img_norm,
               block->img_mlp_gate_weight, block->img_mlp_up_weight,
               block->img_mlp_down_weight,
               img_seq, hidden, mlp_hidden);

#ifdef DEBUG_DOUBLE_BLOCK
    if (block_idx == 0) {
        fprintf(stderr, "[DBL] FFN output img_proj[0,0,:5]: ");
        for (int i = 0; i < 5; i++) fprintf(stderr, "%.6f ", img_proj[i]);
        fprintf(stderr, "\n");
        fprintf(stderr, "[DBL] gate2[0:5]: ");
        for (int i = 0; i < 5; i++) fprintf(stderr, "%.6f ", img_gate2[i]);
        fprintf(stderr, "\n");
    }
#endif

    for (int i = 0; i < img_seq * hidden; i++) {
        img_hidden[i] += img_gate2[i % hidden] * img_proj[i];
    }

#ifdef DEBUG_DOUBLE_BLOCK
    fprintf(stderr, "[DBL%d] After FFN residual img_hidden[0,0,:5]: ", block_idx);
    for (int i = 0; i < 5; i++) fprintf(stderr, "%.6f ", img_hidden[i]);
    fprintf(stderr, "\n");
#endif

    /* FFN for text */
    apply_adaln(txt_norm, txt_hidden, txt_shift2, txt_scale2, txt_seq, hidden, eps);
    swiglu_ffn(txt_proj, txt_norm,
               block->txt_mlp_gate_weight, block->txt_mlp_up_weight,
               block->txt_mlp_down_weight,
               txt_seq, hidden, mlp_hidden);
    for (int i = 0; i < txt_seq * hidden; i++) {
        txt_hidden[i] += txt_gate2[i % hidden] * txt_proj[i];
    }

    free(img_mod);
    free(txt_mod);
    free(img_attn_out);
    free(txt_attn_out);

#ifdef DEBUG_DOUBLE_BLOCK
    block_idx++;
#endif
}

/* ========================================================================
 * Single-Stream Block (Parallel DiT)
 * ======================================================================== */

static void single_block_forward(float *hidden, const single_block_t *block,
                                 const float *t_emb, const float *adaln_weight,
                                 const float *img_rope_cos, const float *img_rope_sin,
                                 const float *txt_rope_cos, const float *txt_rope_sin,
                                 int seq, int img_offset, flux_transformer_t *tf) {
    /* seq = total_seq (txt + img)
     * img_offset = txt_seq (where image starts in the [txt, img] concatenation)
     */
    int h_size = tf->hidden_size;
    int heads = tf->num_heads;
    int head_dim = tf->head_dim;
    int mlp_hidden = tf->mlp_hidden;
    int img_seq = seq - img_offset;  /* Number of image tokens */
    float eps = 1e-6f;

    /* Compute AdaLN parameters (3: shift, scale, gate)
     * adaln_weight is [hidden*3, hidden], t_emb is [hidden]
     * FLUX applies SiLU to t_emb before the modulation projection
     */
    int mod_size = h_size * 3;

    /* Apply SiLU to t_emb for modulation */
    float *t_emb_silu = (float *)malloc(h_size * sizeof(float));
    for (int i = 0; i < h_size; i++) {
        float x = t_emb[i];
        t_emb_silu[i] = x / (1.0f + expf(-x));
    }

    float *mod_params = (float *)malloc(mod_size * sizeof(float));
    flux_linear_nobias(mod_params, t_emb_silu, adaln_weight, 1, h_size, mod_size);
    free(t_emb_silu);

    float *shift = mod_params;
    float *scale = mod_params + h_size;
    float *gate = mod_params + h_size * 2;

    /* Norm */
    float *norm = tf->work1;
    apply_adaln(norm, hidden, shift, scale, seq, h_size, eps);

    /* Fused QKV + FFN input projection
     * Output: [seq, fused_dim] where fused_dim = [Q, K, V, gate, up]
     * Layout per position: [3072 Q, 3072 K, 3072 V, 9216 gate, 9216 up] = 27648 total
     */
    int fused_dim = h_size * 3 + mlp_hidden * 2;
    float *fused_out = tf->work2;
    flux_linear_nobias(fused_out, norm, block->qkv_mlp_weight, seq, h_size, fused_dim);

    /* Split outputs: need to de-interleave from [seq, fused_dim] format
     * Each position has [Q, K, V, gate, up] concatenated
     */
    float *q = (float *)malloc(seq * h_size * sizeof(float));
    float *k = (float *)malloc(seq * h_size * sizeof(float));
    float *v = (float *)malloc(seq * h_size * sizeof(float));
    float *mlp_gate_split = (float *)malloc(seq * mlp_hidden * sizeof(float));
    float *mlp_up_split = (float *)malloc(seq * mlp_hidden * sizeof(float));

    for (int s = 0; s < seq; s++) {
        float *row = fused_out + s * fused_dim;
        memcpy(q + s * h_size, row, h_size * sizeof(float));
        memcpy(k + s * h_size, row + h_size, h_size * sizeof(float));
        memcpy(v + s * h_size, row + h_size * 2, h_size * sizeof(float));
        memcpy(mlp_gate_split + s * mlp_hidden, row + h_size * 3, mlp_hidden * sizeof(float));
        memcpy(mlp_up_split + s * mlp_hidden, row + h_size * 3 + mlp_hidden, mlp_hidden * sizeof(float));
    }
    float *mlp_gate = mlp_gate_split;
    float *mlp_up = mlp_up_split;

    /* Apply QK normalization */
    apply_qk_norm(q, k, block->norm_q_weight, block->norm_k_weight,
                  seq, heads, head_dim, eps);

    /* Apply RoPE: layout is [txt, img]
     * - Text portion (0 to img_offset-1): RoPE in axis 3 (L dimension)
     * - Image portion (img_offset to seq-1): 2D RoPE based on H/W positions
     */
    int axis_dim = 32;
    int txt_seq = img_offset;

    /* Text portion: apply RoPE in axis 3 (L dimension = sequence position) */
    apply_rope_2d(q, txt_rope_cos, txt_rope_sin, txt_seq, heads, head_dim, axis_dim);
    apply_rope_2d(k, txt_rope_cos, txt_rope_sin, txt_seq, heads, head_dim, axis_dim);

    /* Image portion: apply 2D RoPE starting at img_offset */
    float *img_q = q + img_offset * h_size;
    float *img_k = k + img_offset * h_size;
    apply_rope_2d(img_q, img_rope_cos, img_rope_sin, img_seq, heads, head_dim, axis_dim);
    apply_rope_2d(img_k, img_rope_cos, img_rope_sin, img_seq, heads, head_dim, axis_dim);

    /* Self-attention */
    float *attn_out = (float *)malloc(seq * h_size * sizeof(float));
    mha_forward(attn_out, q, k, v, seq, heads, head_dim, tf);

    /* SwiGLU: silu(gate) * up */
    flux_silu(mlp_gate, seq * mlp_hidden);
    flux_mul_inplace(mlp_gate, mlp_up, seq * mlp_hidden);

    /* Fused output projection: [attn_out, mlp_out] -> hidden
     * proj_mlp_weight: [hidden, hidden + mlp_hidden]
     */
    float *concat = (float *)malloc(seq * (h_size + mlp_hidden) * sizeof(float));
    for (int s = 0; s < seq; s++) {
        memcpy(concat + s * (h_size + mlp_hidden),
               attn_out + s * h_size, h_size * sizeof(float));
        memcpy(concat + s * (h_size + mlp_hidden) + h_size,
               mlp_gate + s * mlp_hidden, mlp_hidden * sizeof(float));
    }

    float *proj_out = tf->work1;
    flux_linear_nobias(proj_out, concat, block->proj_mlp_weight,
                       seq, h_size + mlp_hidden, h_size);

    /* Apply gate and add residual */
    for (int i = 0; i < seq * h_size; i++) {
        hidden[i] += gate[i % h_size] * proj_out[i];
    }

    free(mod_params);
    free(attn_out);
    free(concat);
    free(q);
    free(k);
    free(v);
    free(mlp_gate_split);
    free(mlp_up_split);
}

/* ========================================================================
 * Full Transformer Forward Pass
 * ======================================================================== */

float *flux_transformer_forward(flux_transformer_t *tf,
                                const float *img_latent, int img_h, int img_w,
                                const float *txt_emb, int txt_seq,
                                float timestep) {
    int hidden = tf->hidden_size;
    int img_seq = img_h * img_w;
    int head_dim = tf->head_dim;
    int axis_dim = 32;  /* FLUX uses axes_dims_rope: [32, 32, 32, 32] */

    /* Get timestep embedding
     * FLUX.2-klein uses 256-dim sinusoidal (128 frequencies), not hidden_size
     */
    int sincos_dim = tf->time_embed.sincos_dim;
    float *t_emb = (float *)malloc(hidden * sizeof(float));
    float *t_sincos = (float *)malloc(sincos_dim * sizeof(float));
    get_timestep_embedding(t_sincos, timestep * 1000.0f, sincos_dim, 10000.0f);
    time_embed_forward(t_emb, t_sincos, &tf->time_embed, hidden);
    free(t_sincos);

    /* Compute 2D RoPE frequencies for image tokens based on actual dimensions
     * img_h, img_w are the patch grid dimensions (e.g., 4x4 for 64x64 image)
     */
    /* Allocate RoPE: 4 axes * 32 dims = 128 dims per position (matches head_dim) */
    float *img_rope_cos = (float *)malloc(img_seq * axis_dim * 4 * sizeof(float));
    float *img_rope_sin = (float *)malloc(img_seq * axis_dim * 4 * sizeof(float));
    compute_rope_2d(img_rope_cos, img_rope_sin, img_h, img_w, axis_dim, tf->rope_theta);

    /* Compute text RoPE frequencies - text tokens have position IDs (0, 0, 0, L)
     * where L is the sequence index. RoPE is applied in axis 3 (dims 96-127)
     */
    float *txt_rope_cos = (float *)malloc(txt_seq * head_dim * sizeof(float));
    float *txt_rope_sin = (float *)malloc(txt_seq * head_dim * sizeof(float));
    compute_rope_text(txt_rope_cos, txt_rope_sin, txt_seq, axis_dim, tf->rope_theta);

    /* Transpose input from NCHW [channels, h, w] to NLC [seq, channels] format
     * Input: img_latent[c * img_seq + pos] for channel c at position pos
     * Output: transposed[pos * channels + c]
     */
    int channels = tf->latent_channels;
    float *img_transposed = (float *)malloc(img_seq * channels * sizeof(float));
    for (int pos = 0; pos < img_seq; pos++) {
        for (int c = 0; c < channels; c++) {
            img_transposed[pos * channels + c] = img_latent[c * img_seq + pos];
        }
    }

    /* Project image latent to hidden */
    float *img_hidden = tf->img_hidden;
    flux_linear_nobias(img_hidden, img_transposed, tf->img_in_weight,
                       img_seq, tf->latent_channels, hidden);
    free(img_transposed);

    /* Project text embeddings to hidden */
    float *txt_hidden = tf->txt_hidden;
    flux_linear_nobias(txt_hidden, txt_emb, tf->txt_in_weight,
                       txt_seq, tf->text_dim, hidden);

#ifdef DEBUG_TRANSFORMER
    /* Debug: print intermediate values for comparison with Python */
    fprintf(stderr, "\n[DEBUG] t_emb first 10: ");
    for (int i = 0; i < 10; i++) fprintf(stderr, "%.6f ", t_emb[i]);
    fprintf(stderr, "\n");

    fprintf(stderr, "[DEBUG] img_hidden[0,0,:5] (img_proj): ");
    for (int i = 0; i < 5; i++) fprintf(stderr, "%.6f ", img_hidden[i]);
    fprintf(stderr, "\n");

    fprintf(stderr, "[DEBUG] txt_hidden[0,0,:5] (txt_proj): ");
    for (int i = 0; i < 5; i++) fprintf(stderr, "%.6f ", txt_hidden[i]);
    fprintf(stderr, "\n");

    fprintf(stderr, "[DEBUG] img_rope_cos[0, :10]: ");
    for (int i = 0; i < 10; i++) fprintf(stderr, "%.6f ", img_rope_cos[i]);
    fprintf(stderr, "\n");

    fprintf(stderr, "[DEBUG] txt_rope_cos[10, 96:106]: ");
    for (int i = 96; i < 106; i++) fprintf(stderr, "%.6f ", txt_rope_cos[10 * head_dim + i]);
    fprintf(stderr, "\n");
#endif

    /* Double-stream blocks */
    for (int i = 0; i < tf->num_double_layers; i++) {
        double_block_forward(img_hidden, txt_hidden,
                             &tf->double_blocks[i],
                             t_emb,
                             tf->adaln_double_img_weight,
                             tf->adaln_double_txt_weight,
                             img_rope_cos, img_rope_sin,
                             txt_rope_cos, txt_rope_sin,
                             img_seq, txt_seq, tf);
        if (flux_substep_callback)
            flux_substep_callback(FLUX_SUBSTEP_DOUBLE_BLOCK, i, tf->num_double_layers);
#ifdef DEBUG_TRANSFORMER
        if (i == 0) {
            fprintf(stderr, "\n[DEBUG] After double block 0:\n");
            fprintf(stderr, "[DEBUG] img_hidden[0,0,:10]: ");
            for (int d = 0; d < 10; d++) fprintf(stderr, "%.6f ", img_hidden[d]);
            fprintf(stderr, "\n");
            float sum = 0, sum_sq = 0;
            for (int d = 0; d < img_seq * hidden; d++) {
                sum += img_hidden[d];
                sum_sq += img_hidden[d] * img_hidden[d];
            }
            float mean = sum / (img_seq * hidden);
            float std = sqrtf(sum_sq / (img_seq * hidden) - mean * mean);
            fprintf(stderr, "[DEBUG] img_hidden mean=%.6f, std=%.6f\n", mean, std);
        }
#endif
    }

    /* Concatenate text and image for single-stream blocks
     * Python uses [txt, img] order for concatenation
     */
    int total_seq = img_seq + txt_seq;
    float *concat_hidden = (float *)malloc(total_seq * hidden * sizeof(float));
    memcpy(concat_hidden, txt_hidden, txt_seq * hidden * sizeof(float));
    memcpy(concat_hidden + txt_seq * hidden, img_hidden,
           img_seq * hidden * sizeof(float));

    /* Single-stream blocks */
    for (int i = 0; i < tf->num_single_layers; i++) {
        single_block_forward(concat_hidden, &tf->single_blocks[i],
                             t_emb, tf->adaln_single_weight,
                             img_rope_cos, img_rope_sin,
                             txt_rope_cos, txt_rope_sin,
                             total_seq, txt_seq, tf);  /* txt_seq is the offset to image */
        if (flux_substep_callback)
            flux_substep_callback(FLUX_SUBSTEP_SINGLE_BLOCK, i, tf->num_single_layers);

#ifdef DEBUG_SINGLE_BLOCK
        if (i == 0 || i == 9 || i == 19) {
            /* Print image portion (starts at txt_seq offset) */
            float *img_part = concat_hidden + txt_seq * hidden;
            fprintf(stderr, "[SGL%d] img_hidden[0,0,:5]: ", i);
            for (int d = 0; d < 5; d++) fprintf(stderr, "%.6f ", img_part[d]);
            fprintf(stderr, "\n");
        }
#endif
    }

    /* Extract image hidden states (image is after text) */
    memcpy(img_hidden, concat_hidden + txt_seq * hidden, img_seq * hidden * sizeof(float));
    free(concat_hidden);

#ifdef DEBUG_FINAL_LAYER
    fprintf(stderr, "[FINAL] Before final layer img_hidden[0,0,:5]: ");
    for (int d = 0; d < 5; d++) fprintf(stderr, "%.6f ", img_hidden[d]);
    fprintf(stderr, "\n");
#endif

    /* Final layer: AdaLN modulation -> project to latent channels
     * norm_out.linear.weight is [6144, 3072] = [shift, scale] projection
     * Apply SiLU to t_emb before modulation projection (FLUX architecture)
     */
    float *t_emb_silu = (float *)malloc(hidden * sizeof(float));
    for (int i = 0; i < hidden; i++) {
        float x = t_emb[i];
        t_emb_silu[i] = x / (1.0f + expf(-x));
    }

    float *final_mod = (float *)malloc(hidden * 2 * sizeof(float));
    flux_linear_nobias(final_mod, t_emb_silu, tf->final_norm_weight, 1, hidden, hidden * 2);
    free(t_emb_silu);

    /* Python: scale, shift = mod.chunk(2, dim=1) - scale is first half, shift is second half */
    float *final_scale = final_mod;
    float *final_shift = final_mod + hidden;

    float *final_norm = tf->work1;
    apply_adaln(final_norm, img_hidden, final_shift, final_scale, img_seq, hidden, 1e-6f);
    free(final_mod);

    float *output_nlc = (float *)malloc(img_seq * tf->latent_channels * sizeof(float));
    flux_linear_nobias(output_nlc, final_norm, tf->final_proj_weight,
                       img_seq, hidden, tf->latent_channels);

    /* Transpose output from NLC [seq, channels] to NCHW [channels, h, w] format
     * Input: output_nlc[pos * channels + c]
     * Output: output[c * img_seq + pos]
     */
    float *output = (float *)malloc(img_seq * tf->latent_channels * sizeof(float));
    for (int pos = 0; pos < img_seq; pos++) {
        for (int c = 0; c < channels; c++) {
            output[c * img_seq + pos] = output_nlc[pos * channels + c];
        }
    }
    free(output_nlc);

    free(t_emb);
    free(img_rope_cos);
    free(img_rope_sin);
    free(txt_rope_cos);
    free(txt_rope_sin);

    if (flux_substep_callback)
        flux_substep_callback(FLUX_SUBSTEP_FINAL_LAYER, 0, 1);

    return output;
}

/* ========================================================================
 * Transformer Loading
 * ======================================================================== */

static float *read_floats(FILE *f, int count) {
    float *data = (float *)malloc(count * sizeof(float));
    if (!data) return NULL;
    if (fread(data, sizeof(float), count, f) != (size_t)count) {
        free(data);
        return NULL;
    }
    return data;
}

flux_transformer_t *flux_transformer_load(FILE *f) {
    flux_transformer_t *tf = calloc(1, sizeof(flux_transformer_t));
    if (!tf) return NULL;

    /* Read config */
    uint32_t config[10];
    if (fread(config, sizeof(uint32_t), 10, f) != 10) goto error;

    tf->hidden_size = config[0];
    tf->num_heads = config[1];
    tf->head_dim = config[2];
    tf->mlp_hidden = config[3];
    tf->num_double_layers = config[4];
    tf->num_single_layers = config[5];
    tf->text_dim = config[6];
    tf->latent_channels = config[7];
    tf->max_seq_len = config[8];
    tf->rope_dim = config[9];

    float rope_theta;
    if (fread(&rope_theta, sizeof(float), 1, f) != 1) goto error;
    tf->rope_theta = rope_theta;

    /* Read input projections */
    tf->img_in_weight = read_floats(f, tf->hidden_size * tf->latent_channels);
    tf->txt_in_weight = read_floats(f, tf->hidden_size * tf->text_dim);

    /* Read time embedding (binary format - deprecated, use safetensors) */
    tf->time_embed.sincos_dim = 256;  /* Match safetensors model */
    tf->time_embed.fc1_weight = read_floats(f, tf->hidden_size * 256);
    tf->time_embed.fc2_weight = read_floats(f, tf->hidden_size * tf->hidden_size);

    /* Read double blocks (binary format - deprecated, use safetensors) */
    tf->double_blocks = calloc(tf->num_double_layers, sizeof(double_block_t));
    for (int i = 0; i < tf->num_double_layers; i++) {
        double_block_t *b = &tf->double_blocks[i];
        int h = tf->hidden_size;
        int mlp = tf->mlp_hidden;
        int head_dim = tf->head_dim;

        /* QK norm weights (per head) */
        b->img_norm_q_weight = read_floats(f, head_dim);
        b->img_norm_k_weight = read_floats(f, head_dim);
        b->img_q_weight = read_floats(f, h * h);
        b->img_k_weight = read_floats(f, h * h);
        b->img_v_weight = read_floats(f, h * h);
        b->img_proj_weight = read_floats(f, h * h);
        b->img_mlp_gate_weight = read_floats(f, mlp * h);
        b->img_mlp_up_weight = read_floats(f, mlp * h);
        b->img_mlp_down_weight = read_floats(f, h * mlp);

        b->txt_norm_q_weight = read_floats(f, head_dim);
        b->txt_norm_k_weight = read_floats(f, head_dim);
        b->txt_q_weight = read_floats(f, h * h);
        b->txt_k_weight = read_floats(f, h * h);
        b->txt_v_weight = read_floats(f, h * h);
        b->txt_proj_weight = read_floats(f, h * h);
        b->txt_mlp_gate_weight = read_floats(f, mlp * h);
        b->txt_mlp_up_weight = read_floats(f, mlp * h);
        b->txt_mlp_down_weight = read_floats(f, h * mlp);
    }

    /* Read single blocks (binary format - deprecated, use safetensors) */
    tf->single_blocks = calloc(tf->num_single_layers, sizeof(single_block_t));
    for (int i = 0; i < tf->num_single_layers; i++) {
        single_block_t *b = &tf->single_blocks[i];
        int h = tf->hidden_size;
        int mlp = tf->mlp_hidden;
        int head_dim = tf->head_dim;

        b->norm_q_weight = read_floats(f, head_dim);
        b->norm_k_weight = read_floats(f, head_dim);
        b->qkv_mlp_weight = read_floats(f, (h * 3 + mlp * 2) * h);
        b->proj_mlp_weight = read_floats(f, h * (h + mlp));
    }

    /* Read final layer */
    tf->final_norm_weight = read_floats(f, tf->hidden_size);
    tf->final_proj_weight = read_floats(f, tf->latent_channels * tf->hidden_size);

    /* Precompute RoPE frequencies */
    tf->rope_freqs = (float *)malloc(tf->max_seq_len * tf->head_dim * sizeof(float));
    compute_rope_freqs(tf->rope_freqs, tf->max_seq_len, tf->head_dim, tf->rope_theta);

    /* Allocate working memory */
    int max_seq = tf->max_seq_len;
    int hidden = tf->hidden_size;
    tf->img_hidden = (float *)malloc(max_seq * hidden * sizeof(float));
    tf->txt_hidden = (float *)malloc(max_seq * hidden * sizeof(float));
    tf->work_size = max_seq * hidden * 4 * sizeof(float);
    tf->work1 = (float *)malloc(tf->work_size);
    tf->work2 = (float *)malloc(tf->work_size);

    /* Pre-allocated attention workspaces to avoid malloc in hot path */
    tf->attn_q_t = (float *)malloc(max_seq * hidden * sizeof(float));
    tf->attn_k_t = (float *)malloc(max_seq * hidden * sizeof(float));
    tf->attn_v_t = (float *)malloc(max_seq * hidden * sizeof(float));
    tf->attn_out_t = (float *)malloc(max_seq * hidden * sizeof(float));
    tf->attn_scores = (float *)malloc(max_seq * max_seq * sizeof(float));
    tf->attn_cat_k = (float *)malloc(max_seq * hidden * sizeof(float));
    tf->attn_cat_v = (float *)malloc(max_seq * hidden * sizeof(float));

    return tf;

error:
    flux_transformer_free(tf);
    return NULL;
}

void flux_transformer_free(flux_transformer_t *tf) {
    if (!tf) return;

    free(tf->img_in_weight);
    free(tf->txt_in_weight);
    free(tf->time_embed.fc1_weight);
    free(tf->time_embed.fc2_weight);

    if (tf->double_blocks) {
        for (int i = 0; i < tf->num_double_layers; i++) {
            double_block_t *b = &tf->double_blocks[i];
            free(b->img_norm_q_weight);
            free(b->img_norm_k_weight);
            free(b->img_q_weight);
            free(b->img_k_weight);
            free(b->img_v_weight);
            free(b->img_proj_weight);
            free(b->img_mlp_gate_weight);
            free(b->img_mlp_up_weight);
            free(b->img_mlp_down_weight);
            free(b->txt_norm_q_weight);
            free(b->txt_norm_k_weight);
            free(b->txt_q_weight);
            free(b->txt_k_weight);
            free(b->txt_v_weight);
            free(b->txt_proj_weight);
            free(b->txt_mlp_gate_weight);
            free(b->txt_mlp_up_weight);
            free(b->txt_mlp_down_weight);
        }
        free(tf->double_blocks);
    }

    if (tf->single_blocks) {
        for (int i = 0; i < tf->num_single_layers; i++) {
            single_block_t *b = &tf->single_blocks[i];
            free(b->norm_q_weight);
            free(b->norm_k_weight);
            free(b->qkv_mlp_weight);
            free(b->proj_mlp_weight);
        }
        free(tf->single_blocks);
    }

    free(tf->final_norm_weight);
    free(tf->final_proj_weight);
    free(tf->rope_freqs);
    free(tf->img_hidden);
    free(tf->txt_hidden);
    free(tf->work1);
    free(tf->work2);
    free(tf->adaln_double_img_weight);
    free(tf->adaln_double_txt_weight);
    free(tf->adaln_single_weight);

    /* Free attention workspace buffers */
    free(tf->attn_q_t);
    free(tf->attn_k_t);
    free(tf->attn_v_t);
    free(tf->attn_out_t);
    free(tf->attn_scores);
    free(tf->attn_cat_k);
    free(tf->attn_cat_v);

    free(tf);
}

/* ========================================================================
 * Safetensors Loading
 * ======================================================================== */

static float *get_sf_tensor_tf(safetensors_file_t *sf, const char *name) {
    const safetensor_t *t = safetensors_find(sf, name);
    if (!t) {
        fprintf(stderr, "Warning: tensor %s not found\n", name);
        return NULL;
    }
    return safetensors_get_f32(sf, t);
}

flux_transformer_t *flux_transformer_load_safetensors(safetensors_file_t *sf) {
    flux_transformer_t *tf = calloc(1, sizeof(flux_transformer_t));
    if (!tf) return NULL;

    char name[256];

    /* Set config based on FLUX.2-klein-4B */
    tf->hidden_size = 3072;
    tf->num_heads = 24;
    tf->head_dim = 128;
    tf->mlp_hidden = 9216;
    tf->num_double_layers = 5;
    tf->num_single_layers = 20;
    tf->text_dim = 7680;
    tf->latent_channels = 128;
    /* Max sequence length must accommodate image + text tokens combined.
     * At 1024x1024: img_seq = (1024/8)^2 = 16384, txt_seq = 512, total = 16896
     * At 2048x2048: img_seq = (2048/8)^2 = 65536 (requires 17GB+ for attn scores!)
     * We set 18000 to support up to ~1024x1024 with margin.
     */
    tf->max_seq_len = 18000;
    tf->rope_dim = 128;
    tf->rope_theta = 2000.0f;

    int h = tf->hidden_size;
    int mlp = tf->mlp_hidden;

    /* Input projections */
    tf->img_in_weight = get_sf_tensor_tf(sf, "x_embedder.weight");
    tf->txt_in_weight = get_sf_tensor_tf(sf, "context_embedder.weight");

    /* Time embedding
     * FLUX.2-klein uses 256-dim sinusoidal embedding (128 frequencies)
     * linear_1: [3072, 256], linear_2: [3072, 3072]
     */
    tf->time_embed.sincos_dim = 256;
    tf->time_embed.fc1_weight = get_sf_tensor_tf(sf,
        "time_guidance_embed.timestep_embedder.linear_1.weight");
    tf->time_embed.fc2_weight = get_sf_tensor_tf(sf,
        "time_guidance_embed.timestep_embedder.linear_2.weight");

    /* Modulation weights */
    tf->adaln_double_img_weight = get_sf_tensor_tf(sf,
        "double_stream_modulation_img.linear.weight");
    tf->adaln_double_txt_weight = get_sf_tensor_tf(sf,
        "double_stream_modulation_txt.linear.weight");
    tf->adaln_single_weight = get_sf_tensor_tf(sf,
        "single_stream_modulation.linear.weight");

    /* Double blocks */
    tf->double_blocks = calloc(tf->num_double_layers, sizeof(double_block_t));
    for (int i = 0; i < tf->num_double_layers; i++) {
        double_block_t *b = &tf->double_blocks[i];

        /* Image attention - QK norm weights */
        snprintf(name, sizeof(name), "transformer_blocks.%d.attn.norm_q.weight", i);
        b->img_norm_q_weight = get_sf_tensor_tf(sf, name);
        snprintf(name, sizeof(name), "transformer_blocks.%d.attn.norm_k.weight", i);
        b->img_norm_k_weight = get_sf_tensor_tf(sf, name);

        /* Image Q, K, V projections (separate) */
        snprintf(name, sizeof(name), "transformer_blocks.%d.attn.to_q.weight", i);
        b->img_q_weight = get_sf_tensor_tf(sf, name);
        snprintf(name, sizeof(name), "transformer_blocks.%d.attn.to_k.weight", i);
        b->img_k_weight = get_sf_tensor_tf(sf, name);
        snprintf(name, sizeof(name), "transformer_blocks.%d.attn.to_v.weight", i);
        b->img_v_weight = get_sf_tensor_tf(sf, name);

        snprintf(name, sizeof(name), "transformer_blocks.%d.attn.to_out.0.weight", i);
        b->img_proj_weight = get_sf_tensor_tf(sf, name);

        /* Image FFN - linear_in contains gate and up fused (18432 = 2*9216) */
        snprintf(name, sizeof(name), "transformer_blocks.%d.ff.linear_in.weight", i);
        float *ff_in = get_sf_tensor_tf(sf, name);
        if (ff_in) {
            /* Split into gate and up */
            b->img_mlp_gate_weight = malloc(mlp * h * sizeof(float));
            b->img_mlp_up_weight = malloc(mlp * h * sizeof(float));
            memcpy(b->img_mlp_gate_weight, ff_in, mlp * h * sizeof(float));
            memcpy(b->img_mlp_up_weight, ff_in + mlp * h, mlp * h * sizeof(float));
            free(ff_in);
        }

        snprintf(name, sizeof(name), "transformer_blocks.%d.ff.linear_out.weight", i);
        b->img_mlp_down_weight = get_sf_tensor_tf(sf, name);

        /* Text stream - QK norm weights */
        snprintf(name, sizeof(name), "transformer_blocks.%d.attn.norm_added_q.weight", i);
        b->txt_norm_q_weight = get_sf_tensor_tf(sf, name);
        snprintf(name, sizeof(name), "transformer_blocks.%d.attn.norm_added_k.weight", i);
        b->txt_norm_k_weight = get_sf_tensor_tf(sf, name);

        /* Text Q, K, V projections (separate) */
        snprintf(name, sizeof(name), "transformer_blocks.%d.attn.add_q_proj.weight", i);
        b->txt_q_weight = get_sf_tensor_tf(sf, name);
        snprintf(name, sizeof(name), "transformer_blocks.%d.attn.add_k_proj.weight", i);
        b->txt_k_weight = get_sf_tensor_tf(sf, name);
        snprintf(name, sizeof(name), "transformer_blocks.%d.attn.add_v_proj.weight", i);
        b->txt_v_weight = get_sf_tensor_tf(sf, name);

        snprintf(name, sizeof(name), "transformer_blocks.%d.attn.to_add_out.weight", i);
        b->txt_proj_weight = get_sf_tensor_tf(sf, name);

        snprintf(name, sizeof(name), "transformer_blocks.%d.ff_context.linear_in.weight", i);
        float *txt_ff_in = get_sf_tensor_tf(sf, name);
        if (txt_ff_in) {
            b->txt_mlp_gate_weight = malloc(mlp * h * sizeof(float));
            b->txt_mlp_up_weight = malloc(mlp * h * sizeof(float));
            memcpy(b->txt_mlp_gate_weight, txt_ff_in, mlp * h * sizeof(float));
            memcpy(b->txt_mlp_up_weight, txt_ff_in + mlp * h, mlp * h * sizeof(float));
            free(txt_ff_in);
        }

        snprintf(name, sizeof(name), "transformer_blocks.%d.ff_context.linear_out.weight", i);
        b->txt_mlp_down_weight = get_sf_tensor_tf(sf, name);
    }

    /* Single blocks */
    tf->single_blocks = calloc(tf->num_single_layers, sizeof(single_block_t));
    for (int i = 0; i < tf->num_single_layers; i++) {
        single_block_t *b = &tf->single_blocks[i];

        /* QK norm weights */
        snprintf(name, sizeof(name), "single_transformer_blocks.%d.attn.norm_q.weight", i);
        b->norm_q_weight = get_sf_tensor_tf(sf, name);
        snprintf(name, sizeof(name), "single_transformer_blocks.%d.attn.norm_k.weight", i);
        b->norm_k_weight = get_sf_tensor_tf(sf, name);

        snprintf(name, sizeof(name), "single_transformer_blocks.%d.attn.to_qkv_mlp_proj.weight", i);
        b->qkv_mlp_weight = get_sf_tensor_tf(sf, name);

        snprintf(name, sizeof(name), "single_transformer_blocks.%d.attn.to_out.weight", i);
        b->proj_mlp_weight = get_sf_tensor_tf(sf, name);
    }

    /* Final layer */
    tf->final_norm_weight = get_sf_tensor_tf(sf, "norm_out.linear.weight");
    tf->final_proj_weight = get_sf_tensor_tf(sf, "proj_out.weight");

    /* Precompute RoPE frequencies */
    tf->rope_freqs = malloc(tf->max_seq_len * tf->head_dim * sizeof(float));
    if (tf->rope_freqs) {
        compute_rope_freqs(tf->rope_freqs, tf->max_seq_len, tf->head_dim, tf->rope_theta);
    }

    /* Allocate working memory */
    int max_seq = tf->max_seq_len;
    int hidden = tf->hidden_size;
    tf->img_hidden = malloc(max_seq * hidden * sizeof(float));
    tf->txt_hidden = malloc(max_seq * hidden * sizeof(float));
    tf->work_size = max_seq * hidden * 4 * sizeof(float);
    tf->work1 = malloc(tf->work_size);
    tf->work2 = malloc(tf->work_size);

    /* Pre-allocated attention workspaces to avoid malloc in hot path */
    tf->attn_q_t = malloc(max_seq * hidden * sizeof(float));
    tf->attn_k_t = malloc(max_seq * hidden * sizeof(float));
    tf->attn_v_t = malloc(max_seq * hidden * sizeof(float));
    tf->attn_out_t = malloc(max_seq * hidden * sizeof(float));
    tf->attn_scores = malloc(max_seq * max_seq * sizeof(float));
    tf->attn_cat_k = malloc(max_seq * hidden * sizeof(float));
    tf->attn_cat_v = malloc(max_seq * hidden * sizeof(float));

    if (!tf->img_hidden || !tf->txt_hidden || !tf->work1 || !tf->work2 ||
        !tf->attn_q_t || !tf->attn_k_t || !tf->attn_v_t || !tf->attn_out_t ||
        !tf->attn_scores || !tf->attn_cat_k || !tf->attn_cat_v) {
        flux_transformer_free(tf);
        return NULL;
    }

    return tf;
}
