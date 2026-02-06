#include <metal_stdlib>
using namespace metal;

#define TILE_SIZE 32

// Flash Attention 커널 - 완전한 구현
[[kernel]] void flash_attn_kernel(
    const device float* Q [[buffer(0)]],
    const device float* K [[buffer(1)]],
    const device float* V [[buffer(2)]],
    device float* O [[buffer(3)]],
    constant int& N [[buffer(4)]],  // 시퀀스 길이
    constant int& D [[buffer(5)]],  // 헤드 차원
    uint3 threadgroup_pos [[threadgroup_position_in_grid]],
    uint thread_idx [[thread_index_in_threadgroup]]
) {
    // ========================================
    // 1. Threadgroup 메모리 선언
    // ========================================
    threadgroup float s_Q[TILE_SIZE * 64];  // Q 타일 (최대 D=64 가정)
    threadgroup float s_K[TILE_SIZE * 64];  // K 타일
    threadgroup float s_V[TILE_SIZE * 64];  // V 타일
    threadgroup float s_sum[TILE_SIZE];     // Softmax 분모 누적
    threadgroup float s_max[TILE_SIZE];     // 각 행의 최댓값
    
    // ========================================
    // 2. 인덱스 및 오프셋 계산
    // ========================================
    int head_idx = threadgroup_pos.x;       // 헤드 번호
    int q_block_idx = threadgroup_pos.y;    // Q의 블록 인덱스
    int qkv_offset = head_idx * N * D;      // 현재 헤드의 시작 오프셋
    
    // 현재 스레드가 담당하는 Q의 행 번호
    int q_row = q_block_idx * TILE_SIZE + int(thread_idx);
    
    // ========================================
    // 3. Q 타일 로딩 (한 번만 수행)
    // ========================================
    if (q_row < N) {
        for (int i = 0; i < D; ++i) {
            s_Q[thread_idx * D + i] = Q[qkv_offset + q_row * D + i];
        }
    } else {
        for (int i = 0; i < D; ++i) {
            s_Q[thread_idx * D + i] = 0.0f;
        }
    }
    
    // Online Softmax 초기화
    if (thread_idx < TILE_SIZE) {
        s_sum[thread_idx] = 0.0f;
        s_max[thread_idx] = -INFINITY;
    }
    
    // 출력 레지스터 초기화
    float o_reg[64] = {0.0f};  // 최대 D=64 가정
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // ========================================
    // 4. K, V 타일을 순회하며 계산
    // ========================================
    int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        // ─────────────────────────────────────
        // [단계 4-1] K 타일 로딩
        // ─────────────────────────────────────
        int k_row = tile_idx * TILE_SIZE + int(thread_idx);
        if (k_row < N) {
            for (int i = 0; i < D; ++i) {
                s_K[thread_idx * D + i] = K[qkv_offset + k_row * D + i];
            }
        } else {
            for (int i = 0; i < D; ++i) {
                s_K[thread_idx * D + i] = 0.0f;
            }
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // ─────────────────────────────────────
        // [단계 4-2] Q × K^T 계산 (Attention Scores)
        // ─────────────────────────────────────
        float old_max = s_max[thread_idx];
        float row_max = old_max;
        
        // 현재 타일 내의 모든 K 행과 내적
        float scores[TILE_SIZE];
        for (int j = 0; j < TILE_SIZE; ++j) {
            int k_col = tile_idx * TILE_SIZE + j;
            if (k_col < N && q_row < N) {
                // 내적 계산: Q[thread_idx] · K[j]
                float score = 0.0f;
                for (int k = 0; k < D; ++k) {
                    score += s_Q[thread_idx * D + k] * s_K[j * D + k];
                }
                // Scaled Dot-Product
                score *= (1.0f / sqrt(float(D)));
                scores[j] = score;
                // 현재 타일에서의 최댓값 갱신
                row_max = max(row_max, score);
            } else {
                scores[j] = -INFINITY;
            }
        }
        
        // ─────────────────────────────────────
        // [단계 4-3] Online Softmax 업데이트
        // ─────────────────────────────────────
        float new_max = row_max;
        float correction_factor = exp(old_max - new_max);
        
        // 기존 합을 새로운 스케일로 보정
        float new_sum = s_sum[thread_idx] * correction_factor;
        
        // 기존 출력도 보정
        for (int d = 0; d < D; ++d) {
            o_reg[d] *= correction_factor;
        }
        
        // 현재 타일의 exp(score) 계산 및 누적
        float tile_exp_sum = 0.0f;
        float exp_scores[TILE_SIZE];
        for (int j = 0; j < TILE_SIZE; ++j) {
            exp_scores[j] = exp(scores[j] - new_max);
            tile_exp_sum += exp_scores[j];
        }
        new_sum += tile_exp_sum;
        
        // s_max, s_sum 업데이트
        s_max[thread_idx] = new_max;
        s_sum[thread_idx] = new_sum;
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // ─────────────────────────────────────
        // [단계 4-4] V 타일 로딩
        // ─────────────────────────────────────
        int v_row = tile_idx * TILE_SIZE + int(thread_idx);
        if (v_row < N) {
            for (int i = 0; i < D; ++i) {
                s_V[thread_idx * D + i] = V[qkv_offset + v_row * D + i];
            }
        } else {
            for (int i = 0; i < D; ++i) {
                s_V[thread_idx * D + i] = 0.0f;
            }
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // ─────────────────────────────────────
        // [단계 4-5] Attention Weight × V 누적
        // ─────────────────────────────────────
        // O += exp(score - max) × V
        for (int d = 0; d < D; ++d) {
            float weighted_value = 0.0f;
            for (int j = 0; j < TILE_SIZE; ++j) {
                weighted_value += exp_scores[j] * s_V[j * D + d];
            }
            o_reg[d] += weighted_value;
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // ========================================
    // 5. 최종 정규화 및 출력 쓰기
    // ========================================
    if (q_row < N) {
        float normalizer = 1.0f / s_sum[thread_idx];
        for (int d = 0; d < D; ++d) {
            O[qkv_offset + q_row * D + d] = o_reg[d] * normalizer;
        }
    }
}

// ========================================
// 최적화된 버전 - Register Tiling 사용
// ========================================
[[kernel]] void flash_attn_kernel_optimized(
    const device float* Q [[buffer(0)]],
    const device float* K [[buffer(1)]],
    const device float* V [[buffer(2)]],
    device float* O [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& D [[buffer(5)]],
    uint3 threadgroup_pos [[threadgroup_position_in_grid]],
    uint thread_idx [[thread_index_in_threadgroup]]
) {
    // Threadgroup 메모리
    threadgroup float s_Q[TILE_SIZE * 64];
    threadgroup float s_K[TILE_SIZE * 64];
    threadgroup float s_V[TILE_SIZE * 64];
    threadgroup float s_sum[TILE_SIZE];
    threadgroup float s_max[TILE_SIZE];
    
    int head_idx = threadgroup_pos.x;
    int q_block_idx = threadgroup_pos.y;
    int qkv_offset = head_idx * N * D;
    
    int q_row = q_block_idx * TILE_SIZE + int(thread_idx);
    
    // Q 로딩 - Vectorized Load (float4 사용)
    if (q_row < N && D % 4 == 0) {
        for (int i = 0; i < D / 4; ++i) {
            float4 q_vec = *reinterpret_cast<const device float4*>(
                &Q[qkv_offset + q_row * D + i * 4]
            );
            *reinterpret_cast<threadgroup float4*>(
                &s_Q[thread_idx * D + i * 4]
            ) = q_vec;
        }
    }
    
    if (thread_idx < TILE_SIZE) {
        s_sum[thread_idx] = 0.0f;
        s_max[thread_idx] = -INFINITY;
    }
    
    float o_reg[64] = {0.0f};
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        // K 로딩 - Vectorized
        int k_row = tile_idx * TILE_SIZE + int(thread_idx);
        if (k_row < N && D % 4 == 0) {
            for (int i = 0; i < D / 4; ++i) {
                float4 k_vec = *reinterpret_cast<const device float4*>(
                    &K[qkv_offset + k_row * D + i * 4]
                );
                *reinterpret_cast<threadgroup float4*>(
                    &s_K[thread_idx * D + i * 4]
                ) = k_vec;
            }
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Q × K^T - SIMD 최적화
        float old_max = s_max[thread_idx];
        float row_max = old_max;
        
        float scores[TILE_SIZE];
        for (int j = 0; j < TILE_SIZE; ++j) {
            int k_col = tile_idx * TILE_SIZE + j;
            if (k_col < N && q_row < N) {
                float score = 0.0f;
                // SIMD를 활용한 내적 (4개씩 묶어서 계산)
                for (int k = 0; k < D / 4; ++k) {
                    float4 q_vec = *reinterpret_cast<threadgroup float4*>(
                        &s_Q[thread_idx * D + k * 4]
                    );
                    float4 k_vec = *reinterpret_cast<threadgroup float4*>(
                        &s_K[j * D + k * 4]
                    );
                    // Dot product
                    score += dot(q_vec, k_vec);
                }
                score *= (1.0f / sqrt(float(D)));
                scores[j] = score;
                row_max = max(row_max, score);
            } else {
                scores[j] = -INFINITY;
            }
        }
        
        // Online Softmax
        float new_max = row_max;
        float correction = exp(old_max - new_max);
        float new_sum = s_sum[thread_idx] * correction;
        
        for (int d = 0; d < D; ++d) {
            o_reg[d] *= correction;
        }
        
        float exp_scores[TILE_SIZE];
        for (int j = 0; j < TILE_SIZE; ++j) {
            exp_scores[j] = exp(scores[j] - new_max);
            new_sum += exp_scores[j];
        }
        
        s_max[thread_idx] = new_max;
        s_sum[thread_idx] = new_sum;
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // V 로딩
        int v_row = tile_idx * TILE_SIZE + int(thread_idx);
        if (v_row < N && D % 4 == 0) {
            for (int i = 0; i < D / 4; ++i) {
                float4 v_vec = *reinterpret_cast<const device float4*>(
                    &V[qkv_offset + v_row * D + i * 4]
                );
                *reinterpret_cast<threadgroup float4*>(
                    &s_V[thread_idx * D + i * 4]
                ) = v_vec;
            }
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Attention Weight × V
        for (int d = 0; d < D; ++d) {
            float weighted = 0.0f;
            for (int j = 0; j < TILE_SIZE; ++j) {
                weighted += exp_scores[j] * s_V[j * D + d];
            }
            o_reg[d] += weighted;
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // 최종 출력
    if (q_row < N) {
        float normalizer = 1.0f / s_sum[thread_idx];
        if (D % 4 == 0) {
            for (int i = 0; i < D / 4; ++i) {
                float4 o_vec = float4(
                    o_reg[i * 4 + 0] * normalizer,
                    o_reg[i * 4 + 1] * normalizer,
                    o_reg[i * 4 + 2] * normalizer,
                    o_reg[i * 4 + 3] * normalizer
                );
                *reinterpret_cast<device float4*>(
                    &O[qkv_offset + q_row * D + i * 4]
                ) = o_vec;
            }
        } else {
            for (int d = 0; d < D; ++d) {
                O[qkv_offset + q_row * D + d] = o_reg[d] * normalizer;
            }
        }
    }
}
