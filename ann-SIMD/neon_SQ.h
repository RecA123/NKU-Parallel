#pragma once
#include <arm_neon.h>
#include <vector>
#include <queue>
#include <algorithm>
#include <cmath>
#include <omp.h>

struct SQIndex {
    std::vector<uint8_t> quantized_data;  //量化后的数据
    std::vector<float> scales;            //每个维度的缩放因子
    std::vector<float> offsets;           //每个维度的最小值
    size_t vecdim;
};

void build_sq_index(const float* base, size_t base_number, size_t vecdim, SQIndex& index) {
    index.vecdim = vecdim;
    index.quantized_data.resize(base_number * vecdim);  // 改为uint8_t类型
    index.scales.resize(vecdim);
    index.offsets.resize(vecdim);
    
    #pragma omp parallel for
    for (size_t d = 0; d < vecdim; ++d) {
        // 计算最小最大值 (SIMD优化)
        float32x4_t min_v = vdupq_n_f32(INFINITY);
        float32x4_t max_v = vdupq_n_f32(-INFINITY);
        
        size_t i = 0;
        for (; i + 4 <= base_number; i += 4) {
            float32x4_t vals = vld1q_f32(&base[i * vecdim + d]);
            min_v = vminq_f32(min_v, vals);
            max_v = vmaxq_f32(max_v, vals);
        }
        
        // 提取最小最大值
        float min_val = vminvq_f32(min_v);
        float max_val = vmaxvq_f32(max_v);
        
        // 处理尾部元素
        for (; i < base_number; ++i) {
            float val = base[i * vecdim + d];
            min_val = std::min(min_val, val);
            max_val = std::max(max_val, val);
        }
        
        // 存储量化参数
        index.offsets[d] = min_val;
        const float range = max_val - min_val;
        index.scales[d] = (range > 1e-12f) ? (255.0f / range) : 0.0f;

        // 处理uniform维度
        if (range <= 1e-12f) {
            std::memset(&index.quantized_data[d], 0, base_number); // 整列设为0
            continue;
        }

        // 量化数据 (SIMD优化)
        const float32x4_t min_vec = vdupq_n_f32(min_val);
        const float32x4_t scale_vec = vdupq_n_f32(255.0f / range);

        i = 0;
        for (; i + 16 <= base_number; i += 16) {
            // 加载并处理16个向量（4x4交错加载）
            float32x4x4_t vals = vld4q_f32(&base[i * vecdim + d]);
    
            // 量化计算并转换为8-bit
            uint8x8_t quantized_low_0 = vqmovn_u16(vcombine_u16(
                vqmovn_u32(vcvtq_u32_f32(vmulq_f32(vsubq_f32(vals.val[0], min_vec), scale_vec))),
                vqmovn_u32(vcvtq_u32_f32(vmulq_f32(vsubq_f32(vals.val[1], min_vec), scale_vec)))));
    
            uint8x8_t quantized_low_1 = vqmovn_u16(vcombine_u16(
                vqmovn_u32(vcvtq_u32_f32(vmulq_f32(vsubq_f32(vals.val[2], min_vec), scale_vec))),
                vqmovn_u32(vcvtq_u32_f32(vmulq_f32(vsubq_f32(vals.val[3], min_vec), scale_vec)))));
    
            uint8x16_t quantized = vcombine_u8(quantized_low_0, quantized_low_1);
    
            // 存储量化结果
            vst1q_u8(&index.quantized_data[i * vecdim + d], quantized);
        }
        
        // 处理剩余向量（标量处理）
        for (; i < base_number; ++i) {
            const float val = base[i * vecdim + d];
            index.quantized_data[i * vecdim + d] = static_cast<uint8_t>(
                std::round((val - min_val) * (255.0f / range)));
        }
    }
}


//SQ搜索函数
std::priority_queue<std::pair<float, uint32_t>> sq_search(
    const SQIndex& index, const float* query, size_t base_number, size_t k) {
    
    std::priority_queue<std::pair<float, uint32_t>> top_k;
    const size_t vecdim = index.vecdim;
    
    // 预加载查询向量到寄存器（每次处理16维）
    float32x4_t query_vec[4];
    for (size_t d = 0; d + 15 < vecdim; d += 16) {
        query_vec[0] = vld1q_f32(&query[d]);
        query_vec[1] = vld1q_f32(&query[d+4]);
        query_vec[2] = vld1q_f32(&query[d+8]);
        query_vec[3] = vld1q_f32(&query[d+12]);
    }
    
    // 预计算缩放系数（scale/255）
    std::vector<float> scaled_factors(vecdim);
    for (size_t d = 0; d < vecdim; ++d) {
        scaled_factors[d] = index.scales[d] / 255.0f;
    }
    
    #pragma omp parallel
    {
        std::priority_queue<std::pair<float, uint32_t>> local_queue;
        
        #pragma omp for nowait
        for (size_t i = 0; i < base_number; ++i) {
            float32x4_t sum = vdupq_n_f32(0.0f);
            size_t d = 0;
            
            // 主处理循环：每次处理16个维度
            for (; d + 15 < vecdim; d += 16) {
                // 加载8-bit量化数据（16个连续值）
                uint8x16_t qvals = vld1q_u8(&index.quantized_data[i * vecdim + d]);
                
                // 反量化计算: offset + quantized * (scale/255)
                float32x4_t dequant[4];
                dequant[0] = vmlaq_f32(
                    vld1q_f32(&index.offsets[d]),
                    vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(vget_low_u8(qvals))))),
                    vld1q_f32(&scaled_factors[d]));
                dequant[1] = vmlaq_f32(
                    vld1q_f32(&index.offsets[d+4]),
                    vcvtq_f32_u32(vmovl_u16(vget_high_u16(vmovl_u8(vget_low_u8(qvals))))),
                    vld1q_f32(&scaled_factors[d+4]));
                dequant[2] = vmlaq_f32(
                    vld1q_f32(&index.offsets[d+8]),
                    vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(vget_high_u8(qvals))))),
                    vld1q_f32(&scaled_factors[d+8]));
                dequant[3] = vmlaq_f32(
                    vld1q_f32(&index.offsets[d+12]),
                    vcvtq_f32_u32(vmovl_u16(vget_high_u16(vmovl_u8(vget_high_u8(qvals))))),
                    vld1q_f32(&scaled_factors[d+12]));
                
                // 点积累加
                sum = vmlaq_f32(sum, dequant[0], query_vec[0]);
                sum = vmlaq_f32(sum, dequant[1], query_vec[1]);
                sum = vmlaq_f32(sum, dequant[2], query_vec[2]);
                sum = vmlaq_f32(sum, dequant[3], query_vec[3]);
            }
            
            // 计算相似度（1 - 内积）
            float distance = neon_hadd(sum);
            
            // 处理剩余维度（标量处理）
            for (; d < vecdim; ++d) {
                float val = index.offsets[d] + 
                          index.quantized_data[i * vecdim + d] * scaled_factors[d];
                distance += val * query[d];
            }

            //distance = 1.0f - distance;
            
            // 维护线程局部top-k
            if (local_queue.size() < k) {
                local_queue.emplace(distance, i);
            } else if (distance < local_queue.top().first) {
                local_queue.pop();
                local_queue.emplace(distance, i);
            }
        }
        
        // 合并线程局部结果
        #pragma omp critical
        {
            while (!local_queue.empty()) {
                if (top_k.size() < k) {
                    top_k.push(local_queue.top());
                } else if (local_queue.top().first < top_k.top().first) {
                    top_k.pop();
                    top_k.push(local_queue.top());
                }
                local_queue.pop();
            }
        }
    }
    
    return top_k;
}