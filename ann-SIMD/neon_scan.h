#include <arm_neon.h>
#include <cstdint>
#include <queue>
#include <algorithm>
#include <stdlib.h>
#include <malloc.h>

// 优化的水平相加函数
inline float neon_hadd(float32x4_t v) {
    float32x2_t vlow = vget_low_f32(v);
    float32x2_t vhigh = vget_high_f32(v);
    float32x2_t sum2 = vadd_f32(vlow, vhigh);
    return vget_lane_f32(vpadd_f32(sum2, sum2), 0);
}

std::priority_queue<std::pair<float, uint32_t>> 
flat_search_neon(float* base, float* query, size_t base_number, size_t vecdim, size_t k) {
    std::priority_queue<std::pair<float, uint32_t>> top_k;
    
    // 确保查询向量16字节对齐
    float* aligned_query = (float*)memalign(16, vecdim * sizeof(float));
    if (!aligned_query) {
        // 处理内存分配失败
        return top_k;
    }
    memcpy(aligned_query, query, vecdim * sizeof(float));
    
    for (uint32_t i = 0; i < base_number; ++i) {
        const float* vec = base + i * vecdim;
        float32x4_t sum = vdupq_n_f32(0.0f);
        size_t d = 0;
        
        // 主处理循环：每次处理16个元素(4个NEON寄存器)
        for (; d + 15 < vecdim; d += 16) {
            // 加载查询向量部分
            float32x4_t q0 = vld1q_f32(aligned_query + d);
            float32x4_t q1 = vld1q_f32(aligned_query + d + 4);
            float32x4_t q2 = vld1q_f32(aligned_query + d + 8);
            float32x4_t q3 = vld1q_f32(aligned_query + d + 12);
            
            // 加载基准向量部分
            float32x4_t v0 = vld1q_f32(vec + d);
            float32x4_t v1 = vld1q_f32(vec + d + 4);
            float32x4_t v2 = vld1q_f32(vec + d + 8);
            float32x4_t v3 = vld1q_f32(vec + d + 12);
            
            // 乘积累加
            sum = vmlaq_f32(sum, v0, q0);
            sum = vmlaq_f32(sum, v1, q1);
            sum = vmlaq_f32(sum, v2, q2);
            sum = vmlaq_f32(sum, v3, q3);
        }
        
        // 处理剩余8个元素
        for (; d + 7 < vecdim; d += 8) {
            float32x4_t q0 = vld1q_f32(aligned_query + d);
            float32x4_t q1 = vld1q_f32(aligned_query + d + 4);
            
            float32x4_t v0 = vld1q_f32(vec + d);
            float32x4_t v1 = vld1q_f32(vec + d + 4);
            
            sum = vmlaq_f32(sum, v0, q0);
            sum = vmlaq_f32(sum, v1, q1);
        }
        
        // 计算最终距离
        float dis = 1 - neon_hadd(sum);
        
        // 处理尾部剩余元素(不足8个)
        for (; d < vecdim; ++d) {
            dis += vec[d] * aligned_query[d];
        }
        
        // 更新top-k队列
        if (top_k.size() < k) {
            top_k.emplace(dis, i);
        } else if (dis < top_k.top().first) {
            top_k.pop();
            top_k.emplace(dis, i);
        }
    }
    
    free(aligned_query);
    return top_k;
}