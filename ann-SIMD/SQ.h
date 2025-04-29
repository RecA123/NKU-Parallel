#pragma once
#include <arm_neon.h>

#include <algorithm>
#include <cstdint>
#include <queue>
#include <vector>
#include <random>
#include <cmath>
#include <memory>
#include <functional>
#include <chrono>
#include <omp.h>
#include <fstream>

// 全局变量用于缓存已量化的向量
namespace {
    uint8_t* g_base_quantized = nullptr;
    float* g_scales = nullptr;
    float* g_offsets = nullptr;
    size_t g_base_number = 0;
    size_t g_vecdim = 0;
    bool g_initialized = false;
}

template <typename T>
T clamp(T value, T min_val, T max_val) {
    return std::max(min_val, std::min(value, max_val));
}

// 标量量化函数 - 将浮点数向量量化为8位整数
void quantize_vector(const float* src, uint8_t* dst, float& scale,
                     float& offset, size_t vecdim) {
    // 找出最大值和最小值
    float min_val = src[0];
    float max_val = src[0];

    // 使用SIMD优化查找最大最小值
    size_t i = 0;
    float32x4_t vmin = vdupq_n_f32(src[0]);
    float32x4_t vmax = vdupq_n_f32(src[0]);

    // 每次处理4个值
    for (; i + 3 < vecdim; i += 4) {
        float32x4_t v = vld1q_f32(src + i);
        vmin = vminq_f32(vmin, v);
        vmax = vmaxq_f32(vmax, v);
    }

    // 合并最小值和最大值
    float32x2_t min2 = vpmin_f32(vget_low_f32(vmin), vget_high_f32(vmin));
    min2 = vpmin_f32(min2, min2);
    min_val = vget_lane_f32(min2, 0);

    float32x2_t max2 = vpmax_f32(vget_low_f32(vmax), vget_high_f32(vmax));
    max2 = vpmax_f32(max2, max2);
    max_val = vget_lane_f32(max2, 0);

    // 处理剩余元素
    for (; i < vecdim; i++) {
        if (src[i] < min_val) min_val = src[i];
        if (src[i] > max_val) max_val = src[i];
    }

    // 计算缩放因子和偏移量
    scale = (max_val - min_val) / 255.0f;
    offset = min_val;

    if (min_val == max_val || scale < 1e-10) {
        scale = 1.0f;
        std::fill(dst, dst + vecdim, 0);  // 量化结果为0
        offset = min_val;
        return;
    }

    // 预计算量化系数，避免除法
    float inv_scale = 1.0f / scale;
    
    // 使用SIMD优化量化过程
    i = 0;
    float32x4_t voffset = vdupq_n_f32(offset);
    float32x4_t vinv_scale = vdupq_n_f32(inv_scale);
    float32x4_t vzero = vdupq_n_f32(0.0f);
    float32x4_t v255 = vdupq_n_f32(255.0f);
    
    for (; i + 3 < vecdim; i += 4) {
        float32x4_t v = vld1q_f32(src + i);
        // 归一化: (src - offset) / scale
        float32x4_t normalized = vmulq_f32(vsubq_f32(v, voffset), vinv_scale);
        // 裁剪到 [0, 255]
        normalized = vmaxq_f32(normalized, vzero);
        normalized = vminq_f32(normalized, v255);
        // 转为整数
        int32x4_t int_vals = vcvtq_s32_f32(normalized);
        // 转为 16位
        int16x4_t int16_vals = vmovn_s32(int_vals);
        // 转为 8位
        uint8x8_t uint8_vals = vreinterpret_u8_s8(vmovn_s16(vcombine_s16(int16_vals, int16_vals)));
        // 存储4个字节
        vst1_lane_u32((uint32_t*)(dst + i), vreinterpret_u32_u8(uint8_vals), 0);
    }

    // 处理剩余元素
    for (; i < vecdim; i++) {
        float normalized = (src[i] - offset) * inv_scale;
        dst[i] = static_cast<uint8_t>(clamp(std::round(normalized), 0.0f, 255.0f));
    }
}

// 使用量化计算内积距离
float inner_product_quantized(const uint8_t* b1, const uint8_t* b2,
                             const float scale1, const float offset1,
                             const float scale2, const float offset2,
                             size_t vecdim) {
    // SIMD优化的8位整数内积计算
    size_t i = 0;
    
    // 使用NEON进行SIMD加速
    int32x4_t sum_vec1 = vdupq_n_s32(0);
    int32x4_t sum_vec2 = vdupq_n_s32(0);
    int32x4_t sum_vec3 = vdupq_n_s32(0);
    int32x4_t sum_vec4 = vdupq_n_s32(0);
    
    // 累加b1和b2向量的和，后续反量化会用到
    uint32x4_t b1_sum_vec = vdupq_n_u32(0);
    uint32x4_t b2_sum_vec = vdupq_n_u32(0);

    // 每次处理64个元素，提高SIMD并行度
    for (; i + 63 < vecdim; i += 64) {
        // 处理第一组16个元素
        uint8x16_t v1_1 = vld1q_u8(b1 + i);
        uint8x16_t v2_1 = vld1q_u8(b2 + i);
        
        // 累加向量和
        b1_sum_vec = vaddq_u32(b1_sum_vec, vpaddlq_u16(vpaddlq_u8(v1_1)));
        b2_sum_vec = vaddq_u32(b2_sum_vec, vpaddlq_u16(vpaddlq_u8(v2_1)));

        // 转换为16位整数并计算乘积
        int16x8_t low1_1 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(v1_1)));
        int16x8_t high1_1 = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(v1_1)));
        int16x8_t low2_1 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(v2_1)));
        int16x8_t high2_1 = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(v2_1)));

        // 计算乘积并累加
        sum_vec1 = vmlal_s16(sum_vec1, vget_low_s16(low1_1), vget_low_s16(low2_1));
        sum_vec1 = vmlal_s16(sum_vec1, vget_high_s16(low1_1), vget_high_s16(low2_1));
        sum_vec1 = vmlal_s16(sum_vec1, vget_low_s16(high1_1), vget_low_s16(high2_1));
        sum_vec1 = vmlal_s16(sum_vec1, vget_high_s16(high1_1), vget_high_s16(high2_1));

        // 处理第二组16个元素
        uint8x16_t v1_2 = vld1q_u8(b1 + i + 16);
        uint8x16_t v2_2 = vld1q_u8(b2 + i + 16);
        
        // 累加向量和
        b1_sum_vec = vaddq_u32(b1_sum_vec, vpaddlq_u16(vpaddlq_u8(v1_2)));
        b2_sum_vec = vaddq_u32(b2_sum_vec, vpaddlq_u16(vpaddlq_u8(v2_2)));

        int16x8_t low1_2 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(v1_2)));
        int16x8_t high1_2 = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(v1_2)));
        int16x8_t low2_2 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(v2_2)));
        int16x8_t high2_2 = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(v2_2)));

        sum_vec2 = vmlal_s16(sum_vec2, vget_low_s16(low1_2), vget_low_s16(low2_2));
        sum_vec2 = vmlal_s16(sum_vec2, vget_high_s16(low1_2), vget_high_s16(low2_2));
        sum_vec2 = vmlal_s16(sum_vec2, vget_low_s16(high1_2), vget_low_s16(high2_2));
        sum_vec2 = vmlal_s16(sum_vec2, vget_high_s16(high1_2), vget_high_s16(high2_2));
        
        // 处理第三组16个元素
        uint8x16_t v1_3 = vld1q_u8(b1 + i + 32);
        uint8x16_t v2_3 = vld1q_u8(b2 + i + 32);
        
        // 累加向量和
        b1_sum_vec = vaddq_u32(b1_sum_vec, vpaddlq_u16(vpaddlq_u8(v1_3)));
        b2_sum_vec = vaddq_u32(b2_sum_vec, vpaddlq_u16(vpaddlq_u8(v2_3)));

        int16x8_t low1_3 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(v1_3)));
        int16x8_t high1_3 = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(v1_3)));
        int16x8_t low2_3 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(v2_3)));
        int16x8_t high2_3 = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(v2_3)));

        sum_vec3 = vmlal_s16(sum_vec3, vget_low_s16(low1_3), vget_low_s16(low2_3));
        sum_vec3 = vmlal_s16(sum_vec3, vget_high_s16(low1_3), vget_high_s16(low2_3));
        sum_vec3 = vmlal_s16(sum_vec3, vget_low_s16(high1_3), vget_low_s16(high2_3));
        sum_vec3 = vmlal_s16(sum_vec3, vget_high_s16(high1_3), vget_high_s16(high2_3));

        // 处理第四组16个元素
        uint8x16_t v1_4 = vld1q_u8(b1 + i + 48);
        uint8x16_t v2_4 = vld1q_u8(b2 + i + 48);
        
        // 累加向量和
        b1_sum_vec = vaddq_u32(b1_sum_vec, vpaddlq_u16(vpaddlq_u8(v1_4)));
        b2_sum_vec = vaddq_u32(b2_sum_vec, vpaddlq_u16(vpaddlq_u8(v2_4)));

        int16x8_t low1_4 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(v1_4)));
        int16x8_t high1_4 = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(v1_4)));
        int16x8_t low2_4 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(v2_4)));
        int16x8_t high2_4 = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(v2_4)));

        sum_vec4 = vmlal_s16(sum_vec4, vget_low_s16(low1_4), vget_low_s16(low2_4));
        sum_vec4 = vmlal_s16(sum_vec4, vget_high_s16(low1_4), vget_high_s16(low2_4));
        sum_vec4 = vmlal_s16(sum_vec4, vget_low_s16(high1_4), vget_low_s16(high2_4));
        sum_vec4 = vmlal_s16(sum_vec4, vget_high_s16(high1_4), vget_high_s16(high2_4));
    }

    // 继续处理16个元素一组的数据
    for (; i + 15 < vecdim; i += 16) {
        uint8x16_t v1 = vld1q_u8(b1 + i);
        uint8x16_t v2 = vld1q_u8(b2 + i);
        
        // 累加向量和
        b1_sum_vec = vaddq_u32(b1_sum_vec, vpaddlq_u16(vpaddlq_u8(v1)));
        b2_sum_vec = vaddq_u32(b2_sum_vec, vpaddlq_u16(vpaddlq_u8(v2)));

        int16x8_t low1 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(v1)));
        int16x8_t high1 = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(v1)));
        int16x8_t low2 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(v2)));
        int16x8_t high2 = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(v2)));

        sum_vec1 = vmlal_s16(sum_vec1, vget_low_s16(low1), vget_low_s16(low2));
        sum_vec1 = vmlal_s16(sum_vec1, vget_high_s16(low1), vget_high_s16(low2));
        sum_vec1 = vmlal_s16(sum_vec1, vget_low_s16(high1), vget_low_s16(high2));
        sum_vec1 = vmlal_s16(sum_vec1, vget_high_s16(high1), vget_high_s16(high2));
    }

    // 合并所有部分和
    int32x4_t total_sum_vec = vaddq_s32(vaddq_s32(sum_vec1, sum_vec2), vaddq_s32(sum_vec3, sum_vec4));
    
    // 将向量中的元素相加
    int32_t sum = vaddvq_s32(total_sum_vec);
    
    // 计算b1和b2向量的和
    uint32_t b1_sum = vaddvq_u32(b1_sum_vec);
    uint32_t b2_sum = vaddvq_u32(b2_sum_vec);

    // 处理剩余元素
    for (; i < vecdim; i++) {
        sum += static_cast<int32_t>(b1[i]) * static_cast<int32_t>(b2[i]);
        b1_sum += b1[i];
        b2_sum += b2[i];
    }

    // 反量化结果公式: sum(b1[i]*b2[i])*scale1*scale2 + sum(b1[i])*scale1*offset2 + sum(b2[i])*scale2*offset1 + vecdim*offset1*offset2
    // 优化反量化计算
    float scale_product = scale1 * scale2;
    float offset_product = offset1 * offset2;
    
    float ip = static_cast<float>(sum) * scale_product + 
              static_cast<float>(b1_sum) * scale1 * offset2 + 
              static_cast<float>(b2_sum) * scale2 * offset1 + 
              static_cast<float>(vecdim) * offset_product;
    
    return 1.0f - ip;  // 返回内积距离
}

// 初始化量化数据 - 只在第一次调用时执行量化
void init_quantized_data(float* base, size_t base_number, size_t vecdim) {
    if (g_initialized && g_base_number == base_number && g_vecdim == vecdim) {
        return; // 已经初始化过，无需重复操作
    }
    
    // 清理之前的内存（如果有）
    if (g_base_quantized) {
        delete[] g_base_quantized;
        delete[] g_scales;
        delete[] g_offsets;
    }
    
    // 分配新内存
    g_base_quantized = new uint8_t[base_number * vecdim + 64]; // 添加额外对齐空间
    g_scales = new float[base_number];
    g_offsets = new float[base_number];
    
    // 并行量化所有基础向量
    for (size_t i = 0; i < base_number; ++i) {
        quantize_vector(base + i * vecdim, g_base_quantized + i * vecdim,
                      g_scales[i], g_offsets[i], vecdim);
    }
    
    g_base_number = base_number;
    g_vecdim = vecdim;
    g_initialized = true;
}

// 释放量化数据的内存
void cleanup_quantized_data() {
    if (g_base_quantized) {
        delete[] g_base_quantized;
        delete[] g_scales;
        delete[] g_offsets;
        g_base_quantized = nullptr;
        g_scales = nullptr;
        g_offsets = nullptr;
        g_initialized = false;
    }
}

// 使用标量量化的扁平扫描搜索函数 - 优化版本
std::priority_queue<std::pair<float, uint32_t>> flat_search_with_SQ(
    float* base, float* query, size_t base_number, size_t vecdim, size_t k) {
      
    std::priority_queue<std::pair<float, uint32_t>> q;

    // 初始化/确保基础向量已量化
    init_quantized_data(base, base_number, vecdim);
    // 量化查询向量
    uint8_t* query_quantized = new uint8_t[vecdim];
    float query_scale, query_offset;
    quantize_vector(query, query_quantized, query_scale, query_offset, vecdim);

    // 并行计算距离
    std::vector<std::pair<float, uint32_t>> thread_results[omp_get_max_threads()];
    
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int num_threads = omp_get_num_threads();
        std::priority_queue<std::pair<float, uint32_t>> local_q;
        
        // 每个线程处理一部分向量
        size_t start = (base_number * thread_id) / num_threads;
        size_t end = (base_number * (thread_id + 1)) / num_threads;
        
        for (size_t i = start; i < end; ++i) {
            float dis = inner_product_quantized(
                g_base_quantized + i * vecdim, query_quantized, 
                g_scales[i], g_offsets[i],
                query_scale, query_offset, vecdim);

            // 保持 top-k 最小距离
            if (local_q.size() < k) {
                local_q.push({dis, static_cast<uint32_t>(i)});
            } else {
                if (dis < local_q.top().first) {
                    local_q.pop();
                    local_q.push({dis, static_cast<uint32_t>(i)});
                }
            }
        }
        
        // 将局部结果保存到线程结果数组
        while (!local_q.empty()) {
            thread_results[thread_id].push_back(local_q.top());
            local_q.pop();
        }
    }
    
    // 合并所有线程的结果
    for (int t = 0; t < omp_get_max_threads(); ++t) {
        for (const auto& res : thread_results[t]) {
            if (q.size() < k) {
                q.push(res);
            } else if (res.first < q.top().first) {
                q.pop();
                q.push(res);
            }
        }
    }

    // 释放本次查询分配的内存
    delete[] query_quantized;

    return q;
}

// 析构函数释放全局内存 - 在程序结束时调用
class GlobalMemoryCleanup {
public:
    ~GlobalMemoryCleanup() {
        cleanup_quantized_data();
    }
};

static GlobalMemoryCleanup g_cleanup;