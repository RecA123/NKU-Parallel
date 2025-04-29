#pragma once
#include <arm_neon.h>
#include <cstdlib>
#include <cassert>

struct simd8float32 {
    float32x4x2_t data;

    // 默认构造函数
    simd8float32() = default;

    // 从 float 数组构造，加载 8 个 float 到两个寄存器
    explicit simd8float32(const float* x) {
        data.val[0] = vld1q_f32(x);      // 加载前 4 个
        data.val[1] = vld1q_f32(x + 4);  // 加载后 4 个
    }

    // 从标量初始化所有元素
    explicit simd8float32(float val) {
        data.val[0] = vdupq_n_f32(val);
        data.val[1] = vdupq_n_f32(val);
    }

    // 元素级乘法
    simd8float32 operator*(const simd8float32& other) const {
        simd8float32 result;
        result.data.val[0] = vmulq_f32(data.val[0], other.data.val[0]);
        result.data.val[1] = vmulq_f32(data.val[1], other.data.val[1]);
        return result;
    }

    // += 复合操作符
    simd8float32& operator+=(const simd8float32& other) {
        data.val[0] = vaddq_f32(data.val[0], other.data.val[0]);
        data.val[1] = vaddq_f32(data.val[1], other.data.val[1]);
        return *this;
    }

    // 存储结果到 float 数组中
    void storeu(float* out) const {
        vst1q_f32(out, data.val[0]);
        vst1q_f32(out + 4, data.val[1]);
    }
};

// 使用 SIMD Neon 实现内积距离计算
float InnerProductSIMDNeon(const float* b1, const float* b2, size_t vecdim) {
    assert(vecdim % 8 == 0); // 假设维度能被 8 整除

    simd8float32 sum(0.0f); // 初始化为 0
    for (size_t i = 0; i < vecdim; i += 8) {
        simd8float32 s1(b1 + i), s2(b2 + i);
        simd8float32 m = s1 * s2; // 元素级乘法
        sum += m;                 // 累加
    }

    float tmp[8];
    sum.storeu(tmp); // 将结果存储到数组中

    // 求和
    float dis = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
    return 1.0f - dis; // 返回内积距离
}