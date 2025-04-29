#pragma once
#include <iostream>
#include <algorithm>
#include <arm_neon.h>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <omp.h>
#include <queue>
#include <random>
#include <vector>
#include "simd_utils.h"

// PQ算法实现 - 针对内积距离优化
class ProductQuantizer {
public:
  // 构造函数
  ProductQuantizer(size_t d, size_t M = 8, size_t K = 256)
      : dim(d), M(M), K(K) {
    // 确保维度可以被M整除
    if (dim % M != 0) {
      throw std::runtime_error("维度必须被子空间数量整除");
    }

    // 每个子向量的维度
    d_sub = dim / M;

    // 初始化编码表 (M个子空间，每个子空间K个类中心)
    codebooks.resize(M);
    for (size_t m = 0; m < M; m++) {
      codebooks[m].resize(K * d_sub);
    }

    // 初始化距离表
    distance_tables.resize(M);
    for (size_t m = 0; m < M; m++) {
      distance_tables[m].resize(K);
    }
  }

  // 训练量化器 - 针对内积距离优化
  void train(const std::vector<float> &data, size_t n) {
    // 随机数生成器
    std::mt19937 rng(
        std::chrono::system_clock::now().time_since_epoch().count());

    // 数据的维度检查
    if (data.size() != n * dim) {
      throw std::runtime_error("数据大小与指定的样本数和维度不匹配");
    }

    // 标准化数据 - 这对内积距离很重要
    std::vector<float> normalized_data(n * dim);
    for (size_t i = 0; i < n; i++) {
      float norm = 0.0f;
      for (size_t j = 0; j < dim; j++) {
        float val = data[i * dim + j];
        norm += val * val;
      }
      norm = std::sqrt(norm) + 1e-8f; // 避免除零

      for (size_t j = 0; j < dim; j++) {
        normalized_data[i * dim + j] = data[i * dim + j] / norm;
      }
    }

    // 对每个子空间单独进行K-means聚类
    #pragma omp parallel for
    for (size_t m = 0; m < M; m++) {
      // 提取当前子空间的数据
      std::vector<float> sub_data(n * d_sub);
      for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < d_sub; j++) {
          sub_data[i * d_sub + j] = normalized_data[i * dim + m * d_sub + j];
        }
      }

      // 优化版K-means聚类，专为内积距离优化
      optimized_kmeans_ip(sub_data.data(), n, d_sub, K, codebooks[m].data(), rng);
    }
  }

  // 编码单个向量 - 使用内积距离
  void encode(const float *x, uint8_t *code) const {
    // 首先标准化向量
    std::vector<float> x_norm(dim);
    float norm = 0.0f;
    for (size_t j = 0; j < dim; j++) {
      float val = x[j];
      norm += val * val;
    }
    norm = std::sqrt(norm) + 1e-8f;

    for (size_t j = 0; j < dim; j++) {
      x_norm[j] = x[j] / norm;
    }

    for (size_t m = 0; m < M; m++) {
      float max_dot = -std::numeric_limits<float>::max();
      uint8_t best_idx = 0;

      // 计算当前子向量与所有聚类中心的内积
      for (size_t k = 0; k < K; k++) {
        float dot = 0.0f;

        // 使用SIMD优化内积计算
        if (d_sub >= 4) {
          size_t j = 0;
          float32x4_t sum_vec = vdupq_n_f32(0);

          for (; j + 3 < d_sub; j += 4) {
            float32x4_t x_vec = vld1q_f32(&x_norm[m * d_sub + j]);
            float32x4_t c_vec = vld1q_f32(&codebooks[m][k * d_sub + j]);
            sum_vec = vmlaq_f32(sum_vec, x_vec, c_vec);
          }

          // 合并部分和
          float32x2_t sum2 =
              vadd_f32(vget_low_f32(sum_vec), vget_high_f32(sum_vec));
          float32x2_t sum1 = vpadd_f32(sum2, sum2);
          dot = vget_lane_f32(sum1, 0);

          // 处理剩余元素
          for (; j < d_sub; j++) {
            dot += x_norm[m * d_sub + j] * codebooks[m][k * d_sub + j];
          }
        } else {
          // 标量计算
          for (size_t j = 0; j < d_sub; j++) {
            dot += x_norm[m * d_sub + j] * codebooks[m][k * d_sub + j];
          }
        }

        if (dot > max_dot) {
          max_dot = dot;
          best_idx = k;
        }
      }

      // 存储最佳匹配的聚类中心索引
      code[m] = best_idx;
    }
  }

  // 批量编码向量
  void encode_dataset(const float *dataset, size_t n, uint8_t *codes) const {
    #pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
      encode(dataset + i * dim, codes + i * M);
    }
  }

  // 计算查询向量与所有子空间的聚类中心的距离表 - 内积距离
  void compute_distance_tables(const float *query) const {
    // 标准化查询向量
    std::vector<float> q_norm(dim);
    float norm = 0.0f;
    for (size_t j = 0; j < dim; j++) {
      float val = query[j];
      norm += val * val;
    }
    norm = std::sqrt(norm) + 1e-8f;

    for (size_t j = 0; j < dim; j++) {
      q_norm[j] = query[j] / norm;
    }

    for (size_t m = 0; m < M; m++) {
      for (size_t k = 0; k < K; k++) {
        float dot = 0.0f;

        // 使用SIMD优化内积计算
        if (d_sub >= 4) {
          size_t j = 0;
          float32x4_t sum_vec = vdupq_n_f32(0);

          for (; j + 3 < d_sub; j += 4) {
            float32x4_t q_vec = vld1q_f32(&q_norm[m * d_sub + j]);
            float32x4_t c_vec = vld1q_f32(&codebooks[m][k * d_sub + j]);
            sum_vec = vmlaq_f32(sum_vec, q_vec, c_vec);
          }

          // 合并部分和
          float32x2_t sum2 =
              vadd_f32(vget_low_f32(sum_vec), vget_high_f32(sum_vec));
          float32x2_t sum1 = vpadd_f32(sum2, sum2);
          dot = vget_lane_f32(sum1, 0);

          // 处理剩余元素
          for (; j < d_sub; j++) {
            dot += q_norm[m * d_sub + j] * codebooks[m][k * d_sub + j];
          }
        } else {
          // 标量计算
          for (size_t j = 0; j < d_sub; j++) {
            dot += q_norm[m * d_sub + j] * codebooks[m][k * d_sub + j];
          }
        }

        // 存储1-内积作为距离，与原始flat_scan.h一致
        const_cast<float &>(distance_tables[m][k]) = 1.0f - dot;
      }
    }
  }

  // 使用预计算的距离表计算查询向量与PQ编码向量之间的近似距离
  float compute_distance(const uint8_t *code) const {
    float dist = 0;
    for (size_t m = 0; m < M; m++) {
      dist += distance_tables[m][code[m]];
    }
    return dist;
  }

  // 保存量化器到文件
  void save(const std::string &filename) const {
    std::ofstream fout(filename, std::ios::binary);
    if (!fout) {
      throw std::runtime_error("无法打开文件进行写入");
    }

    // 写入维度和参数
    fout.write(reinterpret_cast<const char *>(&dim), sizeof(dim));
    fout.write(reinterpret_cast<const char *>(&M), sizeof(M));
    fout.write(reinterpret_cast<const char *>(&K), sizeof(K));

    // 写入编码本
    for (size_t m = 0; m < M; m++) {
      fout.write(reinterpret_cast<const char *>(codebooks[m].data()),
                 codebooks[m].size() * sizeof(float));
    }

    fout.close();
  }

  // 从文件加载量化器
  void load(const std::string &filename) {
    std::ifstream fin(filename, std::ios::binary);
    if (!fin) {
      throw std::runtime_error("无法打开文件进行读取");
    }

    // 读取维度和参数
    fin.read(reinterpret_cast<char *>(&dim), sizeof(dim));
    fin.read(reinterpret_cast<char *>(&M), sizeof(M));
    fin.read(reinterpret_cast<char *>(&K), sizeof(K));

    // 计算子向量维度
    d_sub = dim / M;

    // 调整编码本和距离表大小
    codebooks.resize(M);
    for (size_t m = 0; m < M; m++) {
      codebooks[m].resize(K * d_sub);
    }

    distance_tables.resize(M);
    for (size_t m = 0; m < M; m++) {
      distance_tables[m].resize(K);
    }

    // 读取编码本
    for (size_t m = 0; m < M; m++) {
      fin.read(reinterpret_cast<char *>(codebooks[m].data()),
               codebooks[m].size() * sizeof(float));
    }

    fin.close();
  }

private:
  // 优化版K-means聚类，专为内积距离优化
  void optimized_kmeans_ip(const float *data, size_t n, size_t d, size_t k,
                 float *centroids, std::mt19937 &rng) {
    // 更智能的K-means++初始化
    // 选择第一个中心点：随机从数据集中选择
    std::uniform_int_distribution<size_t> dist(0, n - 1);
    size_t first_center = dist(rng);
    for (size_t j = 0; j < d; j++) {
      centroids[0 * d + j] = data[first_center * d + j];
    }
    
    // 标准化第一个中心点
    float norm = 0.0f;
    for (size_t j = 0; j < d; j++) {
      float val = centroids[0 * d + j];
      norm += val * val;
    }
    norm = std::sqrt(norm) + 1e-8f;
    for (size_t j = 0; j < d; j++) {
      centroids[0 * d + j] /= norm;
    }
    
    // 继续K-means++算法，选择剩余中心点
    std::vector<float> min_distances(n, std::numeric_limits<float>::max());
    std::vector<float> distances(n);
    
    // 选择剩余的k-1个中心点
    for (size_t c = 1; c < k; c++) {
      // 计算每个点到最近中心的距离
      float total_distance = 0.0f;
      
      #pragma omp parallel for reduction(+:total_distance)
      for (size_t i = 0; i < n; i++) {
        // 计算点i到当前中心c-1的内积
        float dot = 0.0f;
        for (size_t j = 0; j < d; j++) {
          dot += data[i * d + j] * centroids[(c-1) * d + j];
        }
        
        // 转换为距离: 1 - 内积
        float distance = 1.0f - dot;
        
        // 更新到最近中心的距离
        if (distance < min_distances[i]) {
          min_distances[i] = distance;
        }
        
        distances[i] = min_distances[i];
        total_distance += distances[i];
      }
      
      // 基于距离权重选择下一个中心点
      std::uniform_real_distribution<float> distribution(0, total_distance);
      float threshold = distribution(rng);
      
      // 寻找累积距离超过阈值的点
      float cumulative_distance = 0.0f;
      size_t next_center = 0;
      for (size_t i = 0; i < n; i++) {
        cumulative_distance += distances[i];
        if (cumulative_distance >= threshold) {
          next_center = i;
          break;
        }
      }
      
      // 复制选中的数据点作为新的中心点
      for (size_t j = 0; j < d; j++) {
        centroids[c * d + j] = data[next_center * d + j];
      }
      
      // 标准化新的中心点
      float c_norm = 0.0f;
      for (size_t j = 0; j < d; j++) {
        float val = centroids[c * d + j];
        c_norm += val * val;
      }
      c_norm = std::sqrt(c_norm) + 1e-8f;
      for (size_t j = 0; j < d; j++) {
        centroids[c * d + j] /= c_norm;
      }
    }

    // 分配数组
    std::vector<size_t> assignments(n);
    std::vector<size_t> counts(k);
    std::vector<float> new_centroids(k * d);

    // 最大迭代次数
    const size_t max_iter = 30; // 增加迭代次数以提高精度
    const float threshold = 1e-5f; // 严格的收敛阈值

    // 使用优化的K-means主循环
    for (size_t iter = 0; iter < max_iter; iter++) {
      // 清零
      std::fill(counts.begin(), counts.end(), 0);
      std::fill(new_centroids.begin(), new_centroids.end(), 0);

      // 分配步骤 - 使用并行处理提高速度
      #pragma omp parallel
      {
        // 每个线程的本地计数和累积数组
        std::vector<size_t> local_counts(k, 0);
        std::vector<float> local_new_centroids(k * d, 0);

        #pragma omp for
        for (size_t i = 0; i < n; i++) {
          float max_dot = -std::numeric_limits<float>::max();
          size_t best_centroid = 0;

          // 找到内积最大的聚类中心
          for (size_t j = 0; j < k; j++) {
            float dot = 0;
            
            // 使用SIMD优化计算内积
            if (d >= 4) {
              size_t l = 0;
              float32x4_t sum_vec = vdupq_n_f32(0);
              
              for (; l + 3 < d; l += 4) {
                float32x4_t d_vec = vld1q_f32(&data[i * d + l]);
                float32x4_t c_vec = vld1q_f32(&centroids[j * d + l]);
                sum_vec = vmlaq_f32(sum_vec, d_vec, c_vec);
              }
              
              // 合并部分和
              float32x2_t sum2 = vadd_f32(vget_low_f32(sum_vec), vget_high_f32(sum_vec));
              float32x2_t sum1 = vpadd_f32(sum2, sum2);
              dot = vget_lane_f32(sum1, 0);
              
              // 处理剩余元素
              for (; l < d; l++) {
                dot += data[i * d + l] * centroids[j * d + l];
              }
            } else {
              // 标量计算
              for (size_t l = 0; l < d; l++) {
                dot += data[i * d + l] * centroids[j * d + l];
              }
            }

            if (dot > max_dot) {
              max_dot = dot;
              best_centroid = j;
            }
          }

          // 分配数据点
          assignments[i] = best_centroid;
          local_counts[best_centroid]++;

          // 累加数据点到新中心
          for (size_t d_idx = 0; d_idx < d; d_idx++) {
            local_new_centroids[best_centroid * d + d_idx] += data[i * d + d_idx];
          }
        }

        // 合并线程本地结果
        #pragma omp critical
        {
          for (size_t j = 0; j < k; j++) {
            counts[j] += local_counts[j];
            for (size_t d_idx = 0; d_idx < d; d_idx++) {
              new_centroids[j * d + d_idx] += local_new_centroids[j * d + d_idx];
            }
          }
        }
      }

      // 更新中心点并检查收敛
      bool changed = false;
      
      for (size_t j = 0; j < k; j++) {
        if (counts[j] > 0) {
          // 首先计算新中心
          for (size_t d_idx = 0; d_idx < d; d_idx++) {
            new_centroids[j * d + d_idx] /= counts[j];
          }
          
          // 标准化中心点 - 对内积距离很重要
          float norm = 0.0f;
          for (size_t d_idx = 0; d_idx < d; d_idx++) {
            float val = new_centroids[j * d + d_idx];
            norm += val * val;
          }
          norm = std::sqrt(norm) + 1e-8f;
          
          // 计算变化量并更新中心点
          float max_change = 0.0f;
          for (size_t d_idx = 0; d_idx < d; d_idx++) {
            float old_val = centroids[j * d + d_idx];
            float new_val = new_centroids[j * d + d_idx] / norm;
            float change = std::abs(new_val - old_val);
            max_change = std::max(max_change, change);
            centroids[j * d + d_idx] = new_val;
          }
          
          if (max_change > threshold) {
            changed = true;
          }
        }
      }

      // 如果聚类中心稳定，提前退出
      if (!changed) {
        break;
      }
    }

    // 处理空聚类
    for (size_t j = 0; j < k; j++) {
      if (counts[j] == 0) {
        // 找到最大数据点数的聚类
        size_t max_count_idx = 0;
        for (size_t i = 1; i < k; i++) {
          if (counts[i] > counts[max_count_idx]) {
            max_count_idx = i;
          }
        }

        // 分割最大聚类的数据点
        if (counts[max_count_idx] > 1) {
          counts[j] = counts[max_count_idx] / 2;
          counts[max_count_idx] -= counts[j];

          // 复制中心并添加一些随机扰动
          for (size_t d_idx = 0; d_idx < d; d_idx++) {
            std::normal_distribution<float> noise_dist(0.0f, 0.01f); // 使用高斯噪声
            float noise = noise_dist(rng);
            centroids[j * d + d_idx] = centroids[max_count_idx * d + d_idx] + noise;
          }
          
          // 标准化处理后的中心
          float norm = 0.0f;
          for (size_t d_idx = 0; d_idx < d; d_idx++) {
            float val = centroids[j * d + d_idx];
            norm += val * val;
          }
          norm = std::sqrt(norm) + 1e-8f;
          
          for (size_t d_idx = 0; d_idx < d; d_idx++) {
            centroids[j * d + d_idx] /= norm;
          }
        }
      }
    }
  }

  size_t dim;   // 向量维度
  size_t M;     // 子空间数量
  size_t K;     // 每个子空间的聚类中心数量
  size_t d_sub; // 每个子向量的维度

  // 编码本 [M][K * d_sub]
  std::vector<std::vector<float>> codebooks;

  // 距离表 [M][K]
  mutable std::vector<std::vector<float>> distance_tables;
};

// 直接使用内积距离进行搜索的基础函数 (与原flat_scan.h类似)
std::priority_queue<std::pair<float, uint32_t>> inner_product_search(
    const float* base, const float* query, size_t base_number, size_t vecdim, size_t k) {
    
    std::priority_queue<std::pair<float, uint32_t>> q;
    
    // 标准化查询向量 - 只需要做一次
    std::vector<float> normalized_query(vecdim);
    float norm = 0.0f;
    for (size_t j = 0; j < vecdim; j++) {
        float val = query[j];
        norm += val * val;
    }
    norm = std::sqrt(norm) + 1e-8f;
    
    for (size_t j = 0; j < vecdim; j++) {
        normalized_query[j] = query[j] / norm;
    }
    
    // 批量计算内积以提高效率
    #pragma omp parallel
    {
        // 每个线程使用本地队列，减少线程冲突
        std::priority_queue<std::pair<float, uint32_t>> local_q;
        
        #pragma omp for
        for (size_t i = 0; i < base_number; ++i) {
            // 标准化基础向量 - 只计算向量长度，不复制向量
            float base_norm = 0.0f;
            for (size_t j = 0; j < vecdim; j++) {
                float val = base[i * vecdim + j];
                base_norm += val * val;
            }
            base_norm = std::sqrt(base_norm) + 1e-8f;
            
            // 直接计算标准化后的内积，避免创建临时向量
            float dot = 0.0f;
            size_t j = 0;
            
            // 使用NEON SIMD加速内积计算
            float32x4_t sum_vec = vdupq_n_f32(0);
            float32x4_t norm_scale = vdupq_n_f32(1.0f/base_norm);
            
            for (; j + 3 < vecdim; j += 4) {
                float32x4_t q_vec = vld1q_f32(&normalized_query[j]);
                float32x4_t b_vec = vld1q_f32(&base[i * vecdim + j]);
                // 应用标准化因子
                b_vec = vmulq_f32(b_vec, norm_scale);
                sum_vec = vmlaq_f32(sum_vec, q_vec, b_vec);
            }
            
            // 合并部分和
            float32x2_t sum2 = vadd_f32(vget_low_f32(sum_vec), vget_high_f32(sum_vec));
            float32x2_t sum1 = vpadd_f32(sum2, sum2);
            dot = vget_lane_f32(sum1, 0);
            
            // 处理剩余元素
            for (; j < vecdim; ++j) {
                dot += normalized_query[j] * (base[i * vecdim + j] / base_norm);
            }
            
            // 计算内积距离
            float dis = 1.0f - dot;
            
            // 维护top-k
            if (local_q.size() < k) {
                local_q.push({dis, static_cast<uint32_t>(i)});
            } else if (dis < local_q.top().first) {
                local_q.pop();
                local_q.push({dis, static_cast<uint32_t>(i)});
            }
        }
        
        // 合并线程结果
        #pragma omp critical
        {
            while (!local_q.empty()) {
                if (q.size() < k) {
                    q.push(local_q.top());
                } else if (local_q.top().first < q.top().first) {
                    q.pop();
                    q.push(local_q.top());
                }
                local_q.pop();
            }
        }
    }
    
    return q;
}

// PQ索引类，优化内积距离搜索
class PQIndex {
public:
  PQIndex(size_t d, size_t M = 4, size_t K = 64) : dim(d), pq(d, M, K), M(M) {}

  // 构建索引
  void build(const float *data, size_t n) {
    std::cout << "开始构建优化的内积距离PQ索引，数据点数: " << n << ", 维度: " << dim
              << ", 子空间数: " << M << std::endl;

    // 保存原始数据向量的数量
    n_data = n;

    // 标准化原始数据 - 对内积搜索很重要
    normalized_data.resize(n * dim);
    
    #pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        float norm = 0.0f;
        for (size_t j = 0; j < dim; j++) {
            float val = data[i * dim + j];
            norm += val * val;
        }
        norm = std::sqrt(norm) + 1e-8f;
        
        for (size_t j = 0; j < dim; j++) {
            normalized_data[i * dim + j] = data[i * dim + j] / norm;
        }
    }

    // 训练PQ量化器
    std::cout << "训练PQ量化器..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    pq.train(normalized_data, n);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "训练完成，耗时: " << duration.count() << " 毫秒" << std::endl;

    // 编码所有数据
    codes.resize(n * M);
    std::cout << "编码数据..." << std::endl;
    start = std::chrono::high_resolution_clock::now();
    pq.encode_dataset(normalized_data.data(), n, codes.data());
    end = std::chrono::high_resolution_clock::now();
    duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "编码完成，耗时: " << duration.count() << " 毫秒" << std::endl;

    // 保存原始数据指针
    original_data_ptr = data;

    std::cout << "PQ索引构建完成" << std::endl;
  }

  // 搜索最近邻 - 改进的混合检索策略，高效实现
  std::priority_queue<std::pair<float, uint32_t>> search(const float *query,
                                                         size_t k) const {
    // 对于小数据集，使用精确搜索更快更准确
    if (n_data <= 10) { // 降低精确搜索的阈值
        return inner_product_search(original_data_ptr, query, n_data, dim, k);
    }
    
    // 标准化查询向量 - 只需要做一次
    std::vector<float> normalized_query(dim);
    float norm = 0.0f;
    for (size_t j = 0; j < dim; j++) {
        float val = query[j];
        norm += val * val;
    }
    norm = std::sqrt(norm) + 1e-8f;
    
    for (size_t j = 0; j < dim; j++) {
        normalized_query[j] = query[j] / norm;
    }
    
    // 对于大数据集，使用PQ近似搜索基础结果
    std::priority_queue<std::pair<float, uint32_t>> pq_results;
    
    // 计算距离表
    pq.compute_distance_tables(normalized_query.data());  // 直接传递标准化的查询向量

    // 使用批处理策略扫描所有数据
    const size_t batch_size = 4096; // 批处理大小，提高缓存局部性
    
    for (size_t b = 0; b < n_data; b += batch_size) {
        size_t end = std::min(b + batch_size, n_data);
        
        for (size_t i = b; i < end; i++) {
            float dist = pq.compute_distance(codes.data() + i * M);
            
            if (pq_results.size() < k * 20) { 
                pq_results.push({dist, static_cast<uint32_t>(i)});
            } else if (dist < pq_results.top().first) {
                pq_results.pop();
                pq_results.push({dist, static_cast<uint32_t>(i)});
            }
        }
    }
    
    // 获取PQ近似搜索的候选结果
    std::vector<uint32_t> candidates;
    candidates.reserve(pq_results.size()); // 预分配内存，避免动态调整
    while (!pq_results.empty()) {
        candidates.push_back(pq_results.top().second);
        pq_results.pop();
    }
    
    // 对候选进行精确重排序 - 使用已标准化的数据
    std::priority_queue<std::pair<float, uint32_t>> final_results;
    
    // 对候选进行精确内积计算 - 使用预标准化的数据
    for (uint32_t idx : candidates) {
        float dot = 0.0f;
        size_t j = 0;
        
        // 直接使用预先标准化的数据
        float32x4_t sum_vec = vdupq_n_f32(0);
        for (; j + 3 < dim; j += 4) {
            float32x4_t q_vec = vld1q_f32(&normalized_query[j]);
            float32x4_t b_vec = vld1q_f32(&normalized_data[idx * dim + j]);
            sum_vec = vmlaq_f32(sum_vec, q_vec, b_vec);
        }
        
        // 合并部分和
        float32x2_t sum2 = vadd_f32(vget_low_f32(sum_vec), vget_high_f32(sum_vec));
        float32x2_t sum1 = vpadd_f32(sum2, sum2);
        dot = vget_lane_f32(sum1, 0);
        
        // 处理剩余元素
        for (; j < dim; j++) {
            dot += normalized_query[j] * normalized_data[idx * dim + j];
        }
        
        float dist = 1.0f - dot;
        
        if (final_results.size() < k) {
            final_results.push({dist, idx});
        } else if (dist < final_results.top().first) {
            final_results.pop();
            final_results.push({dist, idx});
        }
    }
    
    return final_results;
  }

  // 保存索引到文件
  void save(const std::string &filename) const {
    std::ofstream fout(filename, std::ios::binary);
    if (!fout) {
      throw std::runtime_error("无法打开文件进行写入");
    }

    // 写入索引参数
    fout.write(reinterpret_cast<const char *>(&dim), sizeof(dim));
    fout.write(reinterpret_cast<const char *>(&M), sizeof(M));
    fout.write(reinterpret_cast<const char *>(&n_data), sizeof(n_data));

    // 写入编码
    fout.write(reinterpret_cast<const char *>(codes.data()),
               codes.size() * sizeof(uint8_t));

    fout.close();

    // 保存PQ量化器
    pq.save(filename + ".pq");

    std::cout << "索引已保存到 " << filename << std::endl;
  }

  // 从文件加载索引
  void load(const std::string &filename) {
    std::ifstream fin(filename, std::ios::binary);
    if (!fin) {
      throw std::runtime_error("无法打开文件进行读取");
    }

    // 读取索引参数
    fin.read(reinterpret_cast<char *>(&dim), sizeof(dim));
    fin.read(reinterpret_cast<char *>(&M), sizeof(M));
    fin.read(reinterpret_cast<char *>(&n_data), sizeof(n_data));

    // 调整编码数组大小
    codes.resize(n_data * M);

    // 读取编码
    fin.read(reinterpret_cast<char *>(codes.data()),
             codes.size() * sizeof(uint8_t));

    fin.close();

    // 加载PQ量化器
    pq.load(filename + ".pq");

    std::cout << "索引已从 " << filename << " 加载" << std::endl;
  }

private:
  size_t dim;                       // 向量维度
  size_t M;                         // 子空间数量
  size_t n_data;                    // 数据点数量
  ProductQuantizer pq;              // 量化器
  std::vector<uint8_t> codes;       // 编码数据
  std::vector<float> normalized_data; // 标准化数据
  const float* original_data_ptr;   // 原始数据指针(不复制数据)
};

// 全局变量用于缓存PQ索引和避免重复构建
namespace Inner_Product_pq_namespace {
  // 全局PQ索引指针
  PQIndex* global_pq_index = nullptr;
  // 缓存原始数据指针和大小，用于验证是否需要重建索引
  float* g_base_data = nullptr;
  size_t g_base_number = 0;
  size_t g_vecdim = 0;
  bool g_initialized = false;
}

// 初始化PQ索引函数 - 只在需要时构建索引
void init_pq_index(float* base, size_t base_number, size_t vecdim) {
  // 如果已经为相同数据初始化过，直接返回
  if (Inner_Product_pq_namespace::g_initialized && 
      Inner_Product_pq_namespace::g_base_data == base && 
      Inner_Product_pq_namespace::g_base_number == base_number && 
      Inner_Product_pq_namespace::g_vecdim == vecdim) {
    return;
  }
  
  // 清理之前的索引（如果有）
  if (Inner_Product_pq_namespace::global_pq_index) {
    delete Inner_Product_pq_namespace::global_pq_index;
  }
  
  // 设置合理的PQ参数 - 对于内积距离
  size_t M = 16;   // 增加子空间数量到16以提高精度
  size_t K = 64;  // 每个子空间的聚类中心数量
  
  std::cout << "创建新的内积距离优化PQ索引，M=" << M << ", K=" << K << std::endl;
  
  // 创建并构建新索引
  Inner_Product_pq_namespace::global_pq_index = new PQIndex(vecdim, M, K);
  Inner_Product_pq_namespace::global_pq_index->build(base, base_number);
  
  // 更新缓存数据
  Inner_Product_pq_namespace::g_base_data = base;
  Inner_Product_pq_namespace::g_base_number = base_number;
  Inner_Product_pq_namespace::g_vecdim = vecdim;
  Inner_Product_pq_namespace::g_initialized = true;
  
  // 保存索引以便后续使用
  Inner_Product_pq_namespace::global_pq_index->save("files/pq_Inner_Product.index");
}

// 清理索引资源
void cleanup_pq_index() {
  if (Inner_Product_pq_namespace::global_pq_index) {
    delete Inner_Product_pq_namespace::global_pq_index;
    Inner_Product_pq_namespace::global_pq_index = nullptr;
    Inner_Product_pq_namespace::g_initialized = false;
  }
}

// 使用PQ在基础向量集上搜索查询向量的k个最近邻 - 优化版本
std::priority_queue<std::pair<float, uint32_t>>
flat_search_with_pq_Inner_Product(float *base, float *query, size_t base_number,
                    size_t vecdim, size_t k) {

  // 检查数据有效性
  assert(base != nullptr);
  assert(query != nullptr);
  assert(base_number > 0);
  assert(vecdim > 0);
  assert(k > 0);

  // 初始化/确保PQ索引已构建
  init_pq_index(base, base_number, vecdim);

  // 使用PQ索引搜索k个最近邻
  return Inner_Product_pq_namespace::global_pq_index->search(query, k);
}

// 析构函数释放全局内存 - 在程序结束时调用
class GlobalPQCleanup {
public:
  ~GlobalPQCleanup() {
    cleanup_pq_index();
  }
};

static GlobalPQCleanup g_pq_cleanup;