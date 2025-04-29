#include <arm_neon.h>
#include <vector>
#include <queue>
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <omp.h>

struct PQIndex {
    size_t m;           // 子空间数量
    size_t ks;          // 每个子空间的聚类中心数
    size_t dsub;        // 子空间维度
    std::vector<float> centroids;  // 聚类中心 [m][ks][dsub]
    std::vector<uint8_t> codes;    // 编码数据 [base_number][m]
};



// 计算两个向量之间的L2距离平方（NEON优化）
inline float compute_distance_sq(const float* vec1, const float* vec2, size_t dim) {
    float32x4_t sum = vdupq_n_f32(0.0f);
    size_t d = 0;
    
    for (; d + 4 <= dim; d += 4) {
        float32x4_t v1 = vld1q_f32(vec1 + d);
        float32x4_t v2 = vld1q_f32(vec2 + d);
        float32x4_t diff = vsubq_f32(v1, v2);
        sum = vmlaq_f32(sum, diff, diff);
    }
    
    float distance = vaddvq_f32(sum);
    
    // 处理剩余维度
    for (; d < dim; ++d) {
        float diff = vec1[d] - vec2[d];
        distance += diff * diff;
    }
    
    return distance;
}

// k-means++初始化聚类中心
void kmeans_pp_init(const float* data, size_t n, size_t dim, 
                   float* centroids, size_t k, size_t trials=3) {
    std::vector<float> min_distances(n, std::numeric_limits<float>::max());
    std::vector<size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // 随机选择第一个中心
    std::uniform_int_distribution<size_t> dis(0, n-1);
    size_t first_idx = dis(gen);
    std::memcpy(centroids, data + first_idx * dim, dim * sizeof(float));
    
    // 选择后续中心
    for (size_t ki = 1; ki < k; ++ki) {
        // 计算每个点到最近中心的距离
        #pragma omp parallel for
        for (size_t i = 0; i < n; ++i) {
            float dist = compute_distance_sq(data + i * dim, 
                                           centroids + (ki-1) * dim, dim);
            if (dist < min_distances[i]) {
                min_distances[i] = dist;
            }
        }
        
        // 使用轮盘赌选择下一个中心
        float sum_dist = 0.0f;
        for (size_t i = 0; i < n; ++i) {
            sum_dist += min_distances[i];
        }
        
        std::uniform_real_distribution<float> distr(0.0f, sum_dist);
        float threshold = distr(gen);
        
        float running_sum = 0.0f;
        size_t selected_idx = 0;
        for (; selected_idx < n; ++selected_idx) {
            running_sum += min_distances[selected_idx];
            if (running_sum >= threshold) break;
        }
        
        // 复制选中的样本到中心
        std::memcpy(centroids + ki * dim, data + selected_idx * dim, 
                   dim * sizeof(float));
    }
}
void kmeans(const float* data, size_t n, size_t dim, 
           float* centroids, size_t k, size_t max_iter=20, float tol=1e-6) {
    std::vector<size_t> assignments(n);
    std::vector<float> old_centroids(k * dim);
    std::vector<size_t> counts(k);
    
    // k-means++初始化
    kmeans_pp_init(data, n, dim, centroids, k);
    
    for (size_t iter = 0; iter < max_iter; ++iter) {
        // 保存旧中心用于收敛判断
        std::memcpy(old_centroids.data(), centroids, k * dim * sizeof(float));
        
        // 分配阶段：为每个点找到最近的中心
        #pragma omp parallel for
        for (size_t i = 0; i < n; ++i) {
            float min_dist = std::numeric_limits<float>::max();
            size_t best_cluster = 0;
            
            for (size_t ki = 0; ki < k; ++ki) {
                float dist = compute_distance_sq(data + i * dim, 
                                               centroids + ki * dim, dim);
                if (dist < min_dist) {
                    min_dist = dist;
                    best_cluster = ki;
                }
            }
            assignments[i] = best_cluster;
        }
        
        // 3. 更新阶段：重新计算中心位置
        std::memset(centroids, 0, k * dim * sizeof(float));
        std::memset(counts.data(), 0, k * sizeof(size_t));
        
        #pragma omp parallel for
        for (size_t i = 0; i < n; ++i) {
            size_t cluster = assignments[i];
            #pragma omp atomic
            counts[cluster]++;
            
            // 累加向量到对应的中心
            for (size_t d = 0; d < dim; ++d) {
                #pragma omp atomic
                centroids[cluster * dim + d] += data[i * dim + d];
            }
        }
        
        // 计算新的中心位置（平均值）
        for (size_t ki = 0; ki < k; ++ki) {
            if (counts[ki] > 0) {
                for (size_t d = 0; d < dim; ++d) {
                    centroids[ki * dim + d] /= counts[ki];
                }
            }
        }
        
        // 4. 检查收敛条件
        float max_shift = 0.0f;
        for (size_t ki = 0; ki < k; ++ki) {
            float shift = compute_distance_sq(old_centroids.data() + ki * dim,
                                            centroids + ki * dim, dim);
            if (shift > max_shift) {
                max_shift = shift;
            }
        }
        
        if (max_shift < tol) {
            break;  // 中心移动很小，提前终止
        }
    }
}

void train_pq(const float* base, size_t base_number, size_t vecdim, 
             PQIndex& index, size_t m, size_t ks) {
    index.m = m;
    index.ks = ks;
    index.dsub = vecdim / m;
    index.centroids.resize(m * ks * index.dsub);
    
    #pragma omp parallel for
    for (size_t mi = 0; mi < m; ++mi) {
        // 提取当前子空间的所有数据
        std::vector<float> subspace_data(base_number * index.dsub);
        for (size_t i = 0; i < base_number; ++i) {
            for (size_t d = 0; d < index.dsub; ++d) {
                subspace_data[i * index.dsub + d] = 
                    base[i * vecdim + mi * index.dsub + d];
            }
        }
        
        // 对当前子空间运行k-means
        kmeans(subspace_data.data(), base_number, index.dsub,
              &index.centroids[mi * ks * index.dsub], ks);
    }
}

// 编码数据
void encode_pq(const float* base, size_t base_number, const PQIndex& index, 
              std::vector<uint8_t>& codes) {
    codes.resize(base_number * index.m);
    
    #pragma omp parallel for
    for (size_t i = 0; i < base_number; ++i) {
        for (size_t mi = 0; mi < index.m; ++mi) {
            float min_dist = INFINITY;
            uint8_t best_code = 0;
            
            for (size_t ki = 0; ki < index.ks; ++ki) {
                float dist = 0;
                
                for (size_t d = 0; d < index.dsub; ++d) {
                    float diff = base[i * index.m * index.dsub + mi * index.dsub + d] - 
                               index.centroids[(mi * index.ks + ki) * index.dsub + d];
                    dist += diff * diff;
                }
                
                if (dist < min_dist) {
                    min_dist = dist;
                    best_code = ki;
                }
            }
            
            codes[i * index.m + mi] = best_code;
        }
    }
}

// 预计算PQ查询表
void precompute_pq_table(const float* query, const PQIndex& index, 
                        std::vector<float>& query_table) {
    query_table.resize(index.m * index.ks);
    
    #pragma omp parallel for
    for (size_t mi = 0; mi < index.m; ++mi) {
        const float* query_sub = &query[mi * index.dsub];
        
        for (size_t ki = 0; ki < index.ks; ++ki) {
            const float* centroid = &index.centroids[(mi * index.ks + ki) * index.dsub];
            
            float32x4_t sum = vdupq_n_f32(0.0f);
            size_t d = 0;
            
            for (; d + 4 <= index.dsub; d += 4) {
                float32x4_t q = vld1q_f32(&query_sub[d]);
                float32x4_t c = vld1q_f32(&centroid[d]);
                sum = vmlaq_f32(sum, q, c);
            }
            
            float ip = vaddvq_f32(sum);
            
            for (; d < index.dsub; ++d) {
                ip += query_sub[d] * centroid[d];
            }
            
            query_table[mi * index.ks + ki] = ip;
        }
    }
}

// PQ搜索函数
std::priority_queue<std::pair<float, uint32_t>> pq_search(
    const PQIndex& index, const std::vector<float>& query_table,
    size_t base_number, size_t k) {
    
    std::priority_queue<std::pair<float, uint32_t>> q;
    
    #pragma omp parallel for
    for (size_t i = 0; i < base_number; ++i) {
        float32x4_t sum = vdupq_n_f32(0.0f);
        size_t mi = 0;
        
        for (; mi + 4 <= index.m; mi += 4) {
            uint8x8_t codes8 = vld1_u8(&index.codes[i * index.m + mi]);
            uint16x4_t codes16 = vget_low_u16(vmovl_u8(codes8));
            uint32x4_t codes32 = vmovl_u16(codes16);
            
            float ip0 = query_table[(mi+0) * index.ks + vgetq_lane_u32(codes32, 0)];
            float ip1 = query_table[(mi+1) * index.ks + vgetq_lane_u32(codes32, 1)];
            float ip2 = query_table[(mi+2) * index.ks + vgetq_lane_u32(codes32, 2)];
            float ip3 = query_table[(mi+3) * index.ks + vgetq_lane_u32(codes32, 3)];
            
            float32x4_t combined = vsetq_lane_f32(ip0, sum, 0);
            combined = vsetq_lane_f32(ip1, combined, 1);
            combined = vsetq_lane_f32(ip2, combined, 2);
            combined = vsetq_lane_f32(ip3, combined, 3);
            
            sum = vaddq_f32(sum, combined);
        }
        
        float dis = vaddvq_f32(sum);
        
        for (; mi < index.m; ++mi) {
            uint8_t code = index.codes[i * index.m + mi];
            dis += query_table[mi * index.ks + code];
        }
        
        dis = 1 - dis;
        
        #pragma omp critical
        {
            if (q.size() < k) {
                q.push({dis, static_cast<uint32_t>(i)});
            } else if (dis < q.top().first) {
                q.push({dis, static_cast<uint32_t>(i)});
                if (q.size() > k) q.pop();
            }
        }
    }
    
    return q;
}