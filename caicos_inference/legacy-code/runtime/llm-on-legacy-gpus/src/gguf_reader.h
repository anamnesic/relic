#pragma once
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>
#include <unordered_map>
#include <cstring>
#include <cassert>

constexpr uint32_t GGUF_MAGIC = 0x46554747;
constexpr uint32_t GGUF_VERSION = 3;
constexpr uint32_t GGUF_DEFAULT_ALIGNMENT = 32;

enum class GgufType : int32_t {
    UINT8   = 0,
    INT8    = 1,
    UINT16  = 2,
    INT16   = 3,
    UINT32  = 4,
    INT32   = 5,
    FLOAT32 = 6,
    BOOL    = 7,
    STRING  = 8,
    ARRAY   = 9,
    UINT64  = 10,
    INT64   = 11,
    FLOAT64 = 12,
};

enum class GgmlType : int32_t {
    F32     = 0,
    F16     = 1,
    Q4_0    = 2,
    Q4_1    = 3,
    Q5_0    = 6,
    Q5_1    = 7,
    Q8_0    = 8,
    Q8_1    = 9,
    Q2_K    = 10,
    Q3_K    = 11,
    Q4_K    = 12,
    Q5_K    = 13,
    Q6_K    = 14,
    Q8_K    = 15,
    IQ2_XXS = 16,
    IQ2_XS  = 17,
    IQ3_XXS = 18,
    IQ1_S   = 19,
    IQ4_NL  = 20,
    IQ3_S   = 21,
    IQ2_S   = 22,
    IQ4_XS  = 23,
    I8      = 24,
    I16     = 25,
    I32     = 26,
    I64     = 27,
    F64     = 28,
    IQ1_M   = 29,
    BF16    = 30,
    TQ1_0   = 34,
    TQ2_0   = 35,
    MXFP4   = 39,
    NVFP4   = 40,
    Q1_0    = 41,
};

inline int ggml_blck_size(GgmlType t) {
    switch (t) {
        case GgmlType::F32: case GgmlType::F16: case GgmlType::BF16:
        case GgmlType::I8: case GgmlType::I16: case GgmlType::I32: case GgmlType::I64: case GgmlType::F64:
            return 1;
        case GgmlType::Q4_0: case GgmlType::Q4_1: case GgmlType::Q5_0: case GgmlType::Q5_1:
        case GgmlType::Q8_0: case GgmlType::Q8_1: case GgmlType::MXFP4: case GgmlType::Q1_0:
            return 32;
        case GgmlType::Q2_K: return 256;
        case GgmlType::Q3_K: return 256;
        case GgmlType::Q4_K: return 256;
        case GgmlType::Q5_K: return 256;
        case GgmlType::Q6_K: return 256;
        case GgmlType::Q8_K: return 256;
        case GgmlType::IQ2_XXS: case GgmlType::IQ2_XS: return 256;
        case GgmlType::IQ3_XXS: return 256;
        case GgmlType::IQ1_S: return 256;
        case GgmlType::IQ4_NL: return 32;
        case GgmlType::IQ3_S: return 256;
        case GgmlType::IQ2_S: return 256;
        case GgmlType::IQ4_XS: return 256;
        case GgmlType::IQ1_M: return 256;
        case GgmlType::TQ1_0: return 256;
        case GgmlType::TQ2_0: return 256;
        case GgmlType::NVFP4: return 64;
        default: return 1;
    }
}

inline int ggml_type_size(GgmlType t) {
    switch (t) {
        case GgmlType::F32:  return 4;
        case GgmlType::F16:  return 2;
        case GgmlType::BF16: return 2;
        case GgmlType::I8:   return 1;
        case GgmlType::I16:  return 2;
        case GgmlType::I32:  return 4;
        case GgmlType::I64:  return 8;
        case GgmlType::F64:  return 8;
        case GgmlType::Q4_0: return 2 + 16;
        case GgmlType::Q4_1: return 2 + 2 + 16;
        case GgmlType::Q5_0: return 2 + 4 + 16;
        case GgmlType::Q5_1: return 2 + 2 + 4 + 16;
        case GgmlType::Q8_0: return 2 + 32;
        case GgmlType::Q8_1: return 2 + 2 + 32;
        case GgmlType::Q2_K: return 256/16 + 256/4 + 2 + 2;
        case GgmlType::Q3_K: return 256/8 + 256/4 + 2 + 2;
        case GgmlType::Q4_K: return 2 + 2 + 12 + 256/2;
        case GgmlType::Q5_K: return 2 + 2 + 12 + 256/8 + 256/2;
        case GgmlType::Q6_K: return 256/4 + 256/16 + 256/8 + 2;
        case GgmlType::Q8_K: return 4 + 256 + 4 + 2;
        case GgmlType::IQ2_XXS: return 2 + 4 + 256/8 + 256/4;
        case GgmlType::IQ2_XS:  return 2 + 4 + 256/8 + 256/4;
        case GgmlType::IQ3_XXS: return 2 + 4 + 256/4 + 256/8;
        case GgmlType::IQ1_S:   return 2 + 4 + 256/8 + 256/8;
        case GgmlType::IQ4_NL:  return 2 + 32/2;
        case GgmlType::IQ3_S:   return 2 + 4 + 256/4 + 256/8;
        case GgmlType::IQ2_S:   return 2 + 4 + 256/8 + 256/4;
        case GgmlType::IQ4_XS:  return 2 + 4 + 256/2 + 256/4;
        case GgmlType::IQ1_M:   return 2 + 4 + 256/8 + 256/8;
        case GgmlType::TQ1_0:   return 4 + 256/8;
        case GgmlType::TQ2_0:   return 4 + 256/4;
        case GgmlType::MXFP4:   return 1 + 32/2;
        case GgmlType::NVFP4:   return 4 + 64/2;
        case GgmlType::Q1_0:    return 2 + 128/8;
        default: return 0;
    }
}

inline bool ggml_is_quantized(GgmlType t) {
    return t >= GgmlType::Q4_0;
}

struct GgufTensorInfo {
    std::string name;
    std::vector<int64_t> dims;
    GgmlType type;
    uint64_t offset;
};

struct GgufReader {
    std::vector<uint8_t> data;
    size_t alignment = GGUF_DEFAULT_ALIGNMENT;
    std::unordered_map<std::string, GgufTensorInfo> tensors;
    std::unordered_map<std::string, std::string> metadata_str;
    std::unordered_map<std::string, int64_t> metadata_int;
    std::unordered_map<std::string, float> metadata_float;
    std::unordered_map<std::string, uint32_t> metadata_uint32;
    size_t tensor_data_offset = 0;

    bool load(const char *filename) {
        FILE *f = fopen(filename, "rb");
        if (!f) return false;
        fseek(f, 0, SEEK_END);
        size_t size = ftell(f);
        fseek(f, 0, SEEK_SET);
        data.resize(size);
        if (fread(data.data(), 1, size, f) != size) {
            fclose(f);
            return false;
        }
        fclose(f);

        size_t pos = 0;
        auto read_u8 = [&]() -> uint8_t { return data[pos++]; };
        auto read_i8 = [&]() -> int8_t { return (int8_t)data[pos++]; };
        auto read_u16 = [&]() -> uint16_t { uint16_t v; memcpy(&v, &data[pos], 2); pos += 2; return v; };
        auto read_i16 = [&]() -> int16_t { int16_t v; memcpy(&v, &data[pos], 2); pos += 2; return v; };
        auto read_u32 = [&]() -> uint32_t { uint32_t v; memcpy(&v, &data[pos], 4); pos += 4; return v; };
        auto read_i32 = [&]() -> int32_t { int32_t v; memcpy(&v, &data[pos], 4); pos += 4; return v; };
        auto read_u64 = [&]() -> uint64_t { uint64_t v; memcpy(&v, &data[pos], 8); pos += 8; return v; };
        auto read_i64 = [&]() -> int64_t { int64_t v; memcpy(&v, &data[pos], 8); pos += 8; return v; };
        auto read_f32 = [&]() -> float { float v; memcpy(&v, &data[pos], 4); pos += 4; return v; };
        auto read_f64 = [&]() -> double { double v; memcpy(&v, &data[pos], 8); pos += 8; return v; };

        auto read_string = [&]() -> std::string {
            uint64_t len = read_u64();
            std::string s((const char*)&data[pos], (size_t)len);
            pos += (size_t)len;
            return s;
        };

        uint32_t magic = read_u32();
        if (magic != GGUF_MAGIC) return false;

        uint32_t version = read_u32();
        (void)version;

        uint64_t tensor_count = read_u64();
        uint64_t kv_count = read_u64();

        for (uint64_t i = 0; i < kv_count; i++) {
            std::string key = read_string();
            GgufType val_type = (GgufType)read_i32();

            switch (val_type) {
                case GgufType::UINT8:   metadata_uint32[key] = read_u8(); break;
                case GgufType::INT8:    metadata_int[key] = read_i8(); break;
                case GgufType::UINT16:  metadata_uint32[key] = read_u16(); break;
                case GgufType::INT16:   metadata_int[key] = read_i16(); break;
                case GgufType::UINT32:  metadata_uint32[key] = read_u32(); break;
                case GgufType::INT32:   metadata_int[key] = read_i32(); break;
                case GgufType::FLOAT32: metadata_float[key] = read_f32(); break;
                case GgufType::BOOL:    metadata_int[key] = read_i8(); break;
                case GgufType::STRING:  metadata_str[key] = read_string(); break;
                case GgufType::UINT64:  metadata_int[key] = (int64_t)read_u64(); break;
                case GgufType::INT64:   metadata_int[key] = read_i64(); break;
                case GgufType::FLOAT64: metadata_float[key] = (float)read_f64(); break;
                case GgufType::ARRAY: {
                    GgufType arr_type = (GgufType)read_i32();
                    uint64_t arr_n = read_u64();
                    for (uint64_t j = 0; j < arr_n; j++) {
                        if (arr_type == GgufType::STRING) {
                            metadata_str[key + "_" + std::to_string(j)] = read_string();
                        } else if (arr_type == GgufType::INT32) {
                            (void)read_i32();
                        } else if (arr_type == GgufType::FLOAT32) {
                            (void)read_f32();
                        }
                    }
                    break;
                }
                default: break;
            }

            if (key == "general.alignment") {
                alignment = metadata_uint32.at("general.alignment");
            }
        }

        for (uint64_t i = 0; i < tensor_count; i++) {
            GgufTensorInfo info;
            info.name = read_string();
            uint32_t n_dims = read_u32();
            info.dims.resize(n_dims);
            for (uint32_t j = 0; j < n_dims; j++) {
                info.dims[j] = read_i64();
            }
            info.type = (GgmlType)read_i32();
            info.offset = read_u64();
            tensors[info.name] = info;
        }

        // Align to alignment boundary
        tensor_data_offset = (pos + alignment - 1) & ~(alignment - 1);
        return true;
    }

    const void *tensor_data(const std::string &name) const {
        auto it = tensors.find(name);
        if (it == tensors.end()) return nullptr;
        return data.data() + tensor_data_offset + it->second.offset;
    }

    template<typename T>
    T get_metadata(const std::string &key, T default_val) const {
        return default_val;
    }
};

template<>
inline std::string GgufReader::get_metadata(const std::string &key, std::string default_val) const {
    auto it = metadata_str.find(key);
    return it != metadata_str.end() ? it->second : default_val;
}

template<>
inline int64_t GgufReader::get_metadata(const std::string &key, int64_t default_val) const {
    auto it = metadata_int.find(key);
    if (it != metadata_int.end()) return it->second;
    auto u32 = metadata_uint32.find(key);
    return u32 != metadata_uint32.end() ? (int64_t)u32->second : default_val;
}

template<>
inline float GgufReader::get_metadata(const std::string &key, float default_val) const {
    auto it = metadata_float.find(key);
    return it != metadata_float.end() ? it->second : (float)get_metadata<int64_t>(key, (int64_t)default_val);
}

template<>
inline uint32_t GgufReader::get_metadata(const std::string &key, uint32_t default_val) const {
    auto it = metadata_uint32.find(key);
    return it != metadata_uint32.end() ? it->second : (uint32_t)get_metadata<int64_t>(key, (int64_t)default_val);
}

inline int64_t tensor_nelements(const GgufTensorInfo &t) {
    int64_t n = 1;
    for (auto d : t.dims) n *= d;
    return n;
}
