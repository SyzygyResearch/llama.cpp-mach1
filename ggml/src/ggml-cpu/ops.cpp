#include "ops.h"

#include "ggml-cpu.h"
#include "ggml-impl.h"
#include "binary-ops.h"
#include "simd-gemm.h"
#include "ggml.h"
#include "unary-ops.h"
#include "vec.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <mutex>
#include <unordered_map>
#include <vector>

// ggml_compute_forward_dup

static void ggml_compute_forward_dup_same_cont(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(src0));
    GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src0));
    GGML_ASSERT(src0->type == dst->type);

    const size_t nb0 = ggml_type_size(src0->type);

    const int ith = params->ith; // thread index
    const int nth = params->nth; // number of threads

    // parallelize by blocks
    const int nk = ggml_nelements(src0)/ggml_blck_size(src0->type);
    const int dr = (nk + nth - 1) / nth;
    const int k0 = dr * ith;
    const int k1 = MIN(k0 + dr, nk);

    if (k0 < k1) {
        memcpy(
            ((char *)  dst->data + k0*nb0),
            ((char *) src0->data + k0*nb0),
            (k1 - k0) * nb0);
    }
}

template<typename src_t, typename dst_t>
static void ggml_compute_forward_dup_flt(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(src0));
    GGML_ASSERT(!ggml_is_quantized(src0->type) && !ggml_is_quantized(dst->type));

    GGML_TENSOR_UNARY_OP_LOCALS

    const int ith = params->ith; // thread index
    const int nth = params->nth; // number of threads

    // parallelize by rows
    const int nr = ne01;
    // number of rows per thread
    const int dr = (nr + nth - 1) / nth;
    // row range for this thread
    const int ir0 = dr * ith;
    const int ir1 = MIN(ir0 + dr, nr);

    // case: type & row size equal
    if (src0->type == dst->type &&
        ne00 == ne0 &&
        nb00 == ggml_type_size(src0->type) && nb0 == ggml_type_size(dst->type)) {
        // copy by rows
        const size_t rs = ne00*nb00;
        for (int64_t i03 = 0; i03 < ne03; i03++) {
            for (int64_t i02 = 0; i02 < ne02; i02++) {
                for (int64_t i01 = ir0; i01 < ir1; i01++) {
                    memcpy(
                        ((char *)  dst->data + i01*nb1  + i02*nb2  + i03*nb3),
                        ((char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03),
                        rs);
                }
            }
        }
        return;
    }

    // case: dst tensor is contiguous
    if (ggml_is_contiguous(dst)) {
        if (nb00 == sizeof(src_t)) {
            if constexpr (std::is_same_v<dst_t, src_t>) {
                // same type
                size_t id = 0;
                const size_t rs = ne00 * nb00;
                char * dst_ptr = (char *) dst->data;

                for (int i03 = 0; i03 < ne03; i03++) {
                    for (int i02 = 0; i02 < ne02; i02++) {
                        id += rs * ir0;
                        for (int i01 = ir0; i01 < ir1; i01++) {
                            const char * src0_ptr = (char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03;
                            memcpy(dst_ptr + id, src0_ptr, rs);
                            id += rs;
                        }
                        id += rs * (ne01 - ir1);
                    }
                }
            } else {
                // casting between non-quantized types
                size_t id = 0;
                dst_t * dst_ptr = (dst_t *) dst->data;

                for (int i03 = 0; i03 < ne03; i03++) {
                    for (int i02 = 0; i02 < ne02; i02++) {
                        id += ne00 * ir0;
                        for (int i01 = ir0; i01 < ir1; i01++) {
                            const src_t * src0_ptr = (src_t *) ((char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03);
                            for (int i00 = 0; i00 < ne00; i00++) {
                                float tmp = type_conversion_table<src_t>::to_f32(src0_ptr[i00]);
                                dst_ptr[id] = type_conversion_table<dst_t>::from_f32(tmp);
                                id++;
                            }
                        }
                        id += ne00 * (ne01 - ir1);
                    }
                }
            }
        } else {
            //printf("%s: this is not optimal - fix me\n", __func__);

            size_t id = 0;
            dst_t * dst_ptr = (dst_t *) dst->data;

            for (int i03 = 0; i03 < ne03; i03++) {
                for (int i02 = 0; i02 < ne02; i02++) {
                    id += ne00 * ir0;
                    for (int i01 = ir0; i01 < ir1; i01++) {
                        for (int i00 = 0; i00 < ne00; i00++) {
                            const src_t * src0_ptr = (src_t *) ((char *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);

                            float tmp = type_conversion_table<src_t>::to_f32(*src0_ptr);
                            dst_ptr[id] = type_conversion_table<dst_t>::from_f32(tmp);
                            id++;
                        }
                    }
                    id += ne00 * (ne01 - ir1);
                }
            }
        }
        return;
    }

    // dst counters
    int64_t i10 = 0;
    int64_t i11 = 0;
    int64_t i12 = 0;
    int64_t i13 = 0;

    if constexpr (std::is_same_v<dst_t, src_t>) {
        for (int64_t i03 = 0; i03 < ne03; i03++) {
            for (int64_t i02 = 0; i02 < ne02; i02++) {
                i10 += ne00 * ir0;
                while (i10 >= ne0) {
                    i10 -= ne0;
                    if (++i11 == ne1) {
                        i11 = 0;
                        if (++i12 == ne2) {
                            i12 = 0;
                            if (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
                for (int64_t i01 = ir0; i01 < ir1; i01++) {
                    for (int64_t i00 = 0; i00 < ne00; i00++) {
                        const char * src0_ptr = ((char *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                              char * dst_ptr  = ((char *)  dst->data + i10*nb0  + i11*nb1  + i12*nb2  + i13*nb3);

                        memcpy(dst_ptr, src0_ptr, sizeof(dst_t));

                        if (++i10 == ne00) {
                            i10 = 0;
                            if (++i11 == ne01) {
                                i11 = 0;
                                if (++i12 == ne02) {
                                    i12 = 0;
                                    if (++i13 == ne03) {
                                        i13 = 0;
                                    }
                                }
                            }
                        }
                    }
                }
                i10 += ne00 * (ne01 - ir1);
                while (i10 >= ne0) {
                    i10 -= ne0;
                    if (++i11 == ne1) {
                        i11 = 0;
                        if (++i12 == ne2) {
                            i12 = 0;
                            if (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
            }
        }

    } else {
        for (int64_t i03 = 0; i03 < ne03; i03++) {
            for (int64_t i02 = 0; i02 < ne02; i02++) {
                i10 += ne00 * ir0;
                while (i10 >= ne0) {
                    i10 -= ne0;
                    if (++i11 == ne1) {
                        i11 = 0;
                        if (++i12 == ne2) {
                            i12 = 0;
                            if (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
                for (int64_t i01 = ir0; i01 < ir1; i01++) {
                    for (int64_t i00 = 0; i00 < ne00; i00++) {
                        const char * src0_ptr = ((char *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                              char * dst_ptr  = ((char *)  dst->data + i10*nb0  + i11*nb1  + i12*nb2  + i13*nb3);

                        float tmp = type_conversion_table<src_t>::to_f32(*(const src_t *) src0_ptr);
                        *(dst_t *) dst_ptr = type_conversion_table<dst_t>::from_f32(tmp);

                        if (++i10 == ne0) {
                            i10 = 0;
                            if (++i11 == ne1) {
                                i11 = 0;
                                if (++i12 == ne2) {
                                    i12 = 0;
                                    if (++i13 == ne3) {
                                        i13 = 0;
                                    }
                                }
                            }
                        }
                    }
                }
                i10 += ne00 * (ne01 - ir1);
                while (i10 >= ne0) {
                    i10 -= ne0;
                    if (++i11 == ne1) {
                        i11 = 0;
                        if (++i12 == ne2) {
                            i12 = 0;
                            if (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
            }
        }
    }
}


template<typename src_t>
static void ggml_compute_forward_dup_to_q(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(src0));
    GGML_ASSERT(!ggml_is_quantized(src0->type));

    GGML_TENSOR_UNARY_OP_LOCALS

    const int ith = params->ith; // thread index
    const int nth = params->nth; // number of threads

    // parallelize by rows
    const int nr = ne01;
    // number of rows per thread
    const int dr = (nr + nth - 1) / nth;
    // row range for this thread
    const int ir0 = dr * ith;
    const int ir1 = MIN(ir0 + dr, nr);

    if (ggml_is_contiguous(dst) &&
            nb00 == sizeof(src_t) &&
            ggml_get_type_traits_cpu(dst->type)->from_float) {
        // casting non-quantized types --> intermediate f32 --> quantized
        ggml_from_float_t const quantize_row_q = ggml_get_type_traits_cpu(dst->type)->from_float;
        float * src0_f32 = (float *) params->wdata + (ne00 + CACHE_LINE_SIZE_F32) * ith;

        size_t id = 0;
        size_t rs = nb0 * (ne00 / ggml_blck_size(dst->type));
        char * dst_ptr = (char *) dst->data;

        for (int i03 = 0; i03 < ne03; i03++) {
            for (int i02 = 0; i02 < ne02; i02++) {
                id += rs * ir0;
                for (int i01 = ir0; i01 < ir1; i01++) {
                    const src_t * src0_ptr = (src_t *) ((char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03);

                    for (int i00 = 0; i00 < ne00; i00++) {
                        src0_f32[i00] = type_conversion_table<src_t>::to_f32(src0_ptr[i00]);
                    }

                    quantize_row_q(src0_f32, dst_ptr + id, ne00);
                    id += rs;
                }
                id += rs * (ne01 - ir1);
            }
        }
    } else {
        // printf("%s %s\n", ggml_type_name(src0->type), ggml_type_name(dst->type));
        GGML_ABORT("not implemented");
    }
}

// A simplified version of ggml_compute_forward_dup that doesn't do float upcasting, and just plain old memcpy.
static void ggml_compute_forward_dup_bytes(
        const ggml_compute_params * params,
        ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(src0));
    GGML_ASSERT(src0->type == dst->type);

    GGML_TENSOR_UNARY_OP_LOCALS;

    if (ggml_is_contiguous(src0) && ggml_is_contiguous(dst)) {
        ggml_compute_forward_dup_same_cont(params, dst);
        return;
    }

    const size_t type_size = ggml_type_size(src0->type);

    const int ith = params->ith; // thread index
    const int nth = params->nth; // number of threads

    // parallelize by rows
    const int nr = ne01;
    // number of rows per thread
    const int dr = (nr + nth - 1) / nth;
    // row range for this thread
    const int ir0 = dr * ith;
    const int ir1 = MIN(ir0 + dr, nr);

    if (src0->type == dst->type &&
        ggml_are_same_shape(src0, dst) &&
        nb00 == type_size && nb0 == type_size) {
        // copy by rows
        const size_t rs = ggml_row_size(src0->type, ne00);
        for (int64_t i03 = 0; i03 < ne03; i03++) {
            for (int64_t i02 = 0; i02 < ne02; i02++) {
                for (int64_t i01 = ir0; i01 < ir1; i01++) {
                    memcpy(
                        ((char *)  dst->data + i01*nb1  + i02*nb2  + i03*nb3),
                        ((char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03),
                        rs);
                }
            }
        }
        return;
    }

    if (ggml_is_contiguous(dst)) {
        size_t id = 0;
        char * dst_ptr = (char *) dst->data;
        const size_t rs = ne00 * type_size;

        if (nb00 == type_size) {
            // src0 is contiguous on first dimension, copy by rows
            for (int64_t i03 = 0; i03 < ne03; i03++) {
                for (int64_t i02 = 0; i02 < ne02; i02++) {
                    id += rs * ir0;
                    for (int64_t i01 = ir0; i01 < ir1; i01++) {
                        const char * src0_ptr = (char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03;
                        memcpy(dst_ptr + id, src0_ptr, rs);
                        id += rs;
                    }
                    id += rs * (ne01 - ir1);
                }
            }
        } else {
            //printf("%s: this is not optimal - fix me\n", __func__);

            for (int64_t i03 = 0; i03 < ne03; i03++) {
                for (int64_t i02 = 0; i02 < ne02; i02++) {
                    id += rs * ir0;
                    for (int64_t i01 = ir0; i01 < ir1; i01++) {
                        for (int64_t i00 = 0; i00 < ne00; i00++) {
                            const char * src0_ptr = (char *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03;
                            memcpy(dst_ptr + id, src0_ptr, type_size);

                            id += type_size;
                        }
                    }
                    id += rs * (ne01 - ir1);
                }
            }
        }

        return;
    }

    // dst counters
    int64_t k10 = 0;
    int64_t i11 = 0;
    int64_t i12 = 0;
    int64_t i13 = 0;

    // number of blocks in a row
    const int64_t nk00 = ne00 / ggml_blck_size(src0->type);
    const int64_t nk0  = ne0  / ggml_blck_size(dst->type);

    for (int64_t i03 = 0; i03 < ne03; i03++) {
        for (int64_t i02 = 0; i02 < ne02; i02++) {
            k10 += nk00 * ir0;
            while (k10 >= nk0) {
                k10 -= nk0;
                if (++i11 == ne1) {
                    i11 = 0;
                    if (++i12 == ne2) {
                        i12 = 0;
                        if (++i13 == ne3) {
                            i13 = 0;
                        }
                    }
                }
            }
            for (int64_t i01 = ir0; i01 < ir1; i01++) {
                for (int64_t k00 = 0; k00 < nk00; k00++) {
                    const char * src0_ptr = ((char *) src0->data + k00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                          char * dst_ptr  = ((char *)  dst->data + k10*nb0  + i11*nb1  + i12*nb2  + i13*nb3);

                    memcpy(dst_ptr, src0_ptr, type_size);

                    if (++k10 == nk0) {
                        k10 = 0;
                        if (++i11 == ne1) {
                            i11 = 0;
                            if (++i12 == ne2) {
                                i12 = 0;
                                if (++i13 == ne3) {
                                    i13 = 0;
                                }
                            }
                        }
                    }
                }
            }
            k10 += nk00 * (ne01 - ir1);
            while (k10 >= nk0) {
                k10 -= nk0;
                if (++i11 == ne1) {
                    i11 = 0;
                    if (++i12 == ne2) {
                        i12 = 0;
                        if (++i13 == ne3) {
                            i13 = 0;
                        }
                    }
                }
            }
        }
    }
}

static void ggml_compute_forward_dup_from_q(
        const ggml_compute_params * params,
              ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    const ggml_type type = src0->type;
    ggml_to_float_t const dequantize_row_q = ggml_get_type_traits(type)->to_float;

    size_t qk = ggml_blck_size(type);
    const int64_t nr = ggml_nelements(src1) / qk;

    // destination must be contiguous in the first dimension
    GGML_ASSERT(nb10 == ggml_type_size(dst->type));
    // must either have first dimension large enough to hold a row, or fully contiguous
    GGML_ASSERT((ne10 % qk) == 0 || ggml_is_contiguous(dst));

    const int ith = params->ith;
    const int nth = params->nth;

    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int64_t ir = ir0; ir < ir1; ++ir) {

        uint32_t i = ir * qk;

        const int64_t i03 = i/(ne00 * ne01 * ne02);
        const int64_t i02 = (i - i03*ne00*ne01*ne02 )/ (ne00*ne01);
        const int64_t i01 = (i - i03*ne00*ne01*ne02  -  i02*ne01*ne00) / ne00;
        const int64_t i00 = i - i03*ne00*ne01*ne02 - i02*ne01*ne00 - i01*ne00;
        const int64_t x_offset = (i00/qk)*nb00 + i01*nb01 + i02*nb02 + i03 * nb03;

        const int64_t i13 = i/(ne10 * ne11 * ne12);
        const int64_t i12 = (i - i13*ne10*ne11*ne12) / (ne10*ne11);
        const int64_t i11 = (i - i13*ne10*ne11*ne12 - i12*ne10*ne11) / ne10;
        const int64_t i10 = i - i13*ne10*ne11*ne12 - i12*ne10*ne11 - i11*ne10;
        const int64_t dst_offset = i10*nb10 + i11*nb11 + i12*nb12 + i13*nb13;

        dequantize_row_q(
                (const void *) ((char *) src0->data + x_offset),
                     (float *) ((char *)  dst->data + dst_offset), qk);
    }
}

void ggml_compute_forward_dup(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    if (src0->type == dst->type) {
        ggml_compute_forward_dup_bytes(params, dst);
        return;
    }

    switch (src0->type) {
        case GGML_TYPE_F16:
            {
                /**/ if (dst->type == GGML_TYPE_F16)  ggml_compute_forward_dup_flt<ggml_fp16_t, ggml_fp16_t>(params, dst);
                else if (dst->type == GGML_TYPE_BF16) ggml_compute_forward_dup_flt<ggml_fp16_t, ggml_bf16_t>(params, dst);
                else if (dst->type == GGML_TYPE_F32)  ggml_compute_forward_dup_flt<ggml_fp16_t, float      >(params, dst);
                else ggml_compute_forward_dup_to_q<ggml_fp16_t>(params, dst);
            } break;
        case GGML_TYPE_BF16:
            {
                /**/ if (dst->type == GGML_TYPE_F16)  ggml_compute_forward_dup_flt<ggml_bf16_t, ggml_fp16_t>(params, dst);
                else if (dst->type == GGML_TYPE_BF16) ggml_compute_forward_dup_flt<ggml_bf16_t, ggml_bf16_t>(params, dst);
                else if (dst->type == GGML_TYPE_F32)  ggml_compute_forward_dup_flt<ggml_bf16_t, float      >(params, dst);
                else ggml_compute_forward_dup_to_q<ggml_bf16_t>(params, dst);
            } break;
        case GGML_TYPE_F32:
            {
                /**/ if (dst->type == GGML_TYPE_F16)  ggml_compute_forward_dup_flt<float, ggml_fp16_t>(params, dst);
                else if (dst->type == GGML_TYPE_BF16) ggml_compute_forward_dup_flt<float, ggml_bf16_t>(params, dst);
                else if (dst->type == GGML_TYPE_F32)  ggml_compute_forward_dup_flt<float, float      >(params, dst);
                else if (dst->type == GGML_TYPE_I32)  ggml_compute_forward_dup_flt<float, int32_t    >(params, dst);
                else ggml_compute_forward_dup_to_q<float>(params, dst);
            } break;
        case GGML_TYPE_I32:
            {
                if (dst->type == GGML_TYPE_F32) ggml_compute_forward_dup_flt<int32_t, float>(params, dst);
                else GGML_ABORT("not implemented");
            } break;
        default:
            {
                if (ggml_is_quantized(src0->type) && dst->type == GGML_TYPE_F32) {
                    ggml_compute_forward_dup_from_q(params, dst);
                    break;
                }
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_add

static void ggml_compute_forward_add_q_f32(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, src1) && ggml_are_same_shape(src0, dst));

    const int nr  = ggml_nrows(src0);

    GGML_TENSOR_BINARY_OP_LOCALS

    const int ith = params->ith;
    const int nth = params->nth;

    const ggml_type type = src0->type;
    const ggml_type dtype = dst->type;
    ggml_to_float_t const dequantize_row_q = ggml_get_type_traits(type)->to_float;
    ggml_from_float_t const quantize_row_q = ggml_get_type_traits_cpu(dtype)->from_float;

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(type));
    GGML_ASSERT(nb10 == sizeof(float));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    GGML_ASSERT(ggml_is_quantized(src0->type));
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    float * wdata = (float *) params->wdata + (ne00 + CACHE_LINE_SIZE_F32) * ith;

    for (int ir = ir0; ir < ir1; ++ir) {
        // src0 indices
        const int i03 = ir/(ne02*ne01);
        const int i02 = (ir - i03*ne02*ne01)/ne01;
        const int i01 = (ir - i03*ne02*ne01 - i02*ne01);

        // src1 and dst are same shape as src0 => same indices
        const int i13 = i03;
        const int i12 = i02;
        const int i11 = i01;

        const int i3 = i03;
        const int i2 = i02;
        const int i1 = i01;

        void  * src0_row = (void *) ((char *) src0->data + (i01*nb01 + i02*nb02 + i03*nb03));
        float * src1_row = (float *)((char *) src1->data + (i11*nb11 + i12*nb12 + i13*nb13));
        void  * dst_row  = (void *) ((char *)  dst->data + ( i1*nb1  +  i2*nb2  +  i3*nb3));

        assert(ne00 % 32 == 0);

        // unquantize row from src0 to temp buffer
        dequantize_row_q(src0_row, wdata, ne00);
        // add src1
        ggml_vec_acc_f32(ne00, wdata, src1_row);
        // quantize row to dst
        if (quantize_row_q != NULL) {
            quantize_row_q(wdata, dst_row, ne00);
        } else {
            memcpy(dst_row, wdata, ne0*nb0);
        }
    }
}

void ggml_compute_forward_add(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
            {
                ggml_compute_forward_add_non_quantized(params, dst);
            } break;
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q2_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_NVFP4:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_TQ1_0:
        case GGML_TYPE_TQ2_0:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ2_S:
            {
                ggml_compute_forward_add_q_f32(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_add_id

static void ggml_compute_forward_add_id_f32(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * src2 = dst->src[2];

    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(src2->type == GGML_TYPE_I32);

    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(src1->nb[0] == sizeof(float));

    const int ith = params->ith;
    const int nth = params->nth;

    const int nr  = ggml_nrows(src0);

    GGML_TENSOR_TERNARY_OP_LOCALS

    GGML_ASSERT( nb0 == sizeof(float));
    GGML_ASSERT(nb10 == sizeof(float));

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int ir = ir0; ir < ir1; ++ir) {
        // src0 indices
        const int i3 = ir/(ne2*ne1);
        const int i2 = (ir - i3*ne2*ne1)/ne1;
        const int i1 = (ir - i3*ne2*ne1 - i2*ne1);

        // src1 indices
        const int i11 = *(int32_t *) ((char *) src2->data + i1*nb20 + i2*nb21);

        GGML_ASSERT(i11 >= 0 && i11 < ne11);

        ggml_vec_add_f32(ne0,
                (float *) ((char *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1 ),
                (float *) ((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01),
                (float *) ((char *) src1->data + i11*nb11));
    }
}

void ggml_compute_forward_add_id(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_add_id_f32(params, dst);
            } break;
        default:
            {
                GGML_ABORT("unsupported type for ggml_compute_forward_add_id: %s", ggml_type_name(src0->type));
            }
    }
}

// ggml_compute_forward_add1

static void ggml_compute_forward_add1_f32(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_scalar(src1));

    const int ith = params->ith;
    const int nth = params->nth;

    const int nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT( nb0 == sizeof(float));
    GGML_ASSERT(nb00 == sizeof(float));

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        const int i3 = ir/(ne2*ne1);
        const int i2 = (ir - i3*ne2*ne1)/ne1;
        const int i1 = (ir - i3*ne2*ne1 - i2*ne1);

#ifdef GGML_USE_ACCELERATE
        GGML_UNUSED(ggml_vec_add1_f32);

        vDSP_vadd(
                (float *) ((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01), 1,
                (float *) ((char *) src1->data), 0,
                (float *) ((char *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1 ), 1,
                ne0);
#else
        ggml_vec_add1_f32(ne0,
                (float *) ((char *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1 ),
                (float *) ((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01),
               *(float *) src1->data);
#endif
    }
}

static void ggml_compute_forward_add1_f16_f32(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_scalar(src1));

    // scalar to add
    const float v = *(float *) src1->data;

    const int ith = params->ith;
    const int nth = params->nth;

    const int nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F16);

    GGML_ASSERT( nb0 == sizeof(ggml_fp16_t));
    GGML_ASSERT(nb00 == sizeof(ggml_fp16_t));

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        const int i3 = ir/(ne2*ne1);
        const int i2 = (ir - i3*ne2*ne1)/ne1;
        const int i1 = (ir - i3*ne2*ne1 - i2*ne1);

        ggml_fp16_t * dst_ptr  = (ggml_fp16_t *) ((char *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1 );
        ggml_fp16_t * src0_ptr = (ggml_fp16_t *) ((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01);
        for (int i = 0; i < ne0; i++) {
            dst_ptr[i] = GGML_CPU_FP32_TO_FP16(GGML_CPU_FP16_TO_FP32(src0_ptr[i]) + v);
        }
    }
}

static void ggml_compute_forward_add1_f16_f16(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_scalar(src1));

    // scalar to add
    const float v = GGML_CPU_FP16_TO_FP32(*(ggml_fp16_t *) src1->data);

    const int ith = params->ith;
    const int nth = params->nth;

    const int nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type  == GGML_TYPE_F16);

    GGML_ASSERT( nb0 == sizeof(ggml_fp16_t));
    GGML_ASSERT(nb00 == sizeof(ggml_fp16_t));

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        const int i3 = ir/(ne2*ne1);
        const int i2 = (ir - i3*ne2*ne1)/ne1;
        const int i1 = (ir - i3*ne2*ne1 - i2*ne1);

        ggml_fp16_t * dst_ptr  = (ggml_fp16_t *) ((char *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1 );
        ggml_fp16_t * src0_ptr = (ggml_fp16_t *) ((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01);
        for (int i = 0; i < ne0; i++) {
            dst_ptr[i] = GGML_CPU_FP32_TO_FP16(GGML_CPU_FP16_TO_FP32(src0_ptr[i]) + v);
        }
    }
}

static void ggml_compute_forward_add1_q_f32(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_scalar(src1));

    // scalar to add
    const float v = *(float *) src1->data;

    const int ith = params->ith;
    const int nth = params->nth;

    const int nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS

    const ggml_type type = src0->type;
    ggml_to_float_t const dequantize_row_q = ggml_get_type_traits(type)->to_float;
    ggml_from_float_t const quantize_row_q = ggml_get_type_traits_cpu(type)->from_float;

    // we don't support permuted src0
    GGML_ASSERT(nb00 == ggml_type_size(type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    GGML_ASSERT(ggml_is_quantized(src0->type));
    GGML_ASSERT(dst->type == src0->type);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    float * wdata = (float *) params->wdata + (ne0 + CACHE_LINE_SIZE_F32) * ith;

    for (int ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        const int i3 = ir/(ne2*ne1);
        const int i2 = (ir - i3*ne2*ne1)/ne1;
        const int i1 = (ir - i3*ne2*ne1 - i2*ne1);

        void  * src0_row = (void *) ((char *) src0->data + (i1*nb01 + i2*nb02 + i3*nb03));
        void  * dst_row  = (void *) ((char *)  dst->data + (i1*nb1  + i2*nb2  + i3*nb0 ));

        assert(ne0 % 32 == 0);

        // unquantize row from src0 to temp buffer
        dequantize_row_q(src0_row, wdata, ne0);
        // add src1
        ggml_vec_acc1_f32(ne0, wdata, v);
        // quantize row to dst
        quantize_row_q(wdata, dst_row, ne0);
    }
}

static void ggml_compute_forward_add1_bf16_f32(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_scalar(src1));

    // scalar to add
    const float v = *(float *) src1->data;

    const int ith = params->ith;
    const int nth = params->nth;

    const int nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(src0->type == GGML_TYPE_BF16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_BF16);

    GGML_ASSERT( nb0 == sizeof(ggml_bf16_t));
    GGML_ASSERT(nb00 == sizeof(ggml_bf16_t));

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        const int i3 = ir/(ne2*ne1);
        const int i2 = (ir - i3*ne2*ne1)/ne1;
        const int i1 = (ir - i3*ne2*ne1 - i2*ne1);

        ggml_bf16_t * dst_ptr  = (ggml_bf16_t *) ((char *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1 );
        ggml_bf16_t * src0_ptr = (ggml_bf16_t *) ((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01);
        for (int i = 0; i < ne0; i++) {
            dst_ptr[i] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(src0_ptr[i]) + v);
        }
    }
}

static void ggml_compute_forward_add1_bf16_bf16(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_scalar(src1));

    // scalar to add
    const float v = GGML_BF16_TO_FP32(*(ggml_bf16_t *) src1->data);

    const int ith = params->ith;
    const int nth = params->nth;

    const int nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(src0->type == GGML_TYPE_BF16);
    GGML_ASSERT(src1->type == GGML_TYPE_BF16);
    GGML_ASSERT(dst->type  == GGML_TYPE_BF16);

    GGML_ASSERT( nb0 == sizeof(ggml_bf16_t));
    GGML_ASSERT(nb00 == sizeof(ggml_bf16_t));

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        const int i3 = ir/(ne2*ne1);
        const int i2 = (ir - i3*ne2*ne1)/ne1;
        const int i1 = (ir - i3*ne2*ne1 - i2*ne1);

        ggml_bf16_t * dst_ptr  = (ggml_bf16_t *) ((char *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1 );
        ggml_bf16_t * src0_ptr = (ggml_bf16_t *) ((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01);
        for (int i = 0; i < ne0; i++) {
            dst_ptr[i] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(src0_ptr[i]) + v);
        }
    }
}

void ggml_compute_forward_add1(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_add1_f32(params, dst);
            } break;
        case GGML_TYPE_F16:
            {
                if (src1->type == GGML_TYPE_F16) {
                    ggml_compute_forward_add1_f16_f16(params, dst);
                }
                else if (src1->type == GGML_TYPE_F32) {
                    ggml_compute_forward_add1_f16_f32(params, dst);
                }
                else {
                    GGML_ABORT("fatal error");
                }
            } break;
        case GGML_TYPE_BF16:
            {
                if (src1->type == GGML_TYPE_BF16) {
                    ggml_compute_forward_add1_bf16_bf16(params, dst);
                }
                else if (src1->type == GGML_TYPE_F32) {
                    ggml_compute_forward_add1_bf16_f32(params, dst);
                }
                else {
                    GGML_ABORT("fatal error");
                }
            } break;
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q2_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q8_1:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_NVFP4:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_TQ1_0:
        case GGML_TYPE_TQ2_0:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ2_S:
            {
                ggml_compute_forward_add1_q_f32(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_acc

static void ggml_compute_forward_acc_f32(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src0));

    // view src0 and dst with these strides and data offset inbytes during acc
    // nb0 is implicitly element_size because src0 and dst are contiguous
    size_t nb1     = ((int32_t *) dst->op_params)[0];
    size_t nb2     = ((int32_t *) dst->op_params)[1];
    size_t nb3     = ((int32_t *) dst->op_params)[2];
    size_t offset  = ((int32_t *) dst->op_params)[3];
    bool   inplace = (bool) ((int32_t *) dst->op_params)[4];

    if (!inplace) {
        if (params->ith == 0) {
            // memcpy needs to be synchronized across threads to avoid race conditions.
            // => do it in INIT phase
            memcpy(
                ((char *)  dst->data),
                ((char *) src0->data),
                ggml_nbytes(dst));
        }
        ggml_barrier(params->threadpool);
    }

    const int ith = params->ith;
    const int nth = params->nth;

    const int nr = ggml_nrows(src1);
    const int nc = src1->ne[0];

    GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne)
    GGML_TENSOR_LOCALS(size_t,  nb1, src1, nb)

    // src0 and dst as viewed during acc
    const size_t nb0 = ggml_element_size(src0);

    const size_t nb00 = nb0;
    const size_t nb01 = nb1;
    const size_t nb02 = nb2;
    const size_t nb03 = nb3;

    GGML_ASSERT(offset + (ne10 == 0 ? 0 : ne10-1)*nb0  + (ne11 == 0 ? 0 : ne11-1)*nb1  + (ne12 == 0 ? 0 : ne12-1)*nb2  + (ne13 == 0 ? 0 : ne13-1)*nb3  < ggml_nbytes(dst));
    GGML_ASSERT(offset + (ne10 == 0 ? 0 : ne10-1)*nb00 + (ne11 == 0 ? 0 : ne11-1)*nb01 + (ne12 == 0 ? 0 : ne12-1)*nb02 + (ne13 == 0 ? 0 : ne13-1)*nb03 < ggml_nbytes(src0));

    GGML_ASSERT(nb10 == sizeof(float));

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are viewed with shape of src1 and offset
        // => same indices
        const int i3 = ir/(ne12*ne11);
        const int i2 = (ir - i3*ne12*ne11)/ne11;
        const int i1 = (ir - i3*ne12*ne11 - i2*ne11);

#ifdef GGML_USE_ACCELERATE
        vDSP_vadd(
                (float *) ((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01 + offset), 1,
                (float *) ((char *) src1->data + i3*nb13 + i2*nb12 + i1*nb11), 1,
                (float *) ((char *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1  + offset), 1, nc);
#else
        ggml_vec_add_f32(nc,
                (float *) ((char *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1  + offset),
                (float *) ((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01 + offset),
                (float *) ((char *) src1->data + i3*nb13 + i2*nb12 + i1*nb11));
#endif
    }
}

void ggml_compute_forward_acc(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_acc_f32(params, dst);
            } break;
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q2_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q8_1:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_NVFP4:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_TQ1_0:
        case GGML_TYPE_TQ2_0:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ2_S:
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_sum

static void ggml_compute_forward_sum_f32(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    if (params->ith != 0) {
        return;
    }

    assert(ggml_is_scalar(dst));
    assert(src0->nb[0] == sizeof(float));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)

    ggml_float sum     = 0;
    ggml_float row_sum = 0;

    for (int64_t i03 = 0; i03 < ne03; i03++) {
        for (int64_t i02 = 0; i02 < ne02; i02++) {
            for (int64_t i01 = 0; i01 < ne01; i01++) {
                ggml_vec_sum_f32_ggf(ne00,
                        &row_sum,
                        (float *) ((char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03));
                sum += row_sum;
            }
        }
    }
    ((float *) dst->data)[0] = sum;
}

static void ggml_compute_forward_sum_f16(
    const ggml_compute_params * params,
          ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    if (params->ith != 0) {
        return;
    }

    assert(ggml_is_scalar(dst));

    assert(src0->nb[0] == sizeof(ggml_fp16_t));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)

    float sum = 0;
    float row_sum = 0;

    for (int64_t i03 = 0; i03 < ne03; i03++) {
        for (int64_t i02 = 0; i02 < ne02; i02++) {
            for (int64_t i01 = 0; i01 < ne01; i01++) {
                ggml_vec_sum_f16_ggf(ne00,
                    &row_sum,
                    (ggml_fp16_t *) ((char *) src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03));
                sum += row_sum;
            }
        }
    }
    ((ggml_fp16_t *) dst->data)[0] = GGML_CPU_FP32_TO_FP16(sum);
}

static void ggml_compute_forward_sum_bf16(
    const ggml_compute_params * params,
          ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    if (params->ith != 0) {
        return;
    }

    assert(ggml_is_scalar(dst));

    assert(src0->nb[0] == sizeof(ggml_bf16_t));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)

    float sum = 0;
    float row_sum = 0;

    for (int64_t i03 = 0; i03 < ne03; i03++) {
        for (int64_t i02 = 0; i02 < ne02; i02++) {
            for (int64_t i01 = 0; i01 < ne01; i01++) {
                ggml_vec_sum_bf16_ggf(ne00,
                    &row_sum,
                    (ggml_bf16_t *) ((char *) src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03));
                sum += row_sum;
            }
        }
    }
    ((ggml_bf16_t *) dst->data)[0] = GGML_FP32_TO_BF16(sum);
}

void ggml_compute_forward_sum(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_sum_f32(params, dst);
            } break;
        case GGML_TYPE_F16:
            {
                ggml_compute_forward_sum_f16(params, dst);
            } break;
        case GGML_TYPE_BF16:
            {
                ggml_compute_forward_sum_bf16(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_cumsum

static void ggml_compute_forward_cumsum_f32(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(dst->nb[0] == sizeof(float));

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(ne0 == ne00);
    GGML_ASSERT(ne1 == ne01);
    GGML_ASSERT(ne2 == ne02);
    GGML_ASSERT(ne3 == ne03);

    const auto [ir0, ir1] = get_thread_range(params, src0);

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t i03 = ir/(ne02*ne01);
        const int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        const int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        float * src_row = (float *) ((char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03);
        float * dst_row = (float *) ((char *) dst->data  + i01*nb1  + i02*nb2  + i03*nb3);

        ggml_vec_cumsum_f32(ne00, dst_row, src_row);
    }
}

void ggml_compute_forward_cumsum(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_cumsum_f32(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_sum_rows

static void ggml_compute_forward_sum_rows_f32(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    if (params->ith != 0) {
        return;
    }

    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(dst->nb[0] == sizeof(float));

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(ne0 == 1);
    GGML_ASSERT(ne1 == ne01);
    GGML_ASSERT(ne2 == ne02);
    GGML_ASSERT(ne3 == ne03);

    for (int64_t i3 = 0; i3 < ne03; i3++) {
        for (int64_t i2 = 0; i2 < ne02; i2++) {
            for (int64_t i1 = 0; i1 < ne01; i1++) {
                float * src_row = (float *) ((char *) src0->data + i1*nb01 + i2*nb02 + i3*nb03);
                float * dst_row = (float *) ((char *) dst->data  + i1*nb1  + i2*nb2  + i3*nb3);
                float row_sum = 0;
                ggml_vec_sum_f32(ne00, &row_sum, src_row);
                dst_row[0] = row_sum;
            }
        }
    }
}

void ggml_compute_forward_sum_rows(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_sum_rows_f32(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_mean

static void ggml_compute_forward_mean_f32(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    if (params->ith != 0) {
        return;
    }

    assert(src0->nb[0] == sizeof(float));

    GGML_TENSOR_UNARY_OP_LOCALS

    assert(ne0 == 1);
    assert(ne1 == ne01);
    assert(ne2 == ne02);
    assert(ne3 == ne03);

    GGML_UNUSED(ne0);
    GGML_UNUSED(ne1);
    GGML_UNUSED(ne2);
    GGML_UNUSED(ne3);

    for (int64_t i03 = 0; i03 < ne03; i03++) {
        for (int64_t i02 = 0; i02 < ne02; i02++) {
            for (int64_t i01 = 0; i01 < ne01; i01++) {
                ggml_vec_sum_f32(ne00,
                        (float *) ((char *)  dst->data + i01*nb1  + i02*nb2  + i03*nb3),
                        (float *) ((char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03));

                *(float *) ((char *) dst->data + i01*nb1 + i02*nb2 + i03*nb3) /= (float) ne00;
            }
        }
    }
}

void ggml_compute_forward_mean(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_mean_f32(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_argmax

static void ggml_compute_forward_argmax_f32(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    if (params->ith != 0) {
        return;
    }

    assert(src0->nb[0] == sizeof(float));
    assert(dst->nb[0] == sizeof(float));

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];

    const size_t nb01 = src0->nb[1];
    const size_t nb0 = dst->nb[0];

    for (int64_t i1 = 0; i1 < ne01; i1++) {
        float * src = (float *) ((char *) src0->data + i1*nb01);
        int32_t * dst_ = (int32_t *) ((char *)  dst->data + i1*nb0);
        int v = 0;
        ggml_vec_argmax_f32(ne00, &v, src);
        dst_[0] = v;
    }
}

void ggml_compute_forward_argmax(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_argmax_f32(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

static void ggml_compute_forward_argmax_masked_f32(
        const ggml_compute_params * params,
        ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    if (params->ith != 0) {
        return;
    }

    assert(src0->nb[0] == sizeof(float));
    assert(dst->nb[0] == sizeof(int32_t));

    const int32_t n_masked = ggml_get_op_params_i32(dst, 0);
    GGML_ASSERT(n_masked >= 0 && n_masked <= GGML_MAX_OP_PARAMS / (int) sizeof(int32_t) - 1);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];

    for (int64_t i1 = 0; i1 < ne01; ++i1) {
        const float * row = (const float *) ((const char *) src0->data + i1*src0->nb[1]);
        int32_t best = -1;
        float best_value = -INFINITY;

        for (int32_t col = 0; col < ne00; ++col) {
            bool is_masked = false;
            for (int32_t i = 0; i < n_masked; ++i) {
                is_masked |= col == ggml_get_op_params_i32(dst, i + 1);
            }
            if (!is_masked && (best < 0 || row[col] > best_value)) {
                best_value = row[col];
                best = col;
            }
        }

        int32_t * out = (int32_t *) ((char *) dst->data + i1*dst->nb[0]);
        out[0] = best;
    }
}

void ggml_compute_forward_argmax_masked(
        const ggml_compute_params * params,
        ggml_tensor * dst) {
    switch (dst->src[0]->type) {
        case GGML_TYPE_F32:
            ggml_compute_forward_argmax_masked_f32(params, dst);
            break;
        default:
            GGML_ABORT("fatal error");
    }
}

// ggml_compute_forward_count_equal

static void ggml_compute_forward_count_equal_i32(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS;

    GGML_ASSERT(src0->type == GGML_TYPE_I32);
    GGML_ASSERT(src1->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_are_same_shape(src0, src1));
    GGML_ASSERT(ggml_is_scalar(dst));
    GGML_ASSERT(dst->type == GGML_TYPE_I64);

    const int64_t nr = ggml_nrows(src0);

    const int ith = params->ith;
    const int nth = params->nth;

    int64_t * sums = (int64_t *) params->wdata;
    int64_t sum_thread = 0;

    // rows per thread
    const int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int64_t ir0 = dr*ith;
    const int64_t ir1 = MIN(ir0 + dr, nr);

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t i03 =  ir                        / (ne02*ne01);
        const int64_t i02 = (ir - i03*ne03)            /       ne01;
        const int64_t i01 =  ir - i03*ne03 - i02*ne02;

        const char * data0 = (const char *) src0->data + i03*nb03 + i02*nb02 + i01*nb01;
        const char * data1 = (const char *) src1->data + i03*nb13 + i02*nb12 + i01*nb11;

        for (int64_t i00 = 0; i00 < ne00; ++i00) {
            const int32_t val0 = *((const int32_t *) (data0 + i00*nb00));
            const int32_t val1 = *((const int32_t *) (data1 + i00*nb10));

            sum_thread += val0 == val1;
        }
    }
    if (ith != 0) {
        sums[ith] = sum_thread;
    }
    ggml_barrier(params->threadpool);

    if (ith != 0) {
        return;
    }

    for (int ith_other = 1; ith_other < nth; ++ith_other) {
        sum_thread += sums[ith_other];
    }
    *((int64_t *) dst->data) = sum_thread;
}

void ggml_compute_forward_count_equal(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_I32:
            {
                ggml_compute_forward_count_equal_i32(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_repeat

static void ggml_compute_forward_repeat_f32(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    if (params->ith != 0) {
        return;
    }

    GGML_ASSERT(ggml_can_repeat(src0, dst));

    GGML_TENSOR_UNARY_OP_LOCALS

    // guaranteed to be an integer due to the check in ggml_can_repeat
    const int nr0 = (int)(ne0/ne00);
    const int nr1 = (int)(ne1/ne01);
    const int nr2 = (int)(ne2/ne02);
    const int nr3 = (int)(ne3/ne03);

    // TODO: support for transposed / permuted tensors
    GGML_ASSERT(nb0  == sizeof(float));
    GGML_ASSERT(nb00 == sizeof(float));

    // TODO: maybe this is not optimal?
    for                         (int i3 = 0; i3 < nr3;  i3++) {
        for                     (int k3 = 0; k3 < ne03; k3++) {
            for                 (int i2 = 0; i2 < nr2;  i2++) {
                for             (int k2 = 0; k2 < ne02; k2++) {
                    for         (int i1 = 0; i1 < nr1;  i1++) {
                        for     (int k1 = 0; k1 < ne01; k1++) {
                            for (int i0 = 0; i0 < nr0;  i0++) {
                                ggml_vec_cpy_f32(ne00,
                                        (float *) ((char *)  dst->data + (i3*ne03 + k3)*nb3  + (i2*ne02 + k2)*nb2  + (i1*ne01 + k1)*nb1  + (i0*ne00)*nb0),
                                        (float *) ((char *) src0->data + (          k3)*nb03 + (          k2)*nb02 + (          k1)*nb01));
                            }
                        }
                    }
                }
            }
        }
    }
}

static void ggml_compute_forward_repeat_f16(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    if (params->ith != 0) {
        return;
    }

    GGML_ASSERT(ggml_can_repeat(src0, dst));

    GGML_TENSOR_UNARY_OP_LOCALS

    // guaranteed to be an integer due to the check in ggml_can_repeat
    const int nr0 = (int)(ne0/ne00);
    const int nr1 = (int)(ne1/ne01);
    const int nr2 = (int)(ne2/ne02);
    const int nr3 = (int)(ne3/ne03);

    // TODO: support for transposed / permuted tensors
    GGML_ASSERT(nb0  == sizeof(ggml_fp16_t));
    GGML_ASSERT(nb00 == sizeof(ggml_fp16_t));

    // TODO: maybe this is not optimal?
    for                         (int i3 = 0; i3 < nr3;  i3++) {
        for                     (int k3 = 0; k3 < ne03; k3++) {
            for                 (int i2 = 0; i2 < nr2;  i2++) {
                for             (int k2 = 0; k2 < ne02; k2++) {
                    for         (int i1 = 0; i1 < nr1;  i1++) {
                        for     (int k1 = 0; k1 < ne01; k1++) {
                            for (int i0 = 0; i0 < nr0;  i0++) {
                                ggml_fp16_t * y = (ggml_fp16_t *) ((char *)  dst->data + (i3*ne03 + k3)*nb3  + (i2*ne02 + k2)*nb2  + (i1*ne01 + k1)*nb1  + (i0*ne00)*nb0);
                                ggml_fp16_t * x = (ggml_fp16_t *) ((char *) src0->data + (          k3)*nb03 + (          k2)*nb02 + (          k1)*nb01);
                                // ggml_vec_cpy_f16(ne00, y, x)
                                for (int i = 0; i < ne00; ++i) {
                                    y[i]  = x[i];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void ggml_compute_forward_repeat(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
        case GGML_TYPE_I16:
            {
                ggml_compute_forward_repeat_f16(params, dst);
            } break;
        case GGML_TYPE_F32:
        case GGML_TYPE_I32:
            {
                ggml_compute_forward_repeat_f32(params, dst);
            } break;
        // TODO: templateify the implementation and support for I64
        //       ref https://github.com/ggml-org/llama.cpp/pull/14274#discussion_r2169492225
        //case GGML_TYPE_I64:
        //    {
        //        ggml_compute_forward_repeat_i64(params, dst);
        //    } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_repeat_back

static void ggml_compute_forward_repeat_back_f32(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    if (params->ith != 0) {
        return;
    }

    GGML_ASSERT(ggml_can_repeat(dst, src0));

    GGML_TENSOR_UNARY_OP_LOCALS

    // guaranteed to be an integer due to the check in ggml_can_repeat
    const int nr0 = (int)(ne00/ne0);
    const int nr1 = (int)(ne01/ne1);
    const int nr2 = (int)(ne02/ne2);
    const int nr3 = (int)(ne03/ne3);

    // TODO: support for transposed / permuted tensors
    GGML_ASSERT(nb0  == sizeof(float));
    GGML_ASSERT(nb00 == sizeof(float));

    if (ggml_is_contiguous(dst)) {
        ggml_vec_set_f32(ne0*ne1*ne2*ne3, (float *)dst->data, 0);
    } else {
        for         (int k3 = 0; k3 < ne3; k3++) {
            for     (int k2 = 0; k2 < ne2; k2++) {
                for (int k1 = 0; k1 < ne1; k1++) {
                    ggml_vec_set_f32(ne0,
                        (float *) ((char *) dst->data + k1*nb1 + k2*nb2 + k3*nb3),
                        0);
                }
            }
        }
    }

    // TODO: maybe this is not optimal?
    for                         (int i3 = 0; i3 < nr3; i3++) {
        for                     (int k3 = 0; k3 < ne3; k3++) {
            for                 (int i2 = 0; i2 < nr2; i2++) {
                for             (int k2 = 0; k2 < ne2; k2++) {
                    for         (int i1 = 0; i1 < nr1; i1++) {
                        for     (int k1 = 0; k1 < ne1; k1++) {
                            for (int i0 = 0; i0 < nr0; i0++) {
                                ggml_vec_acc_f32(ne0,
                                        (float *) ((char *)  dst->data + (         k3)*nb3  + (         k2)*nb2  + (         k1)*nb1),
                                        (float *) ((char *) src0->data + (i3*ne3 + k3)*nb03 + (i2*ne2 + k2)*nb02 + (i1*ne1 + k1)*nb01 + (i0*ne0)*nb00));
                            }
                        }
                    }
                }
            }
        }
    }
}

void ggml_compute_forward_repeat_back(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_repeat_back_f32(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_concat

static void ggml_compute_forward_concat_any(
    const ggml_compute_params * params,
    ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    const size_t len = ggml_type_size(src0->type);

    const int ith = params->ith;
    const int nth = params->nth;

    GGML_TENSOR_BINARY_OP_LOCALS

    const int32_t dim = ggml_get_op_params_i32(dst, 0);

    GGML_ASSERT(dim >= 0 && dim < 4);

    int64_t o[4] = {0, 0, 0, 0};
    if (dim == 0) {
        o[dim] = src0->ne[dim]/ggml_blck_size(src0->type);
    } else {
        o[dim] = src0->ne[dim];
    }

    const char * x;

    // TODO: smarter multi-theading
    for (int i3 = 0; i3 < ne3; i3++) {
        for (int i2 = ith; i2 < ne2; i2 += nth) {
            for (int i1 = 0; i1 < ne1; i1++) {
                for (int i0 = 0; i0 < ne0/ggml_blck_size(dst->type); i0++) {
                    if (i0 < ne00/ggml_blck_size(src0->type) && i1 < ne01 && i2 < ne02 && i3 < ne03) {
                        x = (const char *)src0->data + (i0       )*nb00 + (i1       )*nb01 + (i2       )*nb02 + (i3       )*nb03;
                    } else {
                        x = (const char *)src1->data + (i0 - o[0])*nb10 + (i1 - o[1])*nb11 + (i2 - o[2])*nb12 + (i3 - o[3])*nb13;
                    }

                    char * y = (char *)dst->data + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3;

                    memcpy(y, x, len);
                }
            }
        }
    }
}

static void ggml_compute_forward_concat_i8(
    const ggml_compute_params * params,
    ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_type_size(src0->type) == sizeof(int8_t));

    const int ith = params->ith;
    const int nth = params->nth;

    GGML_TENSOR_BINARY_OP_LOCALS

    const int32_t dim = ggml_get_op_params_i32(dst, 0);

    GGML_ASSERT(dim >= 0 && dim < 4);

    int64_t o[4] = {0, 0, 0, 0};
    o[dim] = src0->ne[dim];

    const int8_t * x;

    // TODO: smarter multi-theading
    for (int i3 = 0; i3 < ne3; i3++) {
        for (int i2 = ith; i2 < ne2; i2 += nth) {
            for (int i1 = 0; i1 < ne1; i1++) {
                for (int i0 = 0; i0 < ne0; i0++) {
                    if (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
                        x = (const int8_t *) ((const char *)src0->data + (i0       )*nb00 + (i1       )*nb01 + (i2       )*nb02 + (i3       )*nb03);
                    } else {
                        x = (const int8_t *) ((const char *)src1->data + (i0 - o[0])*nb10 + (i1 - o[1])*nb11 + (i2 - o[2])*nb12 + (i3 - o[3])*nb13);
                    }

                    int8_t * y = (int8_t *)((char *)dst->data + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);

                    *y = *x;
                }
            }
        }
    }
}

static void ggml_compute_forward_concat_f16(
    const ggml_compute_params * params,
    ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_type_size(src0->type) == sizeof(ggml_fp16_t));

    const int ith = params->ith;
    const int nth = params->nth;

    GGML_TENSOR_BINARY_OP_LOCALS

    const int32_t dim = ggml_get_op_params_i32(dst, 0);

    GGML_ASSERT(dim >= 0 && dim < 4);

    int64_t o[4] = {0, 0, 0, 0};
    o[dim] = src0->ne[dim];

    const ggml_fp16_t * x;

    // TODO: smarter multi-theading
    for (int i3 = 0; i3 < ne3; i3++) {
        for (int i2 = ith; i2 < ne2; i2 += nth) {
            for (int i1 = 0; i1 < ne1; i1++) {
                for (int i0 = 0; i0 < ne0; i0++) {
                    if (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
                        x = (const ggml_fp16_t *) ((const char *)src0->data + (i0       )*nb00 + (i1       )*nb01 + (i2       )*nb02 + (i3       )*nb03);
                    } else {
                        x = (const ggml_fp16_t *) ((const char *)src1->data + (i0 - o[0])*nb10 + (i1 - o[1])*nb11 + (i2 - o[2])*nb12 + (i3 - o[3])*nb13);
                    }

                    ggml_fp16_t * y = (ggml_fp16_t *)((char *)dst->data + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);

                    *y = *x;
                }
            }
        }
    }
}

static void ggml_compute_forward_concat_f32(
    const ggml_compute_params * params,
    ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_type_size(src0->type) == sizeof(float));

    const int ith = params->ith;
    const int nth = params->nth;

    GGML_TENSOR_BINARY_OP_LOCALS

    const int32_t dim = ggml_get_op_params_i32(dst, 0);

    GGML_ASSERT(dim >= 0 && dim < 4);

    int64_t o[4] = {0, 0, 0, 0};
    o[dim] = src0->ne[dim];

    const float * x;

    // TODO: smarter multi-theading
    for (int i3 = 0; i3 < ne3; i3++) {
        for (int i2 = ith; i2 < ne2; i2 += nth) {
            for (int i1 = 0; i1 < ne1; i1++) {
                for (int i0 = 0; i0 < ne0; i0++) {
                    if (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
                        x = (const float *) ((const char *)src0->data + (i0       )*nb00 + (i1       )*nb01 + (i2       )*nb02 + (i3       )*nb03);
                    } else {
                        x = (const float *) ((const char *)src1->data + (i0 - o[0])*nb10 + (i1 - o[1])*nb11 + (i2 - o[2])*nb12 + (i3 - o[3])*nb13);
                    }

                    float * y = (float *)((char *)dst->data + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);

                    *y = *x;
                }
            }
        }
    }
}

void ggml_compute_forward_concat(
    const ggml_compute_params * params,
    ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    if (ggml_is_quantized(src0->type)) {
        GGML_ASSERT(ggml_is_contiguous_rows(src0));
        GGML_ASSERT(ggml_is_contiguous_rows(src1));
        GGML_ASSERT(src0->ne[0] % ggml_blck_size(src0->type) == 0);
        GGML_ASSERT(src1->ne[0] % ggml_blck_size(src1->type) == 0);
    }

    switch (src0->type) {
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
        case GGML_TYPE_I16:
            {
                ggml_compute_forward_concat_f16(params, dst);
            } break;
        case GGML_TYPE_I8:
            {
                ggml_compute_forward_concat_i8(params, dst);
            } break;
        case GGML_TYPE_F32:
        case GGML_TYPE_I32:
            {
                ggml_compute_forward_concat_f32(params, dst);
            } break;
        default:
            {
                ggml_compute_forward_concat_any(params, dst);
            }
    }
}

// ggml_compute_forward_gelu

static void ggml_compute_forward_gelu_f32(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    assert(ggml_is_contiguous_rows(src0));
    assert(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    const int ith = params->ith;
    const int nth = params->nth;

    const int nc = src0->ne[0];
    const int nr = ggml_nrows(src0);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int ir = ir0; ir < ir1; ++ir) {
        const int i3 = ir/(ne02*ne01);
        const int i2 = (ir - i3*ne02*ne01)/ne01;
        const int i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_gelu_f32(nc,
                (float *) ((char *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1),
                (float *) ((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01));

#ifndef NDEBUG
        for (int k = 0; k < nc; k++) {
            const float x = ((float *) ((char *) dst->data + i3*nb3 + i2*nb2 + i1*(dst->nb[1])))[k];
            GGML_UNUSED(x);
            assert(!isnan(x));
            assert(!isinf(x));
        }
#endif // NDEBUG
    }
}

static void ggml_compute_forward_gelu_f16(
    const ggml_compute_params * params,
    ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    assert(ggml_is_contiguous_rows(src0));
    assert(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    const int ith = params->ith;
    const int nth = params->nth;

    const int nc = src0->ne[0];
    const int nr = ggml_nrows(src0);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int ir = ir0; ir < ir1; ++ir) {
        const int i3 = ir/(ne02*ne01);
        const int i2 = (ir - i3*ne02*ne01)/ne01;
        const int i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_gelu_f16(nc,
                (ggml_fp16_t *) ((char *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1),
                (ggml_fp16_t *) ((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01));

#ifndef NDEBUG
        for (int k = 0; k < nc; k++) {
            const ggml_fp16_t x = ((ggml_fp16_t *) ((char *) dst->data + i3*nb3 + i2*nb2 + i1*( dst->nb[1])))[k];
            const float v = GGML_CPU_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            assert(!isnan(v));
            assert(!isinf(v));
        }
#endif // NDEBUG
    }
}

static void ggml_compute_forward_gelu(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_gelu_f32(params, dst);
            } break;
        case GGML_TYPE_F16:
            {
                ggml_compute_forward_gelu_f16(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_fill

static void ggml_compute_forward_fill_f32(const ggml_compute_params * params, ggml_tensor * dst) {
    const float c = ggml_get_op_params_f32(dst, 0);

    GGML_TENSOR_LOCALS(int64_t, ne, dst, ne);
    GGML_TENSOR_LOCALS(size_t,  nb, dst, nb);

    const auto [ir0, ir1] = get_thread_range(params, dst);

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t i03 = ir/(ne2*ne1);
        const int64_t i02 = (ir - i03*ne2*ne1)/ne1;
        const int64_t i01 = (ir - i03*ne2*ne1 - i02*ne1);

        float * dst_ptr  = (float *) ((char *) dst->data + i03*nb3 + i02*nb2 + i01*nb1);

        ggml_vec_set_f32(ne0, dst_ptr, c);
    }
}

static void ggml_compute_forward_fill_f16(const ggml_compute_params * params, ggml_tensor * dst) {
    const ggml_fp16_t c = GGML_CPU_FP32_TO_FP16(ggml_get_op_params_f32(dst, 0));

    GGML_TENSOR_LOCALS(int64_t, ne, dst, ne);
    GGML_TENSOR_LOCALS(size_t,  nb, dst, nb);

    const auto [ir0, ir1] = get_thread_range(params, dst);

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t i03 = ir/(ne2*ne1);
        const int64_t i02 = (ir - i03*ne2*ne1)/ne1;
        const int64_t i01 = (ir - i03*ne2*ne1 - i02*ne1);

        ggml_fp16_t * dst_ptr  = (ggml_fp16_t *) ((char *) dst->data + i03*nb3 + i02*nb2 + i01*nb1);

        ggml_vec_set_f16(ne0, dst_ptr, c);
    }
}

void ggml_compute_forward_fill(const ggml_compute_params * params, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_fill_f32(params, dst);
            } break;
        case GGML_TYPE_F16:
            {
                ggml_compute_forward_fill_f16(params, dst);
            } break;
        default:
            {
                GGML_ABORT("unsupported type for ggml_compute_forward_fill: %s", ggml_type_name(src0->type));
            }
    }
}

// ggml_compute_tri

static void ggml_compute_forward_tri_f32(const ggml_compute_params * params, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    const ggml_tri_type ttype = (ggml_tri_type) ggml_get_op_params_i32(dst, 0);

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_TENSOR_UNARY_OP_LOCALS

    const auto [ir0, ir1] = get_thread_range(params, src0);

    bool (*bipred)(int, int);

    switch (ttype) {
        case GGML_TRI_TYPE_LOWER:      bipred = [](int i, int r) { return i <  r; }; break;
        case GGML_TRI_TYPE_LOWER_DIAG: bipred = [](int i, int r) { return i <= r; }; break;
        case GGML_TRI_TYPE_UPPER:      bipred = [](int i, int r) { return i >  r; }; break;
        case GGML_TRI_TYPE_UPPER_DIAG: bipred = [](int i, int r) { return i >= r; }; break;
        default: GGML_ABORT("invalid tri type");
    }

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t i03 = ir/(ne02*ne01);
        const int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        const int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        const float * src_ptr = (const float  *) ((const char *) src0->data + i03*nb03 + i02*nb02 + i01*nb01);
              float * dst_ptr = (      float  *) ((      char *) dst->data  + i03*nb3  + i02*nb2  + i01*nb1);

        for (int i0 = 0; i0 < ne0; ++i0) {
            dst_ptr[i0] = bipred(i0, i01) ? src_ptr[i0] : 0.0f;
        }
    }
}

void ggml_compute_forward_tri(const ggml_compute_params * params, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_tri_f32(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_gelu_erf

static void ggml_compute_forward_gelu_erf_f32(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    assert(ggml_is_contiguous_rows(src0));
    assert(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    const int ith = params->ith;
    const int nth = params->nth;

    const int nc = src0->ne[0];
    const int nr = ggml_nrows(src0);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int ir = ir0; ir < ir1; ++ir) {
        const int i3 = ir/(ne02*ne01);
        const int i2 = (ir - i3*ne02*ne01)/ne01;
        const int i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_gelu_erf_f32(nc,
                (float *) ((char *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1),
                (float *) ((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01));

#ifndef NDEBUG
        for (int k = 0; k < nc; k++) {
            const float x = ((float *) ((char *) dst->data + i3*nb3 + i2*nb2 + i1*(dst->nb[1])))[k];
            GGML_UNUSED(x);
            assert(!isnan(x));
            assert(!isinf(x));
        }
#endif // NDEBUG
    }
}

static void ggml_compute_forward_gelu_erf_f16(
    const ggml_compute_params * params,
    ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    assert(ggml_is_contiguous_rows(src0));
    assert(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    const int ith = params->ith;
    const int nth = params->nth;

    const int nc = src0->ne[0];
    const int nr = ggml_nrows(src0);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int ir = ir0; ir < ir1; ++ir) {
        const int i3 = ir/(ne02*ne01);
        const int i2 = (ir - i3*ne02*ne01)/ne01;
        const int i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_gelu_erf_f16(nc,
                (ggml_fp16_t *) ((char *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1),
                (ggml_fp16_t *) ((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01));

#ifndef NDEBUG
        for (int k = 0; k < nc; k++) {
            const ggml_fp16_t x = ((ggml_fp16_t *) ((char *) dst->data + i3*nb3 + i2*nb2 + i1*( dst->nb[1])))[k];
            const float v = GGML_CPU_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            assert(!isnan(v));
            assert(!isinf(v));
        }
#endif // NDEBUG
    }
}

static void ggml_compute_forward_gelu_erf(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_gelu_erf_f32(params, dst);
            } break;
        case GGML_TYPE_F16:
            {
                ggml_compute_forward_gelu_erf_f16(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_gelu_quick

static void ggml_compute_forward_gelu_quick_f32(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    assert(ggml_is_contiguous_rows(src0));
    assert(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    const int ith = params->ith;
    const int nth = params->nth;

    const int nc = src0->ne[0];
    const int nr = ggml_nrows(src0);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int ir = ir0; ir < ir1; ++ir) {
        const int i3 = ir/(ne02*ne01);
        const int i2 = (ir - i3*ne02*ne01)/ne01;
        const int i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_gelu_quick_f32(nc,
                (float *) ((char *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1),
                (float *) ((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01));

#ifndef NDEBUG
        for (int k = 0; k < nc; k++) {
            const float x = ((float *) ((char *) dst->data + i3*nb3 + i2*nb2 + i1*(dst->nb[1])))[k];
            GGML_UNUSED(x);
            assert(!isnan(x));
            assert(!isinf(x));
        }
#endif // NDEBUG
    }
}

static void ggml_compute_forward_gelu_quick_f16(
    const ggml_compute_params * params,
    ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    assert(ggml_is_contiguous_rows(src0));
    assert(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    const int ith = params->ith;
    const int nth = params->nth;

    const int nc = src0->ne[0];
    const int nr = ggml_nrows(src0);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int ir = ir0; ir < ir1; ++ir) {
        const int i3 = ir/(ne02*ne01);
        const int i2 = (ir - i3*ne02*ne01)/ne01;
        const int i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_gelu_quick_f16(nc,
                (ggml_fp16_t *) ((char *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1),
                (ggml_fp16_t *) ((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01));

#ifndef NDEBUG
        for (int k = 0; k < nc; k++) {
            const ggml_fp16_t x = ((ggml_fp16_t *) ((char *) dst->data + i3*nb3 + i2*nb2 + i1*( dst->nb[1])))[k];
            const float v = GGML_CPU_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            assert(!isnan(v));
            assert(!isinf(v));
        }
#endif // NDEBUG
    }
}

static void ggml_compute_forward_gelu_quick(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_gelu_quick_f32(params, dst);
            } break;
        case GGML_TYPE_F16:
            {
                ggml_compute_forward_gelu_quick_f16(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_silu

static void ggml_compute_forward_silu_f32(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    assert(ggml_is_contiguous_rows(src0));
    assert(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    const int ith = params->ith;
    const int nth = params->nth;

    const int nc = src0->ne[0];
    const int nr = ggml_nrows(src0);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int ir = ir0; ir < ir1; ++ir) {
        const int i3 = ir/(ne02*ne01);
        const int i2 = (ir - i3*ne02*ne01)/ne01;
        const int i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_silu_f32(nc,
                (float *) ((char *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1),
                (float *) ((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01));

#ifndef NDEBUG
        for (int k = 0; k < nc; k++) {
            const float x = ((float *) ((char *) dst->data + i3*nb3 + i2*nb2 + i1*(dst->nb[1])))[k];
            GGML_UNUSED(x);
            assert(!isnan(x));
            assert(!isinf(x));
        }
#endif // NDEBUG
    }
}

static void ggml_compute_forward_silu_f16(
    const ggml_compute_params * params,
    ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    assert(ggml_is_contiguous_rows(src0));
    assert(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    const int ith = params->ith;
    const int nth = params->nth;

    const int nc = src0->ne[0];
    const int nr = ggml_nrows(src0);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int ir = ir0; ir < ir1; ++ir) {
        const int i3 = ir/(ne02*ne01);
        const int i2 = (ir - i3*ne02*ne01)/ne01;
        const int i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_silu_f16(nc,
                (ggml_fp16_t *) ((char *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1),
                (ggml_fp16_t *) ((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01));

#ifndef NDEBUG
        for (int k = 0; k < nc; k++) {
            const ggml_fp16_t x = ((ggml_fp16_t *) ((char *) dst->data + i3*nb3 + i2*nb2 + i1*( dst->nb[1])))[k];
            const float v = GGML_CPU_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            assert(!isnan(v));
            assert(!isinf(v));
        }
#endif // NDEBUG
    }
}

static void ggml_compute_forward_silu(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_silu_f32(params, dst);
            } break;
        case GGML_TYPE_F16:
            {
                ggml_compute_forward_silu_f16(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}
// ggml_compute_forward_leaky_relu

static void ggml_compute_forward_leaky_relu_f32(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    if (params->ith != 0) {
        return;
    }

    assert(ggml_is_contiguous_1(src0));
    assert(ggml_is_contiguous_1(dst));
    assert(ggml_are_same_shape(src0, dst));

    const int n  = ggml_nrows(src0);
    const int nc = src0->ne[0];

    float negative_slope;
    memcpy(&negative_slope, dst->op_params, sizeof(float));

    assert(dst->nb[0]  == sizeof(float));
    assert(src0->nb[0] == sizeof(float));

    for (int i = 0; i < n; i++) {
        ggml_vec_leaky_relu_f32(nc,
                (float *) ((char *) dst->data  + i*( dst->nb[1])),
                (float *) ((char *) src0->data + i*(src0->nb[1])), negative_slope);
    }
}

static void ggml_compute_forward_leaky_relu_f16(
    const ggml_compute_params * params,
    ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    if (params->ith != 0) {
        return;
    }

    assert(ggml_is_contiguous_1(src0));
    assert(ggml_is_contiguous_1(dst));
    assert(ggml_are_same_shape(src0, dst));

    const int n  = ggml_nrows(src0);
    const int nc = src0->ne[0];

    float negative_slope;
    memcpy(&negative_slope, dst->op_params, sizeof(float));

    assert(dst->nb[0]  == sizeof(ggml_fp16_t));
    assert(src0->nb[0] == sizeof(ggml_fp16_t));

    for (int i = 0; i < n; i++) {
        ggml_vec_leaky_relu_f16(nc,
                (ggml_fp16_t *) ((char *) dst->data  + i*( dst->nb[1])),
                (ggml_fp16_t *) ((char *) src0->data + i*(src0->nb[1])), negative_slope);
    }
}

void ggml_compute_forward_leaky_relu(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_leaky_relu_f32(params, dst);
            } break;
        case GGML_TYPE_F16:
            {
                ggml_compute_forward_leaky_relu_f16(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_silu_back

static void ggml_compute_forward_silu_back_f32(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * grad = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    assert(ggml_is_contiguous_1(grad));
    assert(ggml_is_contiguous_1(src1));
    assert(ggml_is_contiguous_1(dst));
    assert(ggml_are_same_shape(src1, dst));
    assert(ggml_are_same_shape(src1, grad));

    const int ith = params->ith;
    const int nth = params->nth;

    const int nc = src1->ne[0];
    const int nr = ggml_nrows(src1);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int i1 = ir0; i1 < ir1; i1++) {
        ggml_vec_silu_backward_f32(nc,
                (float *) ((char *) dst->data  + i1*( dst->nb[1])),
                (float *) ((char *) src1->data + i1*(src1->nb[1])),
                (float *) ((char *) grad->data + i1*(grad->nb[1])));

#ifndef NDEBUG
        for (int k = 0; k < nc; k++) {
            const float x = ((float *) ((char *) dst->data + i1*( dst->nb[1])))[k];
            GGML_UNUSED(x);
            assert(!isnan(x));
            assert(!isinf(x));
        }
#endif // NDEBUG
    }
}

static void ggml_compute_forward_silu_back_f16(
    const ggml_compute_params * params,
    ggml_tensor * dst) {

    const ggml_tensor * grad = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    assert(ggml_is_contiguous_1(grad));
    assert(ggml_is_contiguous_1(src1));
    assert(ggml_is_contiguous_1(dst));
    assert(ggml_are_same_shape(src1, dst));
    assert(ggml_are_same_shape(src1, grad));

    const int ith = params->ith;
    const int nth = params->nth;

    const int nc = src1->ne[0];
    const int nr = ggml_nrows(src1);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int i1 = ir0; i1 < ir1; i1++) {
        ggml_vec_silu_backward_f16(nc,
                (ggml_fp16_t *) ((char *) dst->data  + i1*( dst->nb[1])),
                (ggml_fp16_t *) ((char *) src1->data + i1*(src1->nb[1])),
                (ggml_fp16_t *) ((char *) grad->data + i1*(grad->nb[1])));

#ifndef NDEBUG
        for (int k = 0; k < nc; k++) {
            const float x = ((ggml_fp16_t *) ((char *) dst->data + i1*( dst->nb[1])))[k];
            const float v = GGML_CPU_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            assert(!isnan(v));
            assert(!isinf(v));
        }
#endif // NDEBUG
    }
}

void ggml_compute_forward_silu_back(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_silu_back_f32(params, dst);
            } break;
        case GGML_TYPE_F16:
            {
                ggml_compute_forward_silu_back_f16(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_reglu

static void ggml_compute_forward_reglu_f32(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    char * src0_d = (char *) src0->data;
    char * src1_d = (char *) (src1 ? src1->data : src0->data);
    const size_t src0_o = src0->nb[1];
    const size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    if (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    const int ith = params->ith;
    const int nth = params->nth;

    const int nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    const int nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    const int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int i1 = ir0; i1 < ir1; i1++) {
        float * src0_p = (float *) (src0_d + i1*src0_o);
        float * src1_p = (float *) (src1_d + i1*src1_o);

        if (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_reglu_f32(nc, (float *) ((char *) dst->data + i1*(dst->nb[1])), src0_p, src1_p);

#ifndef NDEBUG
        for (int k = 0; k < nc; k++) {
            const float x = ((float *) ((char *) dst->data + i1*( dst->nb[1])))[k];
            GGML_UNUSED(x);
            assert(!isnan(x));
            assert(!isinf(x));
        }
#endif // NDEBUG
    }
}

static void ggml_compute_forward_reglu_f16(
    const ggml_compute_params * params,
    ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    char * src0_d = (char *) src0->data;
    char * src1_d = (char *) (src1 ? src1->data : src0->data);
    const size_t src0_o = src0->nb[1];
    const size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    if (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    const int ith = params->ith;
    const int nth = params->nth;

    const int nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    const int nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    const int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1*src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1*src1_o);

        if (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_reglu_f16(nc, (ggml_fp16_t *) ((char *) dst->data + i1*(dst->nb[1])), src0_p, src1_p);

#ifndef NDEBUG
        for (int k = 0; k < nc; k++) {
            const ggml_fp16_t x = ((ggml_fp16_t *) ((char *) dst->data + i1*( dst->nb[1])))[k];
            const float v = GGML_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            assert(!isnan(v));
            assert(!isinf(v));
        }
#endif // NDEBUG
    }
}

static void ggml_compute_forward_reglu(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_reglu_f32(params, dst);
            } break;
        case GGML_TYPE_F16:
            {
                ggml_compute_forward_reglu_f16(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_geglu

static void ggml_compute_forward_geglu_f32(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    char * src0_d = (char *) src0->data;
    char * src1_d = (char *) (src1 ? src1->data : src0->data);
    const size_t src0_o = src0->nb[1];
    const size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    if (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    const int ith = params->ith;
    const int nth = params->nth;

    const int nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    const int nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    const int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int i1 = ir0; i1 < ir1; i1++) {
        float * src0_p = (float *) (src0_d + i1*src0_o);
        float * src1_p = (float *) (src1_d + i1*src1_o);

        if (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_geglu_f32(nc, (float *) ((char *) dst->data + i1*(dst->nb[1])), src0_p, src1_p);

#ifndef NDEBUG
        for (int k = 0; k < nc; k++) {
            const float x = ((float *) ((char *) dst->data + i1*( dst->nb[1])))[k];
            GGML_UNUSED(x);
            assert(!isnan(x));
            assert(!isinf(x));
        }
#endif // NDEBUG
    }
}

static void ggml_compute_forward_geglu_f16(
    const ggml_compute_params * params,
    ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    char * src0_d = (char *) src0->data;
    char * src1_d = (char *) (src1 ? src1->data : src0->data);
    const size_t src0_o = src0->nb[1];
    const size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    if (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    const int ith = params->ith;
    const int nth = params->nth;

    const int nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    const int nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    const int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1*src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1*src1_o);

        if (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_geglu_f16(nc, (ggml_fp16_t *) ((char *) dst->data + i1*(dst->nb[1])), src0_p, src1_p);

#ifndef NDEBUG
        for (int k = 0; k < nc; k++) {
            const ggml_fp16_t x = ((ggml_fp16_t *) ((char *) dst->data + i1*( dst->nb[1])))[k];
            const float v = GGML_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            assert(!isnan(v));
            assert(!isinf(v));
        }
#endif // NDEBUG
    }
}

static void ggml_compute_forward_geglu(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_geglu_f32(params, dst);
            } break;
        case GGML_TYPE_F16:
            {
                ggml_compute_forward_geglu_f16(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_swiglu

static void ggml_compute_forward_swiglu_f32(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    char * src0_d = (char *) src0->data;
    char * src1_d = (char *) (src1 ? src1->data : src0->data);
    const size_t src0_o = src0->nb[1];
    const size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    if (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    const int ith = params->ith;
    const int nth = params->nth;

    const int nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    const int nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    const int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int i1 = ir0; i1 < ir1; i1++) {
        float * src0_p = (float *) (src0_d + i1*src0_o);
        float * src1_p = (float *) (src1_d + i1*src1_o);

        if (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_swiglu_f32(nc, (float *) ((char *) dst->data + i1*(dst->nb[1])), src0_p, src1_p);

#ifndef NDEBUG
        for (int k = 0; k < nc; k++) {
            const float x = ((float *) ((char *) dst->data + i1*( dst->nb[1])))[k];
            GGML_UNUSED(x);
            assert(!isnan(x));
            assert(!isinf(x));
        }
#endif // NDEBUG
    }
}

static void ggml_compute_forward_swiglu_f16(
    const ggml_compute_params * params,
    ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    char * src0_d = (char *) src0->data;
    char * src1_d = (char *) (src1 ? src1->data : src0->data);
    const size_t src0_o = src0->nb[1];
    const size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    if (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    const int ith = params->ith;
    const int nth = params->nth;

    const int nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    const int nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    const int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1*src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1*src1_o);

        if (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_swiglu_f16(nc, (ggml_fp16_t *) ((char *) dst->data + i1*(dst->nb[1])), src0_p, src1_p);

#ifndef NDEBUG
        for (int k = 0; k < nc; k++) {
            const ggml_fp16_t x = ((ggml_fp16_t *) ((char *) dst->data + i1*( dst->nb[1])))[k];
            const float v = GGML_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            assert(!isnan(v));
            assert(!isinf(v));
        }
#endif // NDEBUG
    }
}

static void ggml_compute_forward_swiglu(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_swiglu_f32(params, dst);
            } break;
        case GGML_TYPE_F16:
            {
                ggml_compute_forward_swiglu_f16(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_swiglu_oai

static void ggml_compute_forward_swiglu_oai_f32(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    char * src0_d = (char *) src0->data;
    char * src1_d = (char *) (src1 ? src1->data : src0->data);
    const size_t src0_o = src0->nb[1];
    const size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    if (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    const int ith = params->ith;
    const int nth = params->nth;

    const int nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    const int nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    const int32_t swapped = ggml_get_op_params_i32(dst, 1);
    const float alpha = ggml_get_op_params_f32(dst, 2);
    const float limit = ggml_get_op_params_f32(dst, 3);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int i1 = ir0; i1 < ir1; i1++) {
        float * src0_p = (float *) (src0_d + i1*src0_o);
        float * src1_p = (float *) (src1_d + i1*src1_o);
        float * dst_p  = (float *) ((char *) dst->data + i1*(dst->nb[1]));

        if (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        for (int k = 0; k < nc; k++) {
            const float x = std::min(src0_p[k], limit);
            const float y = std::clamp(src1_p[k], -limit, limit);
            const float out_glu = x / (1.f + expf(alpha * (-x)));
            dst_p[k] = out_glu * (y + 1.f);
        }

#ifndef NDEBUG
        for (int k = 0; k < nc; k++) {
            const float x = dst_p[k];
            GGML_UNUSED(x);
            assert(!isnan(x));
            assert(!isinf(x));
        }
#endif // NDEBUG
    }
}

static void ggml_compute_forward_swiglu_oai(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_swiglu_oai_f32(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_geglu_erf

static void ggml_compute_forward_geglu_erf_f32(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    char * src0_d = (char *) src0->data;
    char * src1_d = (char *) (src1 ? src1->data : src0->data);
    const size_t src0_o = src0->nb[1];
    const size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    if (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    const int ith = params->ith;
    const int nth = params->nth;

    const int nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    const int nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    const int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int i1 = ir0; i1 < ir1; i1++) {
        float * src0_p = (float *) (src0_d + i1*src0_o);
        float * src1_p = (float *) (src1_d + i1*src1_o);

        if (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_geglu_erf_f32(nc, (float *) ((char *) dst->data + i1*(dst->nb[1])), src0_p, src1_p);

#ifndef NDEBUG
        for (int k = 0; k < nc; k++) {
            const float x = ((float *) ((char *) dst->data + i1*( dst->nb[1])))[k];
            GGML_UNUSED(x);
            assert(!isnan(x));
            assert(!isinf(x));
        }
#endif // NDEBUG
    }
}

static void ggml_compute_forward_geglu_erf_f16(
    const ggml_compute_params * params,
    ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    char * src0_d = (char *) src0->data;
    char * src1_d = (char *) (src1 ? src1->data : src0->data);
    const size_t src0_o = src0->nb[1];
    const size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    if (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    const int ith = params->ith;
    const int nth = params->nth;

    const int nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    const int nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    const int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1*src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1*src1_o);

        if (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_geglu_erf_f16(nc, (ggml_fp16_t *) ((char *) dst->data + i1*(dst->nb[1])), src0_p, src1_p);

#ifndef NDEBUG
        for (int k = 0; k < nc; k++) {
            const ggml_fp16_t x = ((ggml_fp16_t *) ((char *) dst->data + i1*( dst->nb[1])))[k];
            const float v = GGML_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            assert(!isnan(v));
            assert(!isinf(v));
        }
#endif // NDEBUG
    }
}

static void ggml_compute_forward_geglu_erf(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_geglu_erf_f32(params, dst);
            } break;
        case GGML_TYPE_F16:
            {
                ggml_compute_forward_geglu_erf_f16(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_geglu_quick

static void ggml_compute_forward_geglu_quick_f32(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    char * src0_d = (char *) src0->data;
    char * src1_d = (char *) (src1 ? src1->data : src0->data);
    const size_t src0_o = src0->nb[1];
    const size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    if (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    const int ith = params->ith;
    const int nth = params->nth;

    const int nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    const int nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    const int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int i1 = ir0; i1 < ir1; i1++) {
        float * src0_p = (float *) (src0_d + i1*src0_o);
        float * src1_p = (float *) (src1_d + i1*src1_o);

        if (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_geglu_quick_f32(nc, (float *) ((char *) dst->data + i1*(dst->nb[1])), src0_p, src1_p);

#ifndef NDEBUG
        for (int k = 0; k < nc; k++) {
            const float x = ((float *) ((char *) dst->data + i1*( dst->nb[1])))[k];
            GGML_UNUSED(x);
            assert(!isnan(x));
            assert(!isinf(x));
        }
#endif // NDEBUG
    }
}

static void ggml_compute_forward_geglu_quick_f16(
    const ggml_compute_params * params,
    ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    char * src0_d = (char *) src0->data;
    char * src1_d = (char *) (src1 ? src1->data : src0->data);
    const size_t src0_o = src0->nb[1];
    const size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    if (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    const int ith = params->ith;
    const int nth = params->nth;

    const int nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    const int nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    const int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1*src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1*src1_o);

        if (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_geglu_quick_f16(nc, (ggml_fp16_t *) ((char *) dst->data + i1*(dst->nb[1])), src0_p, src1_p);

#ifndef NDEBUG
        for (int k = 0; k < nc; k++) {
            const ggml_fp16_t x = ((ggml_fp16_t *) ((char *) dst->data + i1*( dst->nb[1])))[k];
            const float v = GGML_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            assert(!isnan(v));
            assert(!isinf(v));
        }
#endif // NDEBUG
    }
}

static void ggml_compute_forward_geglu_quick(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_geglu_quick_f32(params, dst);
            } break;
        case GGML_TYPE_F16:
            {
                ggml_compute_forward_geglu_quick_f16(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_norm

static void ggml_compute_forward_norm_f32(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    const int ith = params->ith;
    const int nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));

    GGML_ASSERT(eps >= 0.0f);

    for (int64_t i03 = 0; i03 < ne03; i03++) {
        for (int64_t i02 = 0; i02 < ne02; i02++) {
            for (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                const char * x = (const char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03;
                char * y = (char *) dst->data + i01*nb1 + i02*nb2 + i03*nb3;

                if (nb00 == sizeof(float) && nb0 == sizeof(float)) {
                    const float * xf = (const float *) x;

                    float sum = 0.0;
                    ggml_vec_sum_f32(ne00, &sum, xf);
                    float mean = sum/ne00;

                    float * yf = (float *) y;
                    float variance = 0;

#ifdef GGML_USE_ACCELERATE
                    mean = -mean;
                    vDSP_vsadd(xf, 1, &mean, yf, 1, ne00);
                    vDSP_measqv(yf, 1, &variance, ne00);
#else
                    variance = ggml_vec_cvar_f32(ne00, yf, xf, mean);
#endif //GGML_USE_ACCELERATE

                    const float scale = 1.0f/sqrtf(variance + eps);
                    ggml_vec_scale_f32(ne00, yf, scale);
                } else {
                    float sum = 0.0;
                    for (int64_t i00 = 0; i00 < ne00; i00++) {
                        sum += *(const float *) (x + i00*nb00);
                    }
                    const float mean = sum/ne00;

                    float variance = 0.0f;
                    for (int64_t i00 = 0; i00 < ne00; i00++) {
                        const float v = *(const float *) (x + i00*nb00) - mean;
                        *(float *) (y + i00*nb0) = v;
                        variance += v * v;
                    }
                    variance /= ne00;

                    const float scale = 1.0f/sqrtf(variance + eps);
                    for (int64_t i00 = 0; i00 < ne00; i00++) {
                        *(float *) (y + i00*nb0) *= scale;
                    }
                }
            }
        }
    }
}

void ggml_compute_forward_norm(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_norm_f32(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_group_rms_norm

// fusion kinds that can be combined with the rms_norm computation in a single pass.
// extend this enum when adding new fused variants (e.g. FUSE_ADD, FUSE_MUL_ADD, ...).
enum ggml_rms_norm_fuse_op {
    GGML_RMS_NORM_FUSE_OP_NONE,
    GGML_RMS_NORM_FUSE_OP_MUL,
};

template <ggml_rms_norm_fuse_op FUSE_OP>
static void ggml_compute_forward_rms_norm_f32(
        const ggml_compute_params * params,
        ggml_tensor * dst_rms_norm,
        ggml_tensor * dst_fused = nullptr) {

    const ggml_tensor * src0 = dst_rms_norm->src[0];
    const ggml_tensor * src1 = nullptr;
    ggml_tensor       * dst  = dst_rms_norm;

    if constexpr (FUSE_OP == GGML_RMS_NORM_FUSE_OP_MUL) {
        src1 = (dst_fused->src[0] == dst_rms_norm) ? dst_fused->src[1] : dst_fused->src[0];
        dst  = dst_fused;
    }

    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    GGML_ASSERT(src0->nb[0] == sizeof(float));

    const int ith = params->ith;
    const int nth = params->nth;

    GGML_TENSOR_BINARY_OP_LOCALS

    float eps;
    memcpy(&eps, dst_rms_norm->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    // TODO: optimize
    for (int64_t i03 = 0; i03 < ne03; i03++) {
        for (int64_t i02 = 0; i02 < ne02; i02++) {
            for (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                const float * x = (float *) ((char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03);

                ggml_float sum = 0.0;
                // worth switching to explicit SIMD?
                for (int64_t i00 = 0; i00 < ne00; i00++) {
                    sum += (ggml_float)(x[i00] * x[i00]);
                }

                const float mean  = sum/ne00;
                const float scale = 1.0f/sqrtf(mean + eps);

                // if you hit this, likely you got an inf somewhere earlier
                assert(scale > 0.0f);

                float * y = (float *) ((char *) dst->data + i01*nb1 + i02*nb2 + i03*nb3);

                if constexpr (FUSE_OP == GGML_RMS_NORM_FUSE_OP_MUL) {
                    const int64_t i11 = i01 % ne11;
                    const int64_t i12 = i02 % ne12;
                    const int64_t i13 = i03 % ne13;
                    const float * w = (float *) ((char *) src1->data + i11*nb11 + i12*nb12 + i13*nb13);

                    for (int64_t i00 = 0; i00 < ne00; i00++) {
                        y[i00] = x[i00] * scale * w[i00];
                    }
                } else {
                    memcpy(y, x, ne00 * sizeof(float));
                    ggml_vec_scale_f32(ne00, y, scale);
                }
            }
        }
    }
}

void ggml_compute_forward_rms_norm(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_rms_norm_f32<GGML_RMS_NORM_FUSE_OP_NONE>(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// Fused RMS_NORM + MUL: computes dst = rms_norm(src0) * src1 in a single pass.
// This avoids materializing the intermediate rms_norm result in memory.
void ggml_compute_forward_rms_norm_mul_fused(
        const ggml_compute_params * params,
        ggml_tensor * dst_rms_norm,
        ggml_tensor * dst_mul) {

    GGML_ASSERT(dst_mul != nullptr);
    GGML_ASSERT(dst_mul->src[0] == dst_rms_norm || dst_mul->src[1] == dst_rms_norm);

    const ggml_tensor * src0 = dst_rms_norm->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_rms_norm_f32<GGML_RMS_NORM_FUSE_OP_MUL>(params, dst_rms_norm, dst_mul);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

static void ggml_compute_forward_rms_norm_back_f32(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0]; // gradients from forward pass output
    const ggml_tensor * src1 = dst->src[1]; // src1 from forward pass

    GGML_ASSERT(ggml_are_same_shape(src0, dst) && ggml_are_same_shape(src0, src1));

    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(src1->nb[0] == sizeof(float));

    const int ith = params->ith;
    const int nth = params->nth;

    GGML_TENSOR_BINARY_OP_LOCALS

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));

    // TODO: optimize
    for (int64_t i03 = 0; i03 < ne03; i03++) {
        for (int64_t i02 = 0; i02 < ne02; i02++) {
            for (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                // src1 is same shape as src0 => same indices
                const int64_t i11 = i01;
                const int64_t i12 = i02;
                const int64_t i13 = i03;

                const float * dz = (float *) ((char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03);
                const float * x  = (float *) ((char *) src1->data + i11*nb11 + i12*nb12 + i13*nb13);

                ggml_float sum_xx  = 0.0;
                ggml_float sum_xdz = 0.0;

                for (int64_t i00 = 0; i00 < ne00; i00++) {
                    sum_xx  += (ggml_float)(x[i00] * x[i00]);
                    sum_xdz += (ggml_float)(x[i00] * dz[i00]);
                }

                //const float mean     = (float)(sum_xx)/ne00;
                const float mean_eps = (float)(sum_xx)/ne00 + eps;
                const float sum_eps  = (float)(sum_xx) + eps*ne00;
                //const float mean_xdz = (float)(sum_xdz)/ne00;
                // we could cache rms from forward pass to improve performance.
                // to do this implement ggml_rms and compose ggml_rms_norm using ggml_rms.
                //const float rms      = sqrtf(mean_eps);
                const float rrms     = 1.0f / sqrtf(mean_eps);
                //const float scale    = -rrms/(ne00 * mean_eps); // -1/(n*rms**3)

                {
                    // z = rms_norm(x)
                    //
                    // rms_norm(src1) =
                    //     scale(
                    //         src1,
                    //         div(
                    //             1,
                    //             sqrt(
                    //                 add(
                    //                     scale(
                    //                         sum(
                    //                             sqr(
                    //                                 src1)),
                    //                         (1.0/N)),
                    //                     eps))));

                    // postorder:
                    // ## op    args         grad
                    // 00 param src1         grad[#00]
                    // 01 const 1
                    // 02 sqr   (#00)        grad[#02]
                    // 03 sum   (#02)        grad[#03]
                    // 04 const 1/N
                    // 05 scale (#03, #04)   grad[#05]
                    // 06 const eps
                    // 07 add   (#05, #06)   grad[#07]
                    // 08 sqrt  (#07)        grad[#08]
                    // 09 div   (#01,#08)    grad[#09]
                    // 10 scale (#00,#09)    grad[#10]
                    //
                    // backward pass, given grad[#10]
                    // #10: scale
                    // grad[#00] += scale(grad[#10],#09)
                    // grad[#09] += sum(mul(grad[#10],#00))
                    // #09: div
                    // grad[#08] += neg(mul(grad[#09], div(#09,#08)))
                    // #08: sqrt
                    // grad[#07] += mul(grad[#08], div(0.5, #08))
                    // #07: add
                    // grad[#05] += grad[#07]
                    // #05: scale
                    // grad[#03] += scale(grad[#05],#04)
                    // #03: sum
                    // grad[#02] += repeat(grad[#03], #02)
                    // #02:
                    // grad[#00] += scale(mul(#00, grad[#02]), 2.0)
                    //
                    // substitute and simplify:
                    // grad[#00] = scale(grad(#10), #09) + scale(mul(#00, grad[#02]), 2.0)
                    // grad[#02] = repeat(grad[#03], #02)
                    // grad[#02] = repeat(scale(grad[#05],#04), #02)
                    // grad[#02] = repeat(scale(grad[#07],#04), #02)
                    // grad[#02] = repeat(scale(mul(grad[#08], div(0.5, #08)),#04), #02)
                    // grad[#02] = repeat(scale(mul(neg(mul(grad[#09], div(#09,#08))), div(0.5, #08)),#04), #02)
                    // grad[#02] = repeat(scale(mul(neg(mul(sum(mul(grad[#10],#00)), div(#09,#08))), div(0.5, #08)),#04), #02)
                    // grad[#02] = repeat(-(sum(mul(grad[#10],#00)) * div(#09,#08) * div(0.5, #08) * (1/N)), #02)
                    // grad[#02] = repeat(-(sum(mul(grad[#10],#00)) * div(div(#01,#08),#08) * div(0.5, #08) * (1/N)), #02)
                    // grad[#02] = repeat(-(sum(mul(grad[#10],#00)) * div(1,#08*#08) * div(0.5, #08) * (1/N)), #02)
                    // grad[#02] = repeat(-(sum(mul(grad[#10],#00)) * div(1,#07) * div(0.5, #08) * (1/N)), #02)
                    // grad[#00] = scale(grad(#10), #09) + scale(mul(#00, grad[#02]), 2.0)
                    // grad[#00] = scale(grad(#10), #09) + scale(mul(#00, repeat(-(sum(mul(grad[#10],#00)) * div(1,#07) * div(0.5, #08) * (1/N)), #02)), 2.0)
                    // grad[#00] = scale(grad(#10), #09) + scale(scale(#00, -(sum(mul(grad[#10],#00)) * div(1,#07) * div(0.5, #08) * (1/N))), 2.0)
                    // grad[#00] = scale(grad(#10), #09) + scale(#00, -(sum(mul(grad[#10],#00)) * div(1,#07) * div(1,#08) * (1/N)))
                    // grad[#00] = scale(grad(#10), #09) + scale(#00, sum(mul(grad[#10],#00)) * div(1,#07*#08) * (-1/N))
                    // grad[#00] = scale(grad(#10), #09) + scale(#00, sum(mul(grad[#10],#00)) * div(1,#07*#08) * (-1/N))
                    // grad[#00] = scale(grad(#10), #09) + scale(#00, sum(mul(grad[#10],#00)) * div(1,mean_eps*rms) * (-1/N))
                    // grad[#00] = scale(grad(#10), #09) + scale(#00, sum(mul(grad[#10],#00)) * div(-1,rms*N*mean_eps))
                    // grad[#00] = scale(grad(#10), #09) + scale(#00, sum(mul(grad[#10],#00)) * div(-1,rms*N*(sum_xx/N+eps)))
                    // grad[#00] = scale(grad(#10), #09) + scale(#00, sum(mul(grad[#10],#00)) * div(-1,rms*N*sum_xx+rms*N*eps))
                    // grad[#00] = scale(dz, rrms) + scale(x, sum(mul(dz,x)) * div(-1,rms*N*mean_eps))
                    // grad[#00] = scale(dz, rrms) + scale(x, sum_xdz * div(-1,rms*N*mean_eps))
                    // a = b*c + d*e
                    // a = b*c*f/f + d*e*f/f
                    // a = (b*c*f + d*e*f)*(1/f)
                    // a = (b*c*(1/c) + d*e*(1/c))*(1/(1/c))
                    // a = (b + d*e/c)*c
                    // b = dz, c = rrms, d = x, e = sum_xdz * div(-1,rms*N*mean_eps)
                    // a = (dz + x*sum_xdz * div(-1,rms*N*mean_eps)/rrms)*rrms
                    // a = (dz + x*sum_xdz * div(-1,rms*N*mean_eps)*rms)*rrms
                    // a = (dz + x*sum_xdz * div(-rms,rms*N*mean_eps))*rrms
                    // a = (dz + x*sum_xdz * div(-1,N*mean_eps))*rrms
                    // a = (dz + x*div(-sum_xdz,N*mean_eps))*rrms
                    // a = (dz + x*div(-mean_xdz,mean_eps))*rrms
                    // grad[#00] = scale(dz + scale(x, div(-mean_xdz,mean_eps)),rrms)
                    // grad[#00] = scale(dz + scale(x, -mean_xdz/mean_eps),rrms)
                    // dx = scale(dz + scale(x, -mean_xdz/mean_eps),rrms)
                }
                // dx = scale(dz + scale(x, -mean_xdz/mean_eps),rrms)
                // post-order:
                // dx := x
                // dx := scale(dx,-mean_xdz/mean_eps)
                // dx := add(dx, dz)
                // dx := scale(dx, rrms)
                float * dx = (float *) ((char *) dst->data + i01*nb1 + i02*nb2 + i03*nb3);

                // dx[i00] = (dz + x*(-sum_xdz/sum_eps)) * rrms
                // note: https://github.com/ggml-org/ggml/issues/1491
                const float scale_x = (float) (-sum_xdz) / sum_eps;
                for (int64_t i00 = 0; i00 < ne00; i00++) {
                    dx[i00] = (dz[i00] + x[i00] * scale_x) * rrms;
                }
            }
        }
    }
}

void ggml_compute_forward_rms_norm_back(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_rms_norm_back_f32(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_group_norm

static void ggml_compute_forward_group_norm_f32(
    const ggml_compute_params * params,
    ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    GGML_ASSERT(src0->nb[0] == sizeof(float));

    const int ith = params->ith;
    const int nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    // TODO: optimize

    float eps;
    memcpy(&eps, dst->op_params + 1, sizeof(float));

    int n_channels = src0->ne[2];
    int n_groups = dst->op_params[0];
    int n_channels_per_group = (n_channels + n_groups - 1) / n_groups;
    for (int i = ith; i < n_groups; i += nth) {
        int start = i * n_channels_per_group;
        int end = start + n_channels_per_group;
        if (end > n_channels) {
            end = n_channels;
        }
        int step = end - start;

        for (int64_t i03 = 0; i03 < ne03; i03++) {
            ggml_float sum = 0.0;
            for (int64_t i02 = start; i02 < end; i02++) {
                for (int64_t i01 = 0; i01 < ne01; i01++) {
                    const float * x = (float *)((char *) src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03);

                    ggml_float sumr = 0.0;
                    for (int64_t i00 = 0; i00 < ne00; i00++) {
                        sumr += (ggml_float)x[i00];
                    }
                    sum += sumr;
                }
            }
            const float mean = sum / (ne00 * ne01 * step);

            ggml_float sum2 = 0.0;
            for (int64_t i02 = start; i02 < end; i02++) {
                for (int64_t i01 = 0; i01 < ne01; i01++) {
                    const float * x = (float *)((char *) src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03);

                    float * y = (float *)((char *) dst->data + i01 * nb1 + i02 * nb2 + i03 * nb3);

                    ggml_float sumr = 0.0;
                    for (int64_t i00 = 0; i00 < ne00; i00++) {
                        float v = x[i00] - mean;
                        y[i00] = v;
                        sumr += (ggml_float)(v * v);
                    }
                    sum2 += sumr;
                }
            }
            const float variance = sum2 / (ne00 * ne01 * step);
            const float scale = 1.0f / sqrtf(variance + eps);

            for (int64_t i02 = start; i02 < end; i02++) {
                for (int64_t i01 = 0; i01 < ne01; i01++) {
                    float * y = (float *)((char *) dst->data + i01 * nb1 + i02 * nb2 + i03 * nb3);
                    ggml_vec_scale_f32(ne00, y, scale);
                }
            }
        }
    }
}

void ggml_compute_forward_group_norm(
    const ggml_compute_params * params,
    ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_group_norm_f32(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_l2_norm

static void ggml_compute_forward_l2_norm_f32(
    const ggml_compute_params * params,
    ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    const int ith = params->ith;
    const int nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));

    GGML_ASSERT(eps >= 0.0f);

    // TODO: optimize
    for (int64_t i03 = 0; i03 < ne03; i03++) {
        for (int64_t i02 = 0; i02 < ne02; i02++) {
            for (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                const char * x = (const char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03;

                ggml_float sum = 0.0;
                for (int64_t i00 = 0; i00 < ne00; i00++) {
                    const float xi = *(const float *) (x + i00*nb00);
                    sum += (ggml_float)(xi * xi);
                }

                const float scale = 1.0f/fmaxf(sqrtf(sum), eps);

                char * y = (char *) dst->data + i01*nb1 + i02*nb2 + i03*nb3;

                if (nb00 == sizeof(float) && nb0 == sizeof(float)) {
                    memcpy(y, x, ne00 * sizeof(float));
                    ggml_vec_scale_f32(ne00, (float *) y, scale);
                } else {
                    for (int64_t i00 = 0; i00 < ne00; i00++) {
                        const float xi = *(const float *) (x + i00*nb00);
                        *(float *) (y + i00*nb0) = xi * scale;
                    }
                }
            }
        }
    }
}

void ggml_compute_forward_l2_norm(
    const ggml_compute_params * params,
    ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_l2_norm_f32(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_out_prod

static void ggml_compute_forward_out_prod_f32(
        const ggml_compute_params * params,
              ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    const int ith = params->ith;
    const int nth = params->nth;

    GGML_ASSERT(ne0 == ne00);
    GGML_ASSERT(ne1 == ne10);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    GGML_ASSERT(ne2 % ne02 == 0);
    GGML_ASSERT(ne3 % ne03 == 0);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == sizeof(float));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    // GGML_ASSERT(nb0 <= nb1);
    // GGML_ASSERT(nb1 <= nb2);
    // GGML_ASSERT(nb2 <= nb3);

    // nb01 >= nb00 - src0 is not transposed
    //   compute by src0 rows

    if (ith == 0) {
        ggml_vec_set_f32(ne0*ne1*ne2*ne3, (float *)dst->data, 0);
    }
    ggml_barrier(params->threadpool);

    // dst[:,:,:,:] = 0
    // for i2,i3:
    //   for i1:
    //     for i01:
    //       for i0:
    //         dst[i0,i1,i2,i3] += src0[i0,i01,i2,i3] * src1[i1,i01,i2,i3]

    // parallelize by last three dimensions

    // total rows in dst
    const int64_t nr = ne1*ne2*ne3;

    // rows per thread
    const int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int64_t ir0 = dr*ith;
    const int64_t ir1 = MIN(ir0 + dr, nr);

    // block-tiling attempt
    const int64_t blck_0 = MAX(GGML_VEC_MAD_UNROLL, 32);
    const int64_t blck_1 = 16;

    // dps == dst per src0, used for group query attention
    const int64_t dps2 = ne2 / ne02;
    const int64_t dps3 = ne3 / ne03;

    for (int64_t bir = ir0; bir < ir1; bir += blck_1) {
        const int64_t bir1 = MIN(bir + blck_1, ir1);
        for (int64_t bi01 = 0; bi01 < ne01; bi01 += blck_0) {
            const int64_t bne01 = MIN(bi01 + blck_0, ne01);
            for (int64_t ir = bir; ir < bir1; ++ir) {
                // dst indices
                const int64_t i3 = ir/(ne2*ne1);
                const int64_t i2 = (ir - i3*ne2*ne1)/ne1;
                const int64_t i1 = (ir - i3*ne2*ne1 - i2*ne1);

                const int64_t i02 = i2 / dps2;
                const int64_t i03 = i3 / dps3;

                //const int64_t i10 = i1;
                const int64_t i12 = i2;
                const int64_t i13 = i3;

#if GGML_VEC_MAD_UNROLL > 2
                const int64_t bne01_unroll = bne01 - (bne01 % GGML_VEC_MAD_UNROLL);
                for (int64_t i01 = bi01; i01 < bne01_unroll; i01 += GGML_VEC_MAD_UNROLL) {
                    const int64_t i11 = i01;

                    float * s0 = (float *) ((char *) src0->data + (          i01*nb01 + i02*nb02 + i03*nb03));
                    float * s1 = (float *) ((char *) src1->data + (i1*nb10 + i11*nb11 + i12*nb12 + i13*nb13));
                    float * d  = (float *) ((char *)  dst->data + (          i1*nb1   + i2*nb2   + i3*nb3));

                    ggml_vec_mad_f32_unroll(ne0, nb01, nb11, d, s0, s1);
                }
                for (int64_t i01 = bne01_unroll; i01 < bne01; ++i01) {
                    const int64_t i11 = i01;

                    float * s0 = (float *) ((char *) src0->data + (          i01*nb01 + i02*nb02 + i03*nb03));
                    float * s1 = (float *) ((char *) src1->data + (i1*nb10 + i11*nb11 + i12*nb12 + i13*nb13));
                    float * d  = (float *) ((char *)  dst->data + (          i1*nb1   + i2*nb2   + i3*nb3));

                    ggml_vec_mad_f32(ne0, d, s0, *s1);
                }
#else
                for (int64_t i01 = bi01; i01 < bne01; ++i01) {
                    const int64_t i11 = i01;

                    float * s0 = (float *) ((char *) src0->data + (          i01*nb01 + i02*nb02 + i03*nb03));
                    float * s1 = (float *) ((char *) src1->data + (i1*nb10 + i11*nb11 + i12*nb12 + i13*nb13));
                    float * d  = (float *) ((char *)  dst->data + (          i1*nb1 + i2*nb2 + i3*nb3));

                    ggml_vec_mad_f32(ne0, d, s0, *s1);
                }
#endif
            }
        }
    }
}

static void ggml_compute_forward_out_prod_q_f32(
        const ggml_compute_params * params,
              ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS;

    const int ith = params->ith;
    const int nth = params->nth;

    const ggml_type type = src0->type;
    ggml_to_float_t const dequantize_row_q = ggml_get_type_traits(type)->to_float;

    GGML_ASSERT(ne02 == ne12);
    GGML_ASSERT(ne03 == ne13);
    GGML_ASSERT(ne2  == ne12);
    GGML_ASSERT(ne3  == ne13);

    // we don't support permuted src0 dim0
    GGML_ASSERT(nb00 == ggml_type_size(type));

    // dst dim0 cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    // GGML_ASSERT(nb0 <= nb1);
    // GGML_ASSERT(nb1 <= nb2);
    // GGML_ASSERT(nb2 <= nb3);

    GGML_ASSERT(ne0 == ne00);
    GGML_ASSERT(ne1 == ne10);
    GGML_ASSERT(ne2 == ne02);
    GGML_ASSERT(ne3 == ne03);

    // nb01 >= nb00 - src0 is not transposed
    //   compute by src0 rows

    if (ith == 0) {
        ggml_vec_set_f32(ne0*ne1*ne2*ne3, (float *)dst->data, 0);
    }
    ggml_barrier(params->threadpool);

    // parallelize by last three dimensions

    // total rows in dst
    const int64_t nr = ne1*ne2*ne3;

    // rows per thread
    const int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int64_t ir0 = dr*ith;
    const int64_t ir1 = MIN(ir0 + dr, nr);

    // dst[:,:,:,:] = 0
    // for i2,i3:
    //   for i1:
    //     for i01:
    //       for i0:
    //         dst[i0,i1,i2,i3] += src0[i0,i01,i2,i3] * src1[i1,i01,i2,i3]

    float * wdata = (float *) params->wdata + (ne0 + CACHE_LINE_SIZE_F32) * ith;

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        // dst indices
        const int64_t i3 = ir/(ne2*ne1);
        const int64_t i2 = (ir - i3*ne2*ne1)/ne1;
        const int64_t i1 = (ir - i3*ne2*ne1 - i2*ne1);

        const int64_t i02 = i2;
        const int64_t i03 = i3;

        //const int64_t i10 = i1;
        const int64_t i12 = i2;
        const int64_t i13 = i3;

        for (int64_t i01 = 0; i01 < ne01; ++i01) {
            const int64_t i11 = i01;

            float * s0 = (float *) ((char *) src0->data + (          i01*nb01 + i02*nb02 + i03*nb03));
            float * s1 = (float *) ((char *) src1->data + (i1*nb10 + i11*nb11 + i12*nb12 + i13*nb13));
            float * d  = (float *) ((char *)  dst->data + (          i1*nb1 + i2*nb2 + i3*nb3));

            dequantize_row_q(s0, wdata, ne0);
            ggml_vec_mad_f32(ne0, d, wdata, *s1);
        }
    }
}

static void ggml_compute_forward_out_prod_f16_f32(
        const ggml_compute_params * params,
              ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS;

    const int ith = params->ith;
    const int nth = params->nth;

    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_ASSERT(ne02 == ne12);
    GGML_ASSERT(ne03 == ne13);
    GGML_ASSERT(ne2  == ne12);
    GGML_ASSERT(ne3  == ne13);

    GGML_ASSERT(nb00 == sizeof(ggml_fp16_t));
    GGML_ASSERT(nb0  == sizeof(float));

    GGML_ASSERT(ne0 == ne00);
    GGML_ASSERT(ne1 == ne10);
    GGML_ASSERT(ne2 == ne02);
    GGML_ASSERT(ne3 == ne03);

    if (ith == 0) {
        ggml_vec_set_f32(ne0*ne1*ne2*ne3, (float *)dst->data, 0);
    }
    ggml_barrier(params->threadpool);

    const int64_t nr = ne1*ne2*ne3;
    const int64_t dr = (nr + nth - 1)/nth;
    const int64_t ir0 = dr*ith;
    const int64_t ir1 = MIN(ir0 + dr, nr);

    float * wdata = (float *) params->wdata + (ne0 + CACHE_LINE_SIZE_F32) * ith;

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t i3 = ir/(ne2*ne1);
        const int64_t i2 = (ir - i3*ne2*ne1)/ne1;
        const int64_t i1 = (ir - i3*ne2*ne1 - i2*ne1);

        const int64_t i02 = i2;
        const int64_t i03 = i3;

        const int64_t i12 = i2;
        const int64_t i13 = i3;

        float * d = (float *) ((char *) dst->data + (i1*nb1 + i2*nb2 + i3*nb3));

        for (int64_t i01 = 0; i01 < ne01; ++i01) {
            const int64_t i11 = i01;
            ggml_fp16_t * s0 = (ggml_fp16_t *) ((char *) src0->data + (i01*nb01 + i02*nb02 + i03*nb03));
            float * s1 = (float *) ((char *) src1->data + (i1*nb10 + i11*nb11 + i12*nb12 + i13*nb13));
            ggml_fp16_to_fp32_row(s0, wdata, ne0);
            ggml_vec_mad_f32(ne0, d, wdata, *s1);
        }
    }
}

void ggml_compute_forward_out_prod(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q2_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_NVFP4:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_TQ1_0:
        case GGML_TYPE_TQ2_0:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ2_S:
            {
                ggml_compute_forward_out_prod_q_f32(params, dst);
            } break;
        case GGML_TYPE_F16:
            {
                ggml_compute_forward_out_prod_f16_f32(params, dst);
            } break;
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_out_prod_f32(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_scale

static void ggml_compute_forward_scale_f32(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    float s; // scale factor
    float b; // bias

    memcpy(&s, (float *) dst->op_params + 0, sizeof(float));
    memcpy(&b, (float *) dst->op_params + 1, sizeof(float));

    const int ith = params->ith;
    const int nth = params->nth;

    const int nc = src0->ne[0];
    const int nr = ggml_nrows(src0);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    const size_t nb01 = src0->nb[1];

    const size_t nb1 = dst->nb[1];

    if (b == 0.0f) {
        for (int i1 = ir0; i1 < ir1; i1++) {
            if (dst->data != src0->data) {
                // src0 is same shape as dst => same indices
                // TODO: add x parameter to ggml_vec_scale_f32 and remove this memcpy
                memcpy((char *)dst->data + i1*nb1, (char *)src0->data + i1*nb01, nc * sizeof(float));
            }
            ggml_vec_scale_f32(nc, (float *) ((char *) dst->data + i1*nb1), s);
        }
    } else {
        for (int i1 = ir0; i1 < ir1; i1++) {
            ggml_vec_mad1_f32(nc,
                (float *) ((char *) dst->data  + i1*nb1),
                (float *) ((char *) src0->data + i1*nb1),
                s, b);
        }
    }
}

void ggml_compute_forward_scale(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_scale_f32(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_set

static void ggml_compute_forward_set_f32(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src0));

    // view src0 and dst with these strides and data offset inbytes during set
    // nb0 is implicitly element_size because src0 and dst are contiguous
    size_t nb1     = ((int32_t *) dst->op_params)[0];
    size_t nb2     = ((int32_t *) dst->op_params)[1];
    size_t nb3     = ((int32_t *) dst->op_params)[2];
    size_t offset  = ((int32_t *) dst->op_params)[3];
    bool   inplace = (bool) ((int32_t *) dst->op_params)[4];

    if (!inplace) {
        if (params->ith == 0) {
            // memcpy needs to be synchronized across threads to avoid race conditions.
            // => do it in INIT phase
            memcpy(
                ((char *)  dst->data),
                ((char *) src0->data),
                ggml_nbytes(dst));
        }
        ggml_barrier(params->threadpool);
    }

    const int ith = params->ith;
    const int nth = params->nth;

    const int nr = ggml_nrows(src1);
    const int nc = src1->ne[0];

    GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne)
    GGML_TENSOR_LOCALS(size_t,  nb1, src1, nb)

    // src0 and dst as viewed during set
    const size_t nb0 = ggml_element_size(src0);

    const int im0 = (ne10 == 0 ? 0 : ne10-1);
    const int im1 = (ne11 == 0 ? 0 : ne11-1);
    const int im2 = (ne12 == 0 ? 0 : ne12-1);
    const int im3 = (ne13 == 0 ? 0 : ne13-1);

    GGML_ASSERT(offset + im0*nb0  + im1*nb1  + im2*nb2  + im3*nb3  <= ggml_nbytes(dst));

    GGML_ASSERT(nb10 == sizeof(float));

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are viewed with shape of src1 and offset
        // => same indices
        const int i3 = ir/(ne12*ne11);
        const int i2 = (ir - i3*ne12*ne11)/ne11;
        const int i1 = (ir - i3*ne12*ne11 - i2*ne11);

        ggml_vec_cpy_f32(nc,
                (float *) ((char *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1  + offset),
                (float *) ((char *) src1->data + i3*nb13 + i2*nb12 + i1*nb11));
    }
}

static void ggml_compute_forward_set_i32(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src0));

    // view src0 and dst with these strides and data offset inbytes during set
    // nb0 is implicitly element_size because src0 and dst are contiguous
    size_t nb1     = ((int32_t *) dst->op_params)[0];
    size_t nb2     = ((int32_t *) dst->op_params)[1];
    size_t nb3     = ((int32_t *) dst->op_params)[2];
    size_t offset  = ((int32_t *) dst->op_params)[3];
    bool   inplace = (bool) ((int32_t *) dst->op_params)[4];

    if (!inplace) {
        if (params->ith == 0) {
            // memcpy needs to be synchronized across threads to avoid race conditions.
            // => do it in INIT phase
            memcpy(
                ((char *)  dst->data),
                ((char *) src0->data),
                ggml_nbytes(dst));
        }
        ggml_barrier(params->threadpool);
    }

    const int ith = params->ith;
    const int nth = params->nth;

    const int nr = ggml_nrows(src1);
    const int nc = src1->ne[0];

    GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne)
    GGML_TENSOR_LOCALS(size_t,  nb1, src1, nb)

    // src0 and dst as viewed during set
    const size_t nb0 = ggml_element_size(src0);

    const int im0 = (ne10 == 0 ? 0 : ne10-1);
    const int im1 = (ne11 == 0 ? 0 : ne11-1);
    const int im2 = (ne12 == 0 ? 0 : ne12-1);
    const int im3 = (ne13 == 0 ? 0 : ne13-1);

    GGML_ASSERT(offset + im0*nb0  + im1*nb1  + im2*nb2  + im3*nb3  <= ggml_nbytes(dst));

    GGML_ASSERT(nb10 == sizeof(int32_t));

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are viewed with shape of src1 and offset
        // => same indices
        const int i3 = ir/(ne12*ne11);
        const int i2 = (ir - i3*ne12*ne11)/ne11;
        const int i1 = (ir - i3*ne12*ne11 - i2*ne11);

        ggml_vec_cpy_i32(nc,
                (int32_t *) ((char *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1  + offset),
                (int32_t *) ((char *) src1->data + i3*nb13 + i2*nb12 + i1*nb11));
    }
}

void ggml_compute_forward_set(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_set_f32(params, dst);
            } break;
        case GGML_TYPE_I32:
            {
                ggml_compute_forward_set_i32(params, dst);
            } break;
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q2_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q8_1:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_NVFP4:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_TQ1_0:
        case GGML_TYPE_TQ2_0:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ2_S:
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_cpy

void ggml_compute_forward_cpy(
        const ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_dup(params, dst);
}

// ggml_compute_forward_cont

void ggml_compute_forward_cont(
        const ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_dup(params, dst);
}

// ggml_compute_forward_get_rows

static void ggml_compute_forward_get_rows_q(
        const ggml_compute_params * params,
              ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    const int64_t nc = ne00;
    const int64_t nr = ggml_nelements(src1);

    const ggml_type type = src0->type;
    ggml_to_float_t const dequantize_row_q = ggml_get_type_traits(type)->to_float;

    assert(ne0  == nc);
    assert(ne02 == ne11);
    assert(nb00 == ggml_type_size(type));
    assert(ggml_nrows(dst) == nr);

    const int ith = params->ith;
    const int nth = params->nth;

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int64_t i = ir0; i < ir1; ++i) {
        const int64_t i12 = i/(ne11*ne10);
        const int64_t i11 = (i - i12*ne11*ne10)/ne10;
        const int64_t i10 = (i - i12*ne11*ne10 - i11*ne10);
        const int64_t i01 = *(int32_t *) ((char *) src1->data + i10*nb10 + i11*nb11 + i12*nb12);

        GGML_ASSERT(i01 >= 0 && i01 < ne01);

        dequantize_row_q(
                (const void *) ((char *) src0->data + i01*nb01 + i11*nb02 + i12*nb03),
                     (float *) ((char *)  dst->data + i10*nb1  + i11*nb2  + i12*nb3), nc);
    }
}

static void ggml_compute_forward_get_rows_f16(
        const ggml_compute_params * params,
              ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    const int64_t nc = ne00;
    const int64_t nr = ggml_nelements(src1);

    assert(ne0  == nc);
    assert(ne02 == ne11);
    assert(nb00 == sizeof(ggml_fp16_t));
    assert(ggml_nrows(dst) == nr);

    const int ith = params->ith;
    const int nth = params->nth;

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int64_t i = ir0; i < ir1; ++i) {
        const int64_t i12 = i/(ne11*ne10);
        const int64_t i11 = (i - i12*ne11*ne10)/ne10;
        const int64_t i10 = (i - i12*ne11*ne10 - i11*ne10);
        const int64_t i01 = *(int32_t *) ((char *) src1->data + i10*nb10 + i11*nb11 + i12*nb12);

        GGML_ASSERT(i01 >= 0 && i01 < ne01);

        ggml_cpu_fp16_to_fp32(
            (const ggml_fp16_t*) ((char *) src0->data + i01*nb01 + i11*nb02 + i12*nb03),
                       (float *) ((char *)  dst->data + i10*nb1  + i11*nb2  + i12*nb3), nc);
    }
}

static void ggml_compute_forward_get_rows_bf16(
        const ggml_compute_params * params,
              ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    const int64_t nc = ne00;
    const int64_t nr = ggml_nelements(src1);

    assert(ne0  == nc);
    assert(ne02 == ne11);
    assert(nb00 == sizeof(ggml_bf16_t));
    assert(ggml_nrows(dst) == nr);

    const int ith = params->ith;
    const int nth = params->nth;

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int64_t i = ir0; i < ir1; ++i) {
        const int64_t i12 = i/(ne11*ne10);
        const int64_t i11 = (i - i12*ne11*ne10)/ne10;
        const int64_t i10 = (i - i12*ne11*ne10 - i11*ne10);
        const int64_t i01 = *(int32_t *) ((char *) src1->data + i10*nb10 + i11*nb11 + i12*nb12);

        GGML_ASSERT(i01 >= 0 && i01 < ne01);

        ggml_cpu_bf16_to_fp32(
            (const ggml_bf16_t *) ((char *) src0->data + i01*nb01 + i11*nb02 + i12*nb03),
                        (float *) ((char *)  dst->data + i10*nb1  + i11*nb2  + i12*nb3), nc);
    }
}

static void ggml_compute_forward_get_rows_f32(
        const ggml_compute_params * params,
              ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    const int64_t nc = ne00;
    const int64_t nr = ggml_nelements(src1);

    assert(ne0  == nc);
    assert(ne02 == ne11);
    assert(nb00 == sizeof(float));
    assert(ggml_nrows(dst) == nr);

    const int ith = params->ith;
    const int nth = params->nth;

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int64_t i = ir0; i < ir1; ++i) {
        const int64_t i12 = i/(ne11*ne10);
        const int64_t i11 = (i - i12*ne11*ne10)/ne10;
        const int64_t i10 = (i - i12*ne11*ne10 - i11*ne10);
        const int64_t i01 = *(int32_t *) ((char *) src1->data + i10*nb10 + i11*nb11 + i12*nb12);

        GGML_ASSERT(i01 >= 0 && i01 < ne01);

        ggml_vec_cpy_f32(nc,
                (float *) ((char *)  dst->data + i10*nb1  + i11*nb2  + i12*nb3),
                (float *) ((char *) src0->data + i01*nb01 + i11*nb02 + i12*nb03));
    }
}

void ggml_compute_forward_get_rows(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q2_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q8_1:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_NVFP4:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_TQ1_0:
        case GGML_TYPE_TQ2_0:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ2_S:
            {
                ggml_compute_forward_get_rows_q(params, dst);
            } break;
        case GGML_TYPE_F16:
            {
                ggml_compute_forward_get_rows_f16(params, dst);
            } break;
        case GGML_TYPE_BF16:
            {
                ggml_compute_forward_get_rows_bf16(params, dst);
            } break;
        case GGML_TYPE_F32:
        case GGML_TYPE_I32:
            {
                ggml_compute_forward_get_rows_f32(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }

    //static bool first = true;
    //printf("ne0 = %d, ne1 = %d, ne2 = %d\n", dst->ne[0], dst->ne[1], dst->ne[2]);
    //if (first) {
    //    first = false;
    //} else {
    //    for (int k = 0; k < dst->ne[1]; ++k) {
    //        for (int j = 0; j < dst->ne[0]/16; ++j) {
    //            for (int i = 0; i < 16; ++i) {
    //                printf("%8.4f ", ((float *) dst->data)[k*dst->ne[0] + j*16 + i]);
    //            }
    //            printf("\n");
    //        }
    //        printf("\n");
    //    }
    //    printf("\n");
    //    exit(0);
    //}
}

template<typename src_t, typename idx_t>
static void ggml_compute_forward_set_rows_impl(
        const ggml_compute_params * params,
              ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    const int64_t nc = ne00;
    const int64_t nr = ne01;

    assert(ne0  == nc);
    assert(ne2  == ne02);
    assert(ne3  == ne03);
    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    assert(ne02 % ne11 == 0);
    assert(ne03 % ne12 == 0);

    const int ith = params->ith;
    const int nth = params->nth;

    // rows per thread
    const int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int64_t ir0 = dr*ith;
    const int64_t ir1 = std::min(ir0 + dr, nr);

    const size_t rs = ggml_row_size(src0->type, nc);

    ggml_from_float_t const from_float = ggml_get_type_traits_cpu(dst->type)->from_float;

    for (int64_t i03 = 0; i03 < ne03; ++i03) {
        for (int64_t i02 = 0; i02 < ne02; ++i02) {
            for (int64_t i = ir0; i < ir1; ++i) {
                const int64_t i12 = i03%ne12;
                const int64_t i11 = i02%ne11;
                const int64_t i10 = i;

                const int64_t i1 = *(idx_t *) ((char *) src1->data + i10*nb10 + i11*nb11 + i12*nb12);

                GGML_ASSERT(i1 >= 0 && i1 < ne1);

                if constexpr (std::is_same_v<src_t, float>) {
                    from_float(
                            (const float *) ((char *) src0->data +  i*nb01 + i02*nb02 + i03*nb03),
                                            ((char *)  dst->data + i1*nb1  + i02*nb2  + i03*nb3), nc);
                } else if constexpr (std::is_same_v<src_t, ggml_fp16_t>) {
                    if (dst->type == GGML_TYPE_F16) {
                        memcpy(
                                            ((char *)  dst->data + i1*nb1  + i02*nb2  + i03*nb3),
                                            ((char *) src0->data +  i*nb01 + i02*nb02 + i03*nb03),
                                            rs);
                    } else {
                        float * wdata = (float *) params->wdata + (nc + CACHE_LINE_SIZE_F32) * ith;
                        ggml_fp16_to_fp32_row(
                                (const ggml_fp16_t *) ((char *) src0->data + i*nb01 + i02*nb02 + i03*nb03),
                                wdata, nc);
                        from_float(wdata,
                                ((char *) dst->data + i1*nb1 + i02*nb2 + i03*nb3), nc);
                    }
                } else {
                    GGML_ABORT("src0->type = %d (%s) not supported", src0->type, ggml_type_name(src0->type));
                }
            }
        }
    }
}

void ggml_compute_forward_set_rows(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                if (src1->type == GGML_TYPE_I64) {
                    ggml_compute_forward_set_rows_impl<float, int64_t>(params, dst);
                } else if (src1->type == GGML_TYPE_I32) {
                    ggml_compute_forward_set_rows_impl<float, int32_t>(params, dst);
                } else {
                    GGML_ABORT("src1->type = %d (%s) not supported", src1->type, ggml_type_name(src1->type));
                }
            } break;
        case GGML_TYPE_F16:
            {
                if (src1->type == GGML_TYPE_I64) {
                    ggml_compute_forward_set_rows_impl<ggml_fp16_t, int64_t>(params, dst);
                } else if (src1->type == GGML_TYPE_I32) {
                    ggml_compute_forward_set_rows_impl<ggml_fp16_t, int32_t>(params, dst);
                } else {
                    GGML_ABORT("src1->type = %d (%s) not supported", src1->type, ggml_type_name(src1->type));
                }
            } break;
        default:
            {
                GGML_ABORT("src0->type = %d (%s) not supported", src0->type, ggml_type_name(src0->type));
            }
    }
}

// ggml_compute_forward_get_rows_back

static void ggml_compute_forward_get_rows_back_f32_f16(
        const ggml_compute_params * params,
              ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    if (params->ith != 0) {
        return;
    }

    GGML_ASSERT(ggml_is_contiguous(dst));

    // ggml_compute_forward_dup_same_cont(params, opt0, dst);

    memset(dst->data, 0, ggml_nbytes(dst));

    const int nc = src0->ne[0];
    const int nr = ggml_nelements(src1);

    GGML_ASSERT( dst->ne[0] == nc);
    GGML_ASSERT(src0->nb[0] == sizeof(ggml_fp16_t));

    for (int i = 0; i < nr; ++i) {
        const int r = ((int32_t *) src1->data)[i];

        for (int j = 0; j < nc; ++j) {
            ggml_fp16_t v = ((ggml_fp16_t *) ((char *) src0->data + i*src0->nb[1]))[j];
            ((float *) ((char *) dst->data + r*dst->nb[1]))[j] += GGML_CPU_FP16_TO_FP32(v);
        }
    }
}

static void ggml_compute_forward_get_rows_back_f32(
        const ggml_compute_params * params,
              ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    if (params->ith != 0) {
        return;
    }

    GGML_ASSERT(ggml_is_contiguous(dst));

    // ggml_compute_forward_dup_same_cont(params, opt0, dst);

    memset(dst->data, 0, ggml_nbytes(dst));

    const int nc = src0->ne[0];
    const int nr = ggml_nelements(src1);

    GGML_ASSERT( dst->ne[0] == nc);
    GGML_ASSERT(src0->nb[0] == sizeof(float));

    for (int i = 0; i < nr; ++i) {
        const int r = ((int32_t *) src1->data)[i];

        ggml_vec_add_f32(nc,
                (float *) ((char *)  dst->data + r*dst->nb[1]),
                (float *) ((char *)  dst->data + r*dst->nb[1]),
                (float *) ((char *) src0->data + i*src0->nb[1]));
    }
}

void ggml_compute_forward_get_rows_back(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F16:
            {
                ggml_compute_forward_get_rows_back_f32_f16(params, dst);
            } break;
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_get_rows_back_f32(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }

    //static bool first = true;
    //printf("ne0 = %d, ne1 = %d, ne2 = %d\n", dst->ne[0], dst->ne[1], dst->ne[2]);
    //if (first) {
    //    first = false;
    //} else {
    //    for (int k = 0; k < dst->ne[1]; ++k) {
    //        for (int j = 0; j < dst->ne[0]/16; ++j) {
    //            for (int i = 0; i < 16; ++i) {
    //                printf("%8.4f ", ((float *) dst->data)[k*dst->ne[0] + j*16 + i]);
    //            }
    //            printf("\n");
    //        }
    //        printf("\n");
    //    }
    //    printf("\n");
    //    exit(0);
    //}
}

// ggml_compute_forward_diag

static void ggml_compute_forward_diag_f32(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    if (params->ith != 0) {
        return;
    }

    // TODO: handle transposed/permuted matrices

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(ne00 == ne0);
    GGML_ASSERT(ne00 == ne1);
    GGML_ASSERT(ne01 == 1);
    GGML_ASSERT(ne02 == ne2);
    GGML_ASSERT(ne03 == ne3);

    GGML_ASSERT(nb00 == sizeof(float));
    GGML_ASSERT(nb0  == sizeof(float));

    for (int i3 = 0; i3 < ne3; i3++) {
        for (int i2 = 0; i2 < ne2; i2++) {
            for (int i1 = 0; i1 < ne1; i1++) {
                float * d = (float *)((char *)  dst->data + i3*nb3  + i2*nb2 + i1*nb1);
                float * s = (float *)((char *) src0->data + i3*nb03 + i2*nb02);
                for (int i0 = 0; i0 < i1; i0++) {
                    d[i0] = 0;
                }
                d[i1] = s[i1];
                for (int i0 = i1+1; i0 < ne0; i0++) {
                    d[i0] = 0;
                }
            }
        }
    }
}

void ggml_compute_forward_diag(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_diag_f32(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_diag_mask_inf

static void ggml_compute_forward_diag_mask_f32(
        const ggml_compute_params * params,
        ggml_tensor * dst,
        const float value) {

    const ggml_tensor * src0 = dst->src[0];

    const int ith = params->ith;
    const int nth = params->nth;

    const int  n_past  = ((int32_t *) dst->op_params)[0];
    const bool inplace = src0->data == dst->data;

    GGML_ASSERT(n_past >= 0);

    if (!inplace) {
        if (ith == 0) {
            // memcpy needs to be synchronized across threads to avoid race conditions.
            // => do it in INIT phase
            GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(src0));
            GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src0));
            memcpy(
                ((char *)  dst->data),
                ((char *) src0->data),
                ggml_nbytes(dst));
        }
        ggml_barrier(params->threadpool);
    }

    // TODO: handle transposed/permuted matrices

    const int n  = ggml_nrows(src0);
    const int nc = src0->ne[0];
    const int nr = src0->ne[1];
    const int nz = n/nr;

    GGML_ASSERT( dst->nb[0] == sizeof(float));
    GGML_ASSERT(src0->nb[0] == sizeof(float));

    for (int k = 0; k < nz; k++) {
        for (int j = ith; j < nr; j += nth) {
            for (int i = n_past; i < nc; i++) {
                if (i > n_past + j) {
                    *(float *)((char *) dst->data + k*dst->nb[2] + j*dst->nb[1] + i*dst->nb[0]) = value;
                }
            }
        }
    }
}

void ggml_compute_forward_diag_mask_inf(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_diag_mask_f32(params, dst, -INFINITY);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

void ggml_compute_forward_diag_mask_zero(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_diag_mask_f32(params, dst, 0);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_soft_max

static void ggml_compute_forward_soft_max_f32(
        const ggml_compute_params * params,
              ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * src2 = dst->src[2];

    assert(ggml_is_contiguous(dst));
    assert(ggml_are_same_shape(src0, dst));

    float scale    = 1.0f;
    float max_bias = 0.0f;

    memcpy(&scale,    (float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias, (float *) dst->op_params + 1, sizeof(float));

    const int ith = params->ith;
    const int nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    const int64_t nb11 = src1 ? src1->nb[1] : 1;
    const int64_t nb12 = src1 ? src1->nb[2] : 1;
    const int64_t nb13 = src1 ? src1->nb[3] : 1;

    const int64_t ne12 = src1 ? src1->ne[2] : 1;
    const int64_t ne13 = src1 ? src1->ne[3] : 1;

    // TODO: is this supposed to be ceil instead of floor?
    //       https://huggingface.co/mosaicml/mpt-7b/blob/main/attention.py#L370
    const uint32_t n_head      = ne02;
    const uint32_t n_head_log2 = 1u << (uint32_t) floor(log2(n_head));

    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    float * wp = (float *) params->wdata + (ne00 + CACHE_LINE_SIZE_F32) * ith;

    const bool use_f16 = (src1 && src1->type == GGML_TYPE_F16);

    // sinks
    const float * sk = src2 ? (float *)((char *) src2->data) : nullptr;

    for (int64_t i03 = 0; i03 < ne03; i03++) {
        for (int64_t i02 = 0; i02 < ne02; i02++) {
            for (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                const int64_t i11 = i01;
                const int64_t i12 = i02%ne12;
                const int64_t i13 = i03%ne13;

                // ALiBi
                const uint32_t h = i02; // head
                const float slope = (max_bias > 0.0f) ? h < n_head_log2 ? powf(m0, h + 1) : powf(m1, 2*(h - n_head_log2) + 1) : 1.0f;

                float * sp = (float *)((char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03);
                float * dp = (float *)((char *)  dst->data + i01*nb1  + i02*nb2  + i03*nb3);

                // broadcast the mask across rows
                ggml_fp16_t * mp_f16 = src1 ? (ggml_fp16_t *)((char *) src1->data + i11*nb11 + i12*nb12 + i13*nb13) : NULL;
                float       * mp_f32 = src1 ? (float       *)((char *) src1->data + i11*nb11 + i12*nb12 + i13*nb13) : NULL;

                ggml_vec_cpy_f32  (ne00, wp, sp);
                ggml_vec_scale_f32(ne00, wp, scale);
                if (mp_f32) {
                    if (use_f16) {
                        for (int i = 0; i < ne00; ++i) {
                            wp[i] += slope*GGML_CPU_FP16_TO_FP32(mp_f16[i]);
                        }
                    } else {
                        for (int i = 0; i < ne00; ++i) {
                            wp[i] += slope*mp_f32[i];
                        }
                    }
                }

#ifndef NDEBUG
                for (int i = 0; i < ne00; ++i) {
                    //printf("p[%d] = %f\n", i, p[i]);
                    assert(!isnan(wp[i]));
                }
#endif // NDEBUG

                float max = -INFINITY;
                ggml_vec_max_f32(ne00, &max, wp);

                // if we have sinks, make a correction as if they were included in the softmax
                if (sk) {
                    max = MAX(max, sk[i02]);
                }

                ggml_float sum = ggml_vec_soft_max_f32(ne00, dp, wp, max);
                assert(sum > 0.0);

                if (sk) {
                    sum += (ggml_float) expf(sk[i02] - max);
                }

                sum = 1.0/sum;
                ggml_vec_scale_f32(ne00, dp, sum);

#ifndef NDEBUG
                for (int i = 0; i < ne00; ++i) {
                    assert(!isnan(dp[i]));
                    assert(!isinf(dp[i]));
                }
#endif // NDEBUG
            }
        }
    }
}

void ggml_compute_forward_soft_max(
        const ggml_compute_params * params,
              ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_soft_max_f32(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}


// ggml_compute_forward_soft_max_ext_back

static void ggml_compute_forward_soft_max_ext_back_f32(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_are_same_shape(src1, dst));

    float scale    = 1.0f;
    float max_bias = 0.0f;

    memcpy(&scale,    (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));

    GGML_ASSERT(max_bias == 0.0f);

    // TODO: handle transposed/permuted matrices

    const int ith = params->ith;
    const int nth = params->nth;

    const int nc = src0->ne[0];
    const int nr = ggml_nrows(src0);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int i1 = ir0; i1 < ir1; i1++) {
        float *dy = (float *)((char *) src0->data + i1*src0->nb[1]);
        float *y  = (float *)((char *) src1->data + i1*src1->nb[1]);
        float *dx = (float *)((char *) dst->data  + i1*dst->nb[1]);

#ifndef NDEBUG
        for (int i = 0; i < nc; ++i) {
            //printf("p[%d] = %f\n", i, p[i]);
            assert(!isnan(dy[i]));
            assert(!isnan(y[i]));
        }
#endif // NDEBUG
        // Jii = yi - yi*yi
        // Jij = -yi*yj
        // J = diag(y)-y.T*y
        // dx = J * dy
        // dxk = sum_i(Jki * dyi)
        // dxk = sum_i(-yk*yi * dyi) - (-yk*yk)*dyk + (yk - yk*yk)*dyk
        // dxk = sum_i(-yk*yi * dyi) + yk*yk*dyk + yk*dyk - yk*yk*dyk
        // dxk = sum_i(-yk*yi * dyi) + yk*dyk
        // dxk = -yk * sum_i(yi * dyi) + yk*dyk
        // dxk = -yk * dot(y, dy) + yk*dyk
        // dxk = yk * (- dot(y, dy) + dyk)
        // dxk = yk * (dyk - dot(y, dy))
        //
        // post-order:
        // dot_y_dy := dot(y, dy)
        // dx := dy
        // dx := dx - dot_y_dy
        // dx := dx * y

        // linear runtime, no additional memory
        float dot_y_dy = 0;
        ggml_vec_dot_f32  (nc, &dot_y_dy, 0, y, 0, dy, 0, 1);
        ggml_vec_cpy_f32  (nc, dx, dy);
        ggml_vec_acc1_f32 (nc, dx, -dot_y_dy);
        ggml_vec_mul_f32  (nc, dx, dx, y);
        ggml_vec_scale_f32(nc, dx, scale);

#ifndef NDEBUG
        for (int i = 0; i < nc; ++i) {
            assert(!isnan(dx[i]));
            assert(!isinf(dx[i]));
        }
#endif // NDEBUG
    }
}

void ggml_compute_forward_soft_max_ext_back(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_soft_max_ext_back_f32(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_clamp

static void ggml_compute_forward_clamp_f32(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    float min;
    float max;
    memcpy(&min, (float *) dst->op_params + 0, sizeof(float));
    memcpy(&max, (float *) dst->op_params + 1, sizeof(float));

    const int ith = params->ith;
    const int nth = params->nth;

    const int n  = ggml_nrows(src0);
    const int nc = src0->ne[0];

    const size_t nb00 = src0->nb[0];
    const size_t nb01 = src0->nb[1];

    const size_t nb0 = dst->nb[0];
    const size_t nb1 = dst->nb[1];

    GGML_ASSERT( nb0 == sizeof(float));
    GGML_ASSERT(nb00 == sizeof(float));

    for (int j = ith; j < n; j += nth) {
        float * dst_ptr  = (float *) ((char *)  dst->data + j*nb1);
        float * src0_ptr = (float *) ((char *) src0->data + j*nb01);

        for (int i = 0; i < nc; i++) {
            dst_ptr[i] = MAX(MIN(src0_ptr[i], max), min);
        }
    }
}

static void ggml_compute_forward_clamp_f16(
    const ggml_compute_params * params,
    ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    float min;
    float max;
    memcpy(&min, (float *) dst->op_params + 0, sizeof(float));
    memcpy(&max, (float *) dst->op_params + 1, sizeof(float));

    const int ith = params->ith;
    const int nth = params->nth;

    const int n  = ggml_nrows(src0);
    const int nc = src0->ne[0];

    const size_t nb00 = src0->nb[0];
    const size_t nb01 = src0->nb[1];

    const size_t nb0 = dst->nb[0];
    const size_t nb1 = dst->nb[1];

    GGML_ASSERT( nb0 == sizeof(ggml_fp16_t));
    GGML_ASSERT(nb00 == sizeof(ggml_fp16_t));

    for (int j = ith; j < n; j += nth) {
        ggml_fp16_t * dst_ptr  = (ggml_fp16_t *) ((char *)  dst->data + j*nb1);
        ggml_fp16_t * src0_ptr = (ggml_fp16_t *) ((char *) src0->data + j*nb01);

        for (int i = 0; i < nc; i++) {
            float v = GGML_CPU_FP16_TO_FP32(src0_ptr[i]);
            dst_ptr[i] = GGML_CPU_FP32_TO_FP16(MAX(MIN(v, max), min));
        }
    }
}

void ggml_compute_forward_clamp(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_clamp_f32(params, dst);
            } break;
        case GGML_TYPE_F16:
            {
                ggml_compute_forward_clamp_f16(params, dst);
            } break;
        case GGML_TYPE_BF16:
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q2_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q8_1:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_NVFP4:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_TQ1_0:
        case GGML_TYPE_TQ2_0:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_Q8_K:
        case GGML_TYPE_I8:
        case GGML_TYPE_I16:
        case GGML_TYPE_I32:
        case GGML_TYPE_I64:
        case GGML_TYPE_F64:
        case GGML_TYPE_COUNT:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_rope

static float rope_yarn_ramp(const float low, const float high, const int i0) {
    const float y = (i0 / 2 - low) / MAX(0.001f, high - low);
    return 1 - MIN(1, MAX(0, y));
}

// YaRN algorithm based on LlamaYaRNScaledRotaryEmbedding.py from https://github.com/jquesnelle/yarn
// MIT licensed. Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng.
static void rope_yarn(
    float theta_extrap, float freq_scale, float corr_dims[2], int64_t i0, float ext_factor, float mscale,
    float * cos_theta, float * sin_theta) {
    // Get n-d rotational scaling corrected for extrapolation
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], i0) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        // Get n-d magnitude scaling corrected for interpolation
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    *cos_theta = cosf(theta) * mscale;
    *sin_theta = sinf(theta) * mscale;
}

static void ggml_rope_cache_init(
     float theta_base, float freq_scale, const float * freq_factors, float corr_dims[2], int64_t ne0, float ext_factor, float mscale,
     float * cache, float sin_sign, float theta_scale) {
    // ref: https://github.com/jquesnelle/yarn/blob/master/scaled_rope/LlamaYaRNScaledRotaryEmbedding.py
    float theta = theta_base;
    for (int64_t i0 = 0; i0 < ne0; i0 += 2) {
        const float ff = freq_factors ? freq_factors[i0/2] : 1.0f;
        rope_yarn(
            theta/ff, freq_scale, corr_dims, i0, ext_factor, mscale, &cache[i0 + 0], &cache[i0 + 1]
        );
        cache[i0 + 1] *= sin_sign;

        theta *= theta_scale;
    }
}

static void ggml_mrope_cache_init(
     float theta_base_t, float theta_base_h, float theta_base_w, float theta_base_e, int sections[4], bool is_imrope, bool indep_sects,
     float freq_scale, const float * freq_factors, float corr_dims[2], int64_t ne0, float ext_factor, float mscale,
     float * cache, float sin_sign, float theta_scale) {
    // ref: https://github.com/jquesnelle/yarn/blob/master/scaled_rope/LlamaYaRNScaledRotaryEmbedding.py
    float theta_t = theta_base_t;
    float theta_h = theta_base_h;
    float theta_w = theta_base_w;
    float theta_e = theta_base_e;  // extra position id for vision encoder
    int sect_dims = sections[0] + sections[1] + sections[2] + sections[3];
    int sec_w = sections[1] + sections[0];
    int sec_e = sections[2] + sec_w;
    GGML_ASSERT(sect_dims <= ne0);

    for (int64_t i0 = 0; i0 < ne0; i0 += 2) {
        const float ff = freq_factors ? freq_factors[i0/2] : 1.0f;

        int sector = (i0 / 2) % sect_dims;
        if (indep_sects) {
            // compute theta independently for each dim sections
            // (i.e. reset corresponding theta when `i0` go from one section to another)
            if (sector == 0) {
                theta_t = theta_base_t;
            }
            else if (sector == sections[0]) {
                theta_h = theta_base_h;;
            }
            else if (sector == sec_w) {
                theta_w = theta_base_w;
            }
            else if (sector == sec_e) {
                theta_e = theta_base_e;
            }
        }

        float theta = theta_t;
        if (is_imrope) { // qwen3vl apply interleaved mrope
            if (sector % 3 == 1 && sector < 3 * sections[1]) {
                theta = theta_h;
            } else if (sector % 3 == 2 && sector < 3 * sections[2]) {
                theta = theta_w;
            } else if (sector % 3 == 0 && sector < 3 * sections[0]) {
                theta = theta_t;
            } else {
                theta = theta_e;
            }
        } else {
            if (sector >= sections[0] && sector < sec_w) {
                theta = theta_h;
            }
            else if (sector >= sec_w && sector < sec_w + sections[2]) {
                theta = theta_w;
            }
            else if (sector >= sec_w + sections[2]) {
                theta = theta_e;
            }
        }

        rope_yarn(
            theta/ff, freq_scale, corr_dims, i0, ext_factor, mscale, &cache[i0 + 0], &cache[i0 + 1]
        );
        cache[i0 + 1] *= sin_sign;

        theta_t *= theta_scale;
        theta_w *= theta_scale;
        theta_h *= theta_scale;
        theta_e *= theta_scale;
    }
}


template<typename T>
static void rotate_pairs(const int64_t n, const int64_t n_offset, const float * cache, const T * src_data, T * dst_data, const int scale = 2) {
  for (int64_t i0 = 0; i0 < n; i0 += 2) {
    const int64_t ic = i0/scale; // hack for GGML_ROPE_TYPE_NORMAL, where we need ic = i0; for all other cases, ic = i0/2

    const float cos_theta = cache[i0 + 0];
    const float sin_theta = cache[i0 + 1];

    const T * const src = src_data + ic;
    T * dst             = dst_data + ic;

    const float x0 = type_conversion_table<T>::to_f32(src[0]);
    const float x1 = type_conversion_table<T>::to_f32(src[n_offset]);

    dst[0]        = type_conversion_table<T>::from_f32(x0*cos_theta - x1*sin_theta);
    dst[n_offset] = type_conversion_table<T>::from_f32(x0*sin_theta + x1*cos_theta);
  }
}

template<typename T> //float or ggml_fp16_t
static void ggml_compute_forward_rope_flt(
        const ggml_compute_params * params,
        ggml_tensor * dst,
        const bool forward) {

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * src2 = dst->src[2];

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_I32);

    float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
    int sections[4];

    //const int n_past     = ((int32_t *) dst->op_params)[0];
    const int n_dims     = ((int32_t *) dst->op_params)[1];
    const int mode       = ((int32_t *) dst->op_params)[2];
    //const int n_ctx      = ((int32_t *) dst->op_params)[3];
    const int n_ctx_orig = ((int32_t *) dst->op_params)[4];

    memcpy(&freq_base,   (int32_t *) dst->op_params +  5, sizeof(float));
    memcpy(&freq_scale,  (int32_t *) dst->op_params +  6, sizeof(float));
    memcpy(&ext_factor,  (int32_t *) dst->op_params +  7, sizeof(float));
    memcpy(&attn_factor, (int32_t *) dst->op_params +  8, sizeof(float));
    memcpy(&beta_fast,   (int32_t *) dst->op_params +  9, sizeof(float));
    memcpy(&beta_slow,   (int32_t *) dst->op_params + 10, sizeof(float));
    memcpy(&sections,    (int32_t *) dst->op_params + 11, sizeof(int)*4);

    GGML_TENSOR_UNARY_OP_LOCALS

    //printf("ne0: %d, ne1: %d, ne2: %d, ne3: %d\n", ne0, ne1, ne2, ne3);
    //printf("n_past = %d, ne2 = %d\n", n_past, ne2);

    GGML_ASSERT(nb0 == nb00);
    GGML_ASSERT(nb0 == sizeof(T));

    const int ith = params->ith;
    const int nth = params->nth;

    const int nr = ggml_nrows(dst);

    GGML_ASSERT(n_dims <= ne0);
    GGML_ASSERT(n_dims % 2 == 0);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    // row index used to determine which thread to use
    int ir = 0;

    const float theta_scale = powf(freq_base, -2.0f/n_dims);

    float corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);

    const bool is_imrope = mode == GGML_ROPE_TYPE_IMROPE; // qwen3vl apply interleaved mrope
    const bool mrope_used = mode & GGML_ROPE_TYPE_MROPE;  // ggml_rope_multi, note: also true for vision (24 & 8 == true) and for imrope
    const bool is_vision = mode == GGML_ROPE_TYPE_VISION;

    if (mrope_used) {
        GGML_ASSERT(sections[0] > 0 || sections[1] > 0 || sections[2] > 0);
    }

    if (is_vision) {
        GGML_ASSERT(n_dims == ne0/2);
    }

    const float * freq_factors = NULL;
    if (src2 != NULL) {
        GGML_ASSERT(src2->type == GGML_TYPE_F32);
        GGML_ASSERT(src2->ne[0] >= n_dims / 2);
        freq_factors = (const float *) src2->data;
    }

    // backward process uses inverse rotation by cos and sin.
    // cos and sin build a rotation matrix, where the inverse is the transpose.
    // this essentially just switches the sign of sin.
    const float sin_sign = forward ? 1.0f : -1.0f;

    const int32_t * pos = (const int32_t *) src1->data;

    int64_t last_i2 = -1;

    for (int64_t i3 = 0; i3 < ne3; i3++) { // batch
        for (int64_t i2 = 0; i2 < ne2; i2++) { // seq-len
            for (int64_t i1 = 0; i1 < ne1; i1++) { // attn-heads
                if (ir++ < ir0) continue; // skip rows mapped to other threads
                if (ir   > ir1) break;

                float * cache = (float *) params->wdata + (ne0 + CACHE_LINE_SIZE_F32)*ith;
                if (last_i2 != i2) {
                    if (!mrope_used) {
                        const int64_t p = pos[i2];
                        ggml_rope_cache_init(p, freq_scale, freq_factors, corr_dims, ne0, ext_factor, attn_factor, cache, sin_sign, theta_scale);
                    }
                    else {
                        const int64_t p_t = pos[i2];
                        const int64_t p_h = pos[i2 + ne2];
                        const int64_t p_w = pos[i2 + ne2 * 2];
                        const int64_t p_e = pos[i2 + ne2 * 3];
                        ggml_mrope_cache_init(
                            p_t, p_h, p_w, p_e, sections, is_imrope, is_vision,
                            freq_scale, freq_factors, corr_dims, ne0, ext_factor, attn_factor, cache, sin_sign, theta_scale);
                    }

                    last_i2 = i2;
                }

                T * src = (T *)((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01);
                T * dst_data  = (T *)((char *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1);

                switch (mode) {
                    case GGML_ROPE_TYPE_NORMAL:
                        rotate_pairs<T>(n_dims, 1, cache, src, dst_data, 1);
                        break;
                    case GGML_ROPE_TYPE_NEOX:
                    case GGML_ROPE_TYPE_MROPE:
                    case GGML_ROPE_TYPE_IMROPE:
                        rotate_pairs<T>(n_dims, n_dims/2, cache, src, dst_data);
                        break;
                    case GGML_ROPE_TYPE_VISION:
                        rotate_pairs<T>(ne0, n_dims, cache, src, dst_data);
                        break;
                    default:
                        GGML_ABORT("rope type not supported");
                }

                if (!is_vision) {
                    // fill the remain channels with data from src tensor
                    for (int64_t i0 = n_dims; i0 < ne0; i0 += 2) {
                        const T * const src = (T *)((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
                        T * dst_data  = (T *)((char *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

                        dst_data[0] = src[0];
                        dst_data[1] = src[1];
                    }
                }
            } //attn-heads
        }
    }
}

void ggml_compute_forward_rope(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F16:
            {
                ggml_compute_forward_rope_flt<ggml_fp16_t>(params, dst, true);
            } break;
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_rope_flt<float>(params, dst, true);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_rope_back

void ggml_compute_forward_rope_back(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F16:
            {
                ggml_compute_forward_rope_flt<ggml_fp16_t>(params, dst, false);
            } break;
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_rope_flt<float>(params, dst, false);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_conv_transpose_1d

static void ggml_compute_forward_conv_transpose_1d_f16_f32(
        const ggml_compute_params * params,
              ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_BINARY_OP_LOCALS

    const int ith = params->ith;
    const int nth = params->nth;

    const int nk = ne00*ne01*ne02;

    GGML_ASSERT(nb00 == sizeof(ggml_fp16_t));
    GGML_ASSERT(nb10 == sizeof(float));

    if (ith == 0) {
        memset(params->wdata, 0, params->wsize);

        // permute kernel data (src0) from (K x Cout x Cin) to (Cin x K x Cout)
        {
            ggml_fp16_t * const wdata = (ggml_fp16_t *) params->wdata + 0;

            for (int64_t i02 = 0; i02 < ne02; i02++) {
                for (int64_t i01 = 0; i01 < ne01; i01++) {
                    const ggml_fp16_t * const src = (ggml_fp16_t *)((char *) src0->data + i02*nb02 + i01*nb01);
                    ggml_fp16_t * dst_data = wdata + i01*ne00*ne02;
                    for (int64_t i00 = 0; i00 < ne00; i00++) {
                        dst_data[i00*ne02 + i02] = src[i00];
                    }
                }
            }
        }

        // permute source data (src1) from (L x Cin) to (Cin x L)
        {
            ggml_fp16_t * const wdata = (ggml_fp16_t *) params->wdata + nk;
            ggml_fp16_t * dst_data = wdata;

            for (int64_t i11 = 0; i11 < ne11; i11++) {
                const float * const src = (float *)((char *) src1->data + i11*nb11);
                for (int64_t i10 = 0; i10 < ne10; i10++) {
                    dst_data[i10*ne11 + i11] = GGML_CPU_FP32_TO_FP16(src[i10]);
                }
            }
        }

        // need to zero dst since we are accumulating into it
        memset(dst->data, 0, ggml_nbytes(dst));
    }
    ggml_barrier(params->threadpool);

    const int32_t s0 = ((const int32_t*)(dst->op_params))[0];

    // total rows in dst
    const int nr = ne1;

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    ggml_fp16_t * const wdata     = (ggml_fp16_t *) params->wdata + 0;
    ggml_fp16_t * const wdata_src = wdata + nk;

    for (int i1 = ir0; i1 < ir1; i1++) {
        float * dst_data = (float *)((char *) dst->data + i1*nb1);
        ggml_fp16_t * wdata_kernel = wdata + i1*ne02*ne00;
        for (int i10 = 0; i10 < ne10; i10++) {
            const int i1n = i10*ne11;
            for (int i00 = 0; i00 < ne00; i00++) {
                float v = 0;
                ggml_vec_dot_f16(ne02, &v, 0,
                        (ggml_fp16_t *)    wdata_src + i1n, 0,
                        (ggml_fp16_t *) wdata_kernel + i00*ne02, 0, 1);
                dst_data[i10*s0 + i00] += v;
            }
        }
    }
}

static void ggml_compute_forward_conv_transpose_1d_f32(
        const ggml_compute_params * params,
              ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_BINARY_OP_LOCALS

    const int ith = params->ith;
    const int nth = params->nth;

    const int nk = ne00*ne01*ne02;

    GGML_ASSERT(nb00 == sizeof(float));
    GGML_ASSERT(nb10 == sizeof(float));

    if (ith == 0) {
        memset(params->wdata, 0, params->wsize);

        // prepare kernel data (src0) from (K x Cout x Cin) to (Cin x K x Cout)
        {
            float * const wdata = (float *) params->wdata + 0;

            for (int64_t i02 = 0; i02 < ne02; i02++) {
                for (int64_t i01 = 0; i01 < ne01; i01++) {
                    const float * const src = (float *)((char *) src0->data + i02*nb02 + i01*nb01);
                    float * dst_data = wdata + i01*ne00*ne02;
                    for (int64_t i00 = 0; i00 < ne00; i00++) {
                        dst_data[i00*ne02 + i02] = src[i00];
                    }
                }
            }
        }

        // prepare source data (src1)
        {
            float * const wdata = (float *) params->wdata + nk;
            float * dst_data = wdata;

            for (int64_t i11 = 0; i11 < ne11; i11++) {
                const float * const src = (float *)((char *) src1->data + i11*nb11);
                for (int64_t i10 = 0; i10 < ne10; i10++) {
                    dst_data[i10*ne11 + i11] = src[i10];
                }
            }
        }

        // need to zero dst since we are accumulating into it
        memset(dst->data, 0, ggml_nbytes(dst));
    }
    ggml_barrier(params->threadpool);

    const int32_t s0 = ((const int32_t*)(dst->op_params))[0];

    // total rows in dst
    const int nr = ne1;

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    float * const wdata     = (float *) params->wdata + 0;
    float * const wdata_src = wdata + nk;

    for (int i1 = ir0; i1 < ir1; i1++) {
        float * dst_data = (float *)((char *) dst->data + i1*nb1);
        float * wdata_kernel = wdata + i1*ne02*ne00;
        for (int i10 = 0; i10 < ne10; i10++) {
            const int i1n = i10*ne11;
            for (int i00 = 0; i00 < ne00; i00++) {
                float v = 0;
                ggml_vec_dot_f32(ne02, &v, 0,
                        wdata_src + i1n, 0,
                        wdata_kernel + i00*ne02, 0, 1);
                dst_data[i10*s0 + i00] += v;
            }
        }
    }
}

void ggml_compute_forward_conv_transpose_1d(
        const ggml_compute_params * params,
              ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F16:
            {
                ggml_compute_forward_conv_transpose_1d_f16_f32(params, dst);
            } break;
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_conv_transpose_1d_f32(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_im2col_f32
// src0: kernel [OC, IC, KH, KW]
// src1: image [N, IC, IH, IW]
// dst:  result [N, OH, OW, IC*KH*KW]
static void ggml_compute_forward_im2col_f32(
        const ggml_compute_params * params,
              ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_BINARY_OP_LOCALS;

    const int32_t s0 = ((const int32_t *)(dst->op_params))[0];
    const int32_t s1 = ((const int32_t *)(dst->op_params))[1];
    const int32_t p0 = ((const int32_t *)(dst->op_params))[2];
    const int32_t p1 = ((const int32_t *)(dst->op_params))[3];
    const int32_t d0 = ((const int32_t *)(dst->op_params))[4];
    const int32_t d1 = ((const int32_t *)(dst->op_params))[5];
    const bool is_2D = ((const int32_t *)(dst->op_params))[6] == 1;

    const int ith = params->ith;
    const int nth = params->nth;

    const int64_t N  = is_2D ? ne13 : ne12;
    const int64_t IC = is_2D ? ne12 : ne11;
    const int64_t IH = is_2D ? ne11 : 1;
    const int64_t IW = ne10;

    const int64_t KH = is_2D ? ne01 : 1;
    const int64_t KW = ne00;

    const int64_t OH = is_2D ? ne2 : 1;
    const int64_t OW = ne1;

    int ofs0 = is_2D ? nb13 : nb12;
    int ofs1 = is_2D ? nb12 : nb11;

    GGML_ASSERT(nb10 == sizeof(float));

    // im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
    {
        float * const wdata = (float *) dst->data;

        for (int64_t in = 0; in < N; in++) {
            for (int64_t ioh = 0; ioh < OH; ioh++) { // 1
                for (int64_t iow = 0; iow < OW; iow++) {
                    for (int64_t iic = ith; iic < IC; iic += nth) {

                        // micro kernel
                        float * dst_data = wdata + (in*OH*OW + ioh*OW + iow)*(IC*KH*KW); // [IC, KH, KW]
                        const float * const src_data = (float *)((char *) src1->data + in*ofs0 + iic*ofs1); // [IH, IW]

                        for (int64_t ikh = 0; ikh < KH; ikh++) {  // 1
                            for (int64_t ikw = 0; ikw < KW; ikw++) {
                                const int64_t iiw = iow*s0 + ikw*d0 - p0;
                                const int64_t iih = ioh*s1 + ikh*d1 - p1;

                                if (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
                                    dst_data[iic*(KH*KW) + ikh*KW + ikw] = 0;
                                } else {
                                    dst_data[iic*(KH*KW) + ikh*KW + ikw] = (src_data[iih*IW + iiw]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}


// ggml_compute_forward_im2col_f16
// src0: kernel [OC, IC, KH, KW]
// src1: image [N, IC, IH, IW]
// dst:  result [N, OH, OW, IC*KH*KW]
static void ggml_compute_forward_im2col_f16(
        const ggml_compute_params * params,
              ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F16);

    GGML_TENSOR_BINARY_OP_LOCALS;

    const int32_t s0 = ((const int32_t *)(dst->op_params))[0];
    const int32_t s1 = ((const int32_t *)(dst->op_params))[1];
    const int32_t p0 = ((const int32_t *)(dst->op_params))[2];
    const int32_t p1 = ((const int32_t *)(dst->op_params))[3];
    const int32_t d0 = ((const int32_t *)(dst->op_params))[4];
    const int32_t d1 = ((const int32_t *)(dst->op_params))[5];
    const bool is_2D = ((const int32_t *)(dst->op_params))[6] == 1;

    const int ith = params->ith;
    const int nth = params->nth;

    const int64_t N  = is_2D ? ne13 : ne12;
    const int64_t IC = is_2D ? ne12 : ne11;
    const int64_t IH = is_2D ? ne11 : 1;
    const int64_t IW = ne10;

    const int64_t KH = is_2D ? ne01 : 1;
    const int64_t KW = ne00;

    const int64_t OH = is_2D ? ne2 : 1;
    const int64_t OW = ne1;

    int ofs0 = is_2D ? nb13 : nb12;
    int ofs1 = is_2D ? nb12 : nb11;

    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
    {
        ggml_fp16_t * const wdata = (ggml_fp16_t *) dst->data;

        for (int64_t in = 0; in < N; in++) {
            for (int64_t ioh = 0; ioh < OH; ioh++) { // 1
                for (int64_t iow = 0; iow < OW; iow++) {
                    for (int64_t iic = ith; iic < IC; iic += nth) {

                        // micro kernel
                        ggml_fp16_t * dst_data = wdata + (in*OH*OW + ioh*OW + iow)*(IC*KH*KW); // [IC, KH, KW]
                        const float * const src_data_f32 = src1->type == GGML_TYPE_F32
                            ? (const float *)((const char *) src1->data + in*ofs0 + iic*ofs1)
                            : nullptr; // [IH, IW]
                        const ggml_fp16_t * const src_data_f16 = src1->type == GGML_TYPE_F16
                            ? (const ggml_fp16_t *)((const char *) src1->data + in*ofs0 + iic*ofs1)
                            : nullptr; // [IH, IW]

                        for (int64_t ikh = 0; ikh < KH; ikh++) {  // 1
                            for (int64_t ikw = 0; ikw < KW; ikw++) {
                                const int64_t iiw = iow*s0 + ikw*d0 - p0;
                                const int64_t iih = ioh*s1 + ikh*d1 - p1;

                                if (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
                                    dst_data[iic*(KH*KW) + ikh*KW + ikw] = 0;
                                } else {
                                    if (src_data_f32 != nullptr) {
                                        dst_data[iic*(KH*KW) + ikh*KW + ikw] = GGML_CPU_FP32_TO_FP16(src_data_f32[iih*IW + iiw]);
                                    } else {
                                        dst_data[iic*(KH*KW) + ikh*KW + ikw] = src_data_f16[iih*IW + iiw];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void ggml_compute_forward_im2col(
        const ggml_compute_params * params,
              ggml_tensor * dst) {
    switch (dst->type) {
        case GGML_TYPE_F16:
            {
                ggml_compute_forward_im2col_f16(params, dst);
            } break;
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_im2col_f32(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_im2col_back_f32

void ggml_compute_forward_im2col_back_f32(
        const ggml_compute_params * params,
              ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0]; // gradients of forward pass output
    const ggml_tensor * src1 = dst->src[1]; // convolution kernel

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_BINARY_OP_LOCALS;

    const int32_t s0 = ((const int32_t *)(dst->op_params))[0];
    const int32_t s1 = ((const int32_t *)(dst->op_params))[1];
    const int32_t p0 = ((const int32_t *)(dst->op_params))[2];
    const int32_t p1 = ((const int32_t *)(dst->op_params))[3];
    const int32_t d0 = ((const int32_t *)(dst->op_params))[4];
    const int32_t d1 = ((const int32_t *)(dst->op_params))[5];
    const bool is_2D = ((const int32_t *)(dst->op_params))[6] == 1;

    const int ith = params->ith;
    const int nth = params->nth;

    const int64_t N  = is_2D ? ne3 : ne2;
    const int64_t IC = is_2D ? ne2 : ne1;
    const int64_t IH = is_2D ? ne1 : 1;
    const int64_t IW = ne0;

    const int64_t KH = is_2D ? ne11 : 1;
    const int64_t KW = ne10;

    const int64_t OH = is_2D ? ne02 : 1;
    const int64_t OW = ne01;

    int ofs0 = is_2D ? nb3 : nb2;
    int ofs1 = is_2D ? nb2 : nb1;

    GGML_ASSERT(nb0  == sizeof(float));

    // im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
    {
        float * const wdata = (float *) dst->data;

        for (int64_t in = 0; in < N; in++) {
            for (int64_t iic = ith; iic < IC; iic += nth) {
                for (int64_t iih = 0; iih < IH; iih++) {
                    for (int64_t iiw = 0; iiw < IW; iiw++) {

                        // micro kernel
                        float grad = 0.0f;
                        for (int64_t ikh = 0; ikh < KH; ikh++) {
                            for (int64_t ikw = 0; ikw < KW; ikw++) {
                                // For s0 > 1 some values were skipped over in the forward pass.
                                // These values have tmpw % s0 != 0 and need to be skipped in the backwards pass as well.
                                const int64_t tmpw = (iiw + p0 - ikw*d0);
                                if (tmpw % s0 != 0) {
                                    continue;
                                }
                                const int64_t iow = tmpw / s0;

                                // Equivalent logic as above except for s1.
                                int64_t ioh;
                                if (is_2D) {
                                    const int64_t tmph = iih + p1 - ikh*d1;

                                    if (tmph % s1 != 0) {
                                        continue;
                                    }

                                    ioh = tmph / s1;
                                } else {
                                    ioh = 0;
                                }

                                if (iow < 0 || iow >= OW || ioh < 0 || ioh >= OH) {
                                    continue;
                                }

                                const float * const grad_in = (const float *) src0->data
                                    + (in*OH*OW + ioh*OW + iow)*(IC*KH*KW); // [IC, KH, KW]
                                grad += grad_in[iic*(KH*KW) + ikh*KW + ikw];
                            }
                        }
                        float * dst_data = (float *)((char *) wdata + (in*ofs0 + iic*ofs1)); // [IH, IW]
                        dst_data[iih*IW + iiw] = grad;
                    }
                }
            }
        }
    }
}


// ggml_compute_forward_im2col_3d_f16
// src0: kernel [OC*IC, KD, KH, KW]
// src1: image [N*IC, ID, IH, IW]
// dst:  result [N*OD, OH, OW, IC * KD * KH * KW]
static void ggml_compute_forward_im2col_3d_f16(
        const ggml_compute_params * params,
              ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F16);

    GGML_TENSOR_BINARY_OP_LOCALS;

    const int32_t s0 = ((const int32_t *)(dst->op_params))[0];
    const int32_t s1 = ((const int32_t *)(dst->op_params))[1];
    const int32_t s2 = ((const int32_t *)(dst->op_params))[2];
    const int32_t p0 = ((const int32_t *)(dst->op_params))[3];
    const int32_t p1 = ((const int32_t *)(dst->op_params))[4];
    const int32_t p2 = ((const int32_t *)(dst->op_params))[5];
    const int32_t d0 = ((const int32_t *)(dst->op_params))[6];
    const int32_t d1 = ((const int32_t *)(dst->op_params))[7];
    const int32_t d2 = ((const int32_t *)(dst->op_params))[8];
    const int32_t IC = ((const int32_t *)(dst->op_params))[9];


    const int ith = params->ith;
    const int nth = params->nth;

    const int64_t N  = ne13 / IC;
    const int64_t ID = ne12;
    const int64_t IH = ne11;
    const int64_t IW = ne10;

    const int64_t OC = ne03 / IC;
    GGML_UNUSED(OC);
    const int64_t KD = ne02;
    const int64_t KH = ne01;
    const int64_t KW = ne00;

    const int64_t OD = ne3 / N;
    const int64_t OH = ne2;
    const int64_t OW = ne1;
    const int64_t OH_OW = OH*OW;
    const int64_t KD_KH_KW = KD*KH*KW;
    const int64_t KH_KW = KH*KW;
    const int64_t IC_KD_KH_KW = IC*KD*KH*KW;

    GGML_ASSERT(nb10 == sizeof(float));

    // im2col: [N*IC, ID, IH, IW] => [N*OD, OH, OW, IC * KD * KH * KW]
    {
        ggml_fp16_t * const wdata = (ggml_fp16_t *) dst->data;

        for (int64_t in = 0; in < N; in++) {
            for (int64_t iod = 0; iod < OD; iod++) {
                for (int64_t ioh = 0; ioh < OH; ioh++) {
                    for (int64_t iow = 0; iow < OW; iow++) {
                        for (int64_t iic = ith; iic < IC; iic += nth) {

                            // micro kernel
                            ggml_fp16_t * dst_data = wdata + (in*OD*OH_OW + iod*OH_OW + ioh*OW + iow)*IC_KD_KH_KW; // [IC, KD, KH, KW]
                            const float * const src_data = (const float *) ((const char *)src1->data + (in*IC + iic)*nb13); // [ID, IH, IW]

                            for (int64_t ikd = 0; ikd < KD; ikd++) {
                                for (int64_t ikh = 0; ikh < KH; ikh++) {
                                    for (int64_t ikw = 0; ikw < KW; ikw++) {
                                        const int64_t iiw = iow*s0 + ikw*d0 - p0;
                                        const int64_t iih = ioh*s1 + ikh*d1 - p1;
                                        const int64_t iid = iod*s2 + ikd*d2 - p2;

                                        if (iid < 0 || iid >= ID || iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
                                            dst_data[iic*KD_KH_KW + ikd * KH_KW + ikh*KW + ikw] = 0;
                                        } else {
                                            const float * const s = (const float *) ((const char *)src_data + iid*nb12 + iih*nb11 + iiw*nb10); // [ID, IH, IW]
                                            dst_data[iic*KD_KH_KW + ikd * KH_KW + ikh*KW + ikw] = GGML_CPU_FP32_TO_FP16(*s);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// ggml_compute_forward_im2col_3d_f32
// src0: kernel [OC*IC, KD, KH, KW]
// src1: image [N*IC, ID, IH, IW]
// dst:  result [N*OD, OH, OW, IC * KD * KH * KW]
static void ggml_compute_forward_im2col_3d_f32(
        const ggml_compute_params * params,
              ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_BINARY_OP_LOCALS;

    const int32_t s0 = ((const int32_t *)(dst->op_params))[0];
    const int32_t s1 = ((const int32_t *)(dst->op_params))[1];
    const int32_t s2 = ((const int32_t *)(dst->op_params))[2];
    const int32_t p0 = ((const int32_t *)(dst->op_params))[3];
    const int32_t p1 = ((const int32_t *)(dst->op_params))[4];
    const int32_t p2 = ((const int32_t *)(dst->op_params))[5];
    const int32_t d0 = ((const int32_t *)(dst->op_params))[6];
    const int32_t d1 = ((const int32_t *)(dst->op_params))[7];
    const int32_t d2 = ((const int32_t *)(dst->op_params))[8];
    const int32_t IC = ((const int32_t *)(dst->op_params))[9];


    const int ith = params->ith;
    const int nth = params->nth;

    const int64_t N  = ne13 / IC;
    const int64_t ID = ne12;
    const int64_t IH = ne11;
    const int64_t IW = ne10;

    const int64_t OC = ne03 / IC;
    GGML_UNUSED(OC);
    const int64_t KD = ne02;
    const int64_t KH = ne01;
    const int64_t KW = ne00;

    const int64_t OD = ne3 / N;
    const int64_t OH = ne2;
    const int64_t OW = ne1;

    const int64_t OH_OW = OH*OW;
    const int64_t KD_KH_KW = KD*KH*KW;
    const int64_t KH_KW = KH*KW;
    const int64_t IC_KD_KH_KW = IC*KD*KH*KW;

    GGML_ASSERT(nb10 == sizeof(float));

    // im2col: [N*IC, ID, IH, IW] => [N*OD, OH, OW, IC * KD * KH * KW]
    {
        float * const wdata = (float *) dst->data;

        for (int64_t in = 0; in < N; in++) {
            for (int64_t iod = 0; iod < OD; iod++) {
                for (int64_t ioh = 0; ioh < OH; ioh++) {
                    for (int64_t iow = 0; iow < OW; iow++) {
                        for (int64_t iic = ith; iic < IC; iic += nth) {

                            // micro kernel
                            float * dst_data = wdata + (in*OD*OH_OW + iod*OH_OW + ioh*OW + iow)*IC_KD_KH_KW; // [IC, KD, KH, KW]
                            const float * const src_data = (const float *) ((const char *)src1->data + (in*IC + iic)*nb13); // [ID, IH, IW]

                            for (int64_t ikd = 0; ikd < KD; ikd++) {
                                for (int64_t ikh = 0; ikh < KH; ikh++) {
                                    for (int64_t ikw = 0; ikw < KW; ikw++) {
                                        const int64_t iiw = iow*s0 + ikw*d0 - p0;
                                        const int64_t iih = ioh*s1 + ikh*d1 - p1;
                                        const int64_t iid = iod*s2 + ikd*d2 - p2;

                                        if (iid < 0 || iid >= ID || iih < 0 || iih >= IH || iiw < 0 || iiw >= IW || iid < 0 || iid >= ID) {
                                            dst_data[iic*KD_KH_KW + ikd * KH_KW + ikh*KW + ikw] = 0;
                                        } else {
                                            const float * const s = (const float *) ((const char *)src_data + iid*nb12 + iih*nb11 + iiw*nb10); // [ID, IH, IW]
                                            dst_data[iic*KD_KH_KW + ikd * KH_KW + ikh*KW + ikw] = *s;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}


void ggml_compute_forward_im2col_3d(
        const ggml_compute_params * params,
              ggml_tensor * dst) {
    switch (dst->type) {
        case GGML_TYPE_F16:
            {
                ggml_compute_forward_im2col_3d_f16(params, dst);
            } break;
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_im2col_3d_f32(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

static void ggml_call_mul_mat(ggml_type type, const ggml_compute_params * params, int64_t m, int64_t n, int64_t k,
                              void * a, void * b, float * c) {
    const ggml_type_traits * traits = ggml_get_type_traits(type);
    struct ggml_tensor src1 = {};
    src1.type  = type;
    src1.ne[0] = k;
    src1.ne[1] = m;
    src1.ne[2] = 1;
    src1.ne[3] = 1;
    src1.nb[0] = traits->type_size;
    src1.nb[1] = k * traits->type_size;
    src1.nb[2] = src1.nb[1];
    src1.nb[3] = src1.nb[2];
    src1.data  = a;

    struct ggml_tensor src0 = {};
    src0.type  = type;
    src0.ne[0] = k;
    src0.ne[1] = n;
    src0.ne[2] = 1;
    src0.ne[3] = 1;
    src0.nb[0] = traits->type_size;
    src0.nb[1] = k * traits->type_size;
    src0.nb[2] = src0.nb[1];
    src0.nb[3] = src0.nb[2];
    src0.data  = b;

    struct ggml_tensor dst = {};
    dst.ne[0] = n;
    dst.ne[1] = m;
    dst.ne[2] = 1;
    dst.ne[3] = 1;
    dst.nb[0] = sizeof(float);
    dst.nb[1] = n * sizeof(float);
    dst.nb[2] = dst.nb[1];
    dst.nb[3] = dst.nb[2];
    dst.data  = c;
    dst.src[0] = &src0;
    dst.src[1] = &src1;

    ggml_compute_forward_mul_mat(params, &dst);
}

static inline int64_t ggml_wrap_around(int64_t coord, int64_t size) {
    return (coord  + size) % size; // adding size avoids negative number weirdness
}

// ggml_compute_forward_col2im_1d
//
// Scatter-add columns [K*OC, T_in] -> signal [T_out, OC]
// where T_out = (T_in - 1)*s + K - 2*p.  Gather approach: each output reads ceil(K/s) inputs.
// Parallelized over the time axis so the split stays balanced whatever OC is.
// Supports F32, F16, BF16 input/output (same type), F32 accumulator.

template <typename elem_t>
static void ggml_compute_forward_col2im_1d_impl(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src = dst->src[0];  // [K*OC, T_in]

    GGML_ASSERT(ggml_is_contiguous(src));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int32_t s0 = ((const int32_t *)(dst->op_params))[0];
    const int32_t OC = ((const int32_t *)(dst->op_params))[1];
    const int32_t p0 = ((const int32_t *)(dst->op_params))[2];

    const int64_t K_OC = src->ne[0];
    const int64_t T_in = src->ne[1];
    const int64_t K    = K_OC / OC;
    const int64_t T_out = dst->ne[0];

    const elem_t * col_data = (const elem_t *) src->data;
    elem_t       * dst_data = (elem_t *) dst->data;

    const int ith = params->ith;
    const int nth = params->nth;

    // Parallelize over the time axis: the split stays balanced whatever OC is,
    // down to OC = 1 for mono audio, and threads read disjoint column bands
    const int64_t dr = (T_out + nth - 1) / nth;
    const int64_t it0 = dr * ith;
    const int64_t it1 = it0 + dr < T_out ? it0 + dr : T_out;

    for (int64_t oc = 0; oc < OC; oc++) {
        for (int64_t t_out = it0; t_out < it1; t_out++) {
            const int64_t t_abs = t_out + p0;  // absolute position in uncropped signal
            // Gather: find all (t_in, k) where t_in * s + k == t_abs, 0 <= k < K
            int64_t t_in_min = (t_abs - K + 1 + s0 - 1) / s0;  // ceil((t_abs-K+1)/s)
            if (t_in_min < 0) t_in_min = 0;
            int64_t t_in_max = t_abs / s0;
            if (t_in_max >= T_in) t_in_max = T_in - 1;

            float sum = 0.0f;
            for (int64_t t_in = t_in_min; t_in <= t_in_max; t_in++) {
                int64_t k = t_abs - t_in * s0;
                if (k >= 0 && k < K) {
                    // col layout: [K*OC, T_in], element (oc*K+k, t_in)
                    sum += type_conversion_table<elem_t>::to_f32(col_data[(oc * K + k) + t_in * K_OC]);
                }
            }
            // dst layout: [T_out, OC], element (t_out, oc)
            dst_data[t_out + oc * T_out] = type_conversion_table<elem_t>::from_f32(sum);
        }
    }
}

void ggml_compute_forward_col2im_1d(
        const ggml_compute_params * params,
        ggml_tensor * dst) {
    switch (dst->src[0]->type) {
        case GGML_TYPE_F32:  ggml_compute_forward_col2im_1d_impl<float>      (params, dst); break;
        case GGML_TYPE_F16:  ggml_compute_forward_col2im_1d_impl<ggml_fp16_t>(params, dst); break;
        case GGML_TYPE_BF16: ggml_compute_forward_col2im_1d_impl<ggml_bf16_t>(params, dst); break;
        default: GGML_ABORT("col2im_1d: unsupported type %d", dst->src[0]->type);
    }
}

// ggml_compute_forward_conv_2d


static void ggml_compute_forward_conv_2d_impl(const ggml_compute_params * params,
                                              const ggml_tensor *         kernel,  // [KW, KH, IC, OC]
                                              const ggml_tensor *         src,     // [W, H, C, N]
                                              ggml_tensor *               dst,     // [OW, OH, OC, N]
                                              ggml_type                   kernel_type) {

    GGML_ASSERT(ggml_is_contiguous(kernel));
    GGML_ASSERT(kernel_type == GGML_TYPE_F16 || kernel_type == GGML_TYPE_F32);
    GGML_ASSERT(kernel->type == kernel_type);

    const ggml_type_traits * traits = ggml_get_type_traits(kernel_type);

    const int32_t stride_x   = dst->op_params[0];
    const int32_t stride_y   = dst->op_params[1];
    const int32_t pad_x      = dst->op_params[2];
    const int32_t pad_y      = dst->op_params[3];
    const int32_t dilation_x = dst->op_params[4];
    const int32_t dilation_y = dst->op_params[5];

    const int64_t c_in  = src->ne[2];
    const int64_t c_out = kernel->ne[3];
    GGML_ASSERT(c_in == kernel->ne[2]);

    const int64_t src_w = src->ne[0];
    const int64_t src_h = src->ne[1];
    const int64_t knl_w = kernel->ne[0];
    const int64_t knl_h = kernel->ne[1];
    const int64_t dst_w = dst->ne[0];
    const int64_t dst_h = dst->ne[1];

    const float * src_data = (float *) src->data;
    void  * knl_data       = kernel->data;
    float * dst_data       = (float *) dst->data;

    const int64_t knl_n           = knl_w * knl_h * c_in;
    const int64_t patch_total     = dst->ne[3] * dst_w * dst_h;

    const int64_t space_per_patch   = knl_n * traits->type_size + c_out * sizeof(float);
    const int64_t batch_size        = params->wsize / space_per_patch;
    const int64_t patches_per_batch = batch_size > 8 ? (batch_size / 8) * 8 : batch_size;
    const int64_t batch_n           = (patch_total + patches_per_batch - 1) / patches_per_batch;

    GGML_ASSERT(patches_per_batch > 0 && batch_size >= 1);

    void * tmp = params->wdata;

    for (int64_t batch_i = 0; batch_i < batch_n; ++batch_i) {

        const int64_t patch_start_batch = batch_i * patches_per_batch;
        const int64_t patch_end_batch   = std::min(patch_start_batch + patches_per_batch,
                                              patch_total);
        const int64_t patch_n           = patch_end_batch - patch_start_batch;

        const int64_t patch_per_thread  = (patch_n + params->nth - 1) / params->nth;
        const int64_t patch_start       = patch_start_batch + params->ith * patch_per_thread;
        const int64_t patch_end         = std::min(patch_start + patch_per_thread, patch_end_batch);

        //im2col for a patch
        for (int64_t p = patch_start; p < patch_end; ++p) {
            const int64_t  batch_n     =  p / (dst_w * dst_h);
            const int64_t  src_x       = (p / dst_w) % dst_h;
            const int64_t  src_y       =  p % dst_w;

            const float * src_base = (const float *)((const char *)src_data + batch_n * src->nb[3]);
            char *        dst_row  = (char *) tmp + (p % patches_per_batch) * knl_n * traits->type_size;

            for (int64_t ic = 0; ic < c_in; ++ic) {
                for (int64_t ky = 0; ky < knl_h; ++ky) {
                    for (int64_t kx = 0; kx < knl_w; ++kx) {
                        const int64_t sy = src_x * stride_y + ky * dilation_y - pad_y;
                        const int64_t sx = src_y * stride_x + kx * dilation_x - pad_x;

                        int64_t dst_idx = ic * (knl_h * knl_w) + ky * knl_w + kx;

                        float src_val;
                        if (sy < 0 || sy >= src_h || sx < 0 || sx >= src_w) {
                            src_val = 0.0f;
                        } else {
                            const float * src_ptr = (const float *)((const char *)src_base + sx * src->nb[0] + sy * src->nb[1] + ic * src->nb[2]);
                            src_val               = *src_ptr;
                        }

                        char * element_ptr = dst_row + dst_idx * traits->type_size;
                        if (kernel_type == GGML_TYPE_F32) {
                            *(float *) element_ptr = src_val;
                        } else if (kernel_type == GGML_TYPE_F16) {
                            *(ggml_fp16_t *) element_ptr = GGML_CPU_FP32_TO_FP16(src_val);
                        }
                    }
                }
            }
        }   // patches handled by this thread

        ggml_barrier(params->threadpool);

        float * gemm_output = (float *) ((char *) tmp + patches_per_batch * knl_n * traits->type_size);

        GGML_ASSERT(gemm_output + patch_n * c_out <= (float*)tmp + params->wsize);

        // GEMM: patches[patch_n, knl_n] × kernel[knl_n, c_out] = output[patch_n, c_out]
        ggml_call_mul_mat(kernel_type, params, patch_n, c_out, knl_n, tmp, knl_data, gemm_output);

        ggml_barrier(params->threadpool);


        //permute back [OC, N, OH, OW] to [N, OC, OH, OW]
        const int64_t permute_per_thread = (patch_n + params->nth - 1) / params->nth;
        const int64_t permute_start = params->ith * permute_per_thread;
        const int64_t permute_end = std::min(permute_start + permute_per_thread, patch_n);

        for (int64_t i = permute_start; i < permute_end; ++i) {
            const int64_t p       = patch_start_batch + i;
            const int64_t batch_n = p / (dst_w * dst_h);
            const int64_t dst_y   = (p / dst_w) % dst_h;
            const int64_t dst_x   = p % dst_w;

            for (int64_t oc = 0; oc < c_out; ++oc) {
                const float value = gemm_output[i * c_out + oc];
                float * dst_ptr = (float *)((char *)dst_data + dst_x * dst->nb[0] + dst_y * dst->nb[1] + oc * dst->nb[2] + batch_n * dst->nb[3]);
                *dst_ptr = value;
            }
        }
    }
}

void ggml_compute_forward_conv_2d(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    ggml_compute_forward_conv_2d_impl(params, src0, src1, dst, src0->type);
}

// ggml_compute_forward_conv_3d

static void ggml_compute_forward_conv_3d_impl(const ggml_compute_params * params,
                                              const ggml_tensor *         kernel,
                                              const ggml_tensor *         src,
                                              ggml_tensor *               dst,
                                              ggml_type                   kernel_type) {

    GGML_ASSERT(ggml_is_contiguous(kernel));
    GGML_ASSERT(kernel_type == GGML_TYPE_F16 || kernel_type == GGML_TYPE_F32);
    GGML_ASSERT(kernel->type == kernel_type);

    const ggml_type_traits * traits = ggml_get_type_traits(kernel_type);

    const int32_t s0 = dst->op_params[0];
    const int32_t s1 = dst->op_params[1];
    const int32_t s2 = dst->op_params[2];
    const int32_t p0 = dst->op_params[3];
    const int32_t p1 = dst->op_params[4];
    const int32_t p2 = dst->op_params[5];
    const int32_t d0 = dst->op_params[6];
    const int32_t d1 = dst->op_params[7];
    const int32_t d2 = dst->op_params[8];
    const int32_t c  = dst->op_params[9];
    const int32_t n  = dst->op_params[10];
    const int32_t oc = dst->op_params[11];

    const int64_t src_w = src->ne[0];
    const int64_t src_h = src->ne[1];
    const int64_t src_d = src->ne[2];
    const int64_t knl_w = kernel->ne[0];
    const int64_t knl_h = kernel->ne[1];
    const int64_t knl_d = kernel->ne[2];
    const int64_t dst_w = dst->ne[0];
    const int64_t dst_h = dst->ne[1];
    const int64_t dst_d = dst->ne[2];

    const float * src_data = (float *) src->data;
    void  * knl_data       = kernel->data;
    float * dst_data       = (float *) dst->data;

    const int64_t knl_n_per_channel = knl_w * knl_h * knl_d;
    const int64_t knl_n_total       = knl_n_per_channel * c;
    const int64_t patch_total       = n * dst_w * dst_h * dst_d;

    const int64_t space_per_patch   = knl_n_total * traits->type_size + oc * sizeof(float);
    const int64_t batch_size        = params->wsize / space_per_patch;
    const int64_t patches_per_batch = batch_size > 8 ? (batch_size / 8) * 8 : batch_size;
    const int64_t batch_n           = (patch_total + patches_per_batch - 1) / patches_per_batch;

    GGML_ASSERT(patches_per_batch > 0 && batch_size >= 1);

    void * tmp = params->wdata;

    for (int64_t batch_i = 0; batch_i < batch_n; ++batch_i) {
        const int64_t patch_start_batch = batch_i * patches_per_batch;
        const int64_t patch_end_batch   = std::min(patch_start_batch + patches_per_batch, patch_total);
        const int64_t patch_n_in_batch  = patch_end_batch - patch_start_batch;

        const int64_t patch_per_thread  = (patch_n_in_batch + params->nth - 1) / params->nth;
        const int64_t patch_start       = patch_start_batch + params->ith * patch_per_thread;
        const int64_t patch_end         = std::min(patch_start + patch_per_thread, patch_end_batch);

        for (int64_t p = patch_start; p < patch_end; ++p) {
            const int64_t p_in_batch = p % (dst_w * dst_h * dst_d);
            const int64_t p_in_depth = p_in_batch % (dst_w * dst_h);
            const int64_t batch_idx  = p / (dst_w * dst_h * dst_d);
            const int64_t dst_z      = p_in_batch / (dst_w * dst_h);
            const int64_t dst_y      = p_in_depth / dst_w;
            const int64_t dst_x      = p_in_depth % dst_w;

            char * dst_row = (char *) tmp + (p % patches_per_batch) * knl_n_total * traits->type_size;

            for (int64_t ic = 0; ic < c; ++ic) {
                for (int64_t kz = 0; kz < knl_d; ++kz) {
                    for (int64_t ky = 0; ky < knl_h; ++ky) {
                        for (int64_t kx = 0; kx < knl_w; ++kx) {
                            const int64_t sz = dst_z * s2 + kz * d2 - p2;
                            const int64_t sy = dst_y * s1 + ky * d1 - p1;
                            const int64_t sx = dst_x * s0 + kx * d0 - p0;

                            int64_t dst_idx = ic * knl_n_per_channel + kz * (knl_h * knl_w) + ky * knl_w + kx;

                            float src_val;
                            if (sz < 0 || sz >= src_d || sy < 0 || sy >= src_h || sx < 0 || sx >= src_w) {
                                src_val = 0.0f;
                            } else {
                                const int64_t cn_idx = batch_idx * c + ic;
                                const float * src_ptr = (const float *)((const char *)src_data + sx*src->nb[0] + sy*src->nb[1] + sz*src->nb[2] + cn_idx*src->nb[3]);
                                src_val = *src_ptr;
                            }

                            char * element_ptr = dst_row + dst_idx * traits->type_size;
                            if (kernel_type == GGML_TYPE_F32) {
                                *(float *)element_ptr = src_val;
                            } else if (kernel_type == GGML_TYPE_F16) {
                                *(ggml_fp16_t *)element_ptr = GGML_CPU_FP32_TO_FP16(src_val);
                            }
                        }
                    }
                }
            }
        }

        ggml_barrier(params->threadpool);

        float * gemm_output = (float *) ((char *) tmp + patches_per_batch * knl_n_total * traits->type_size);
        ggml_call_mul_mat(kernel_type, params, patch_n_in_batch, oc, knl_n_total, tmp, knl_data, gemm_output);

        ggml_barrier(params->threadpool);

        const int64_t permute_per_thread = (patch_n_in_batch + params->nth - 1) / params->nth;
        const int64_t permute_start = params->ith * permute_per_thread;
        const int64_t permute_end = std::min(permute_start + permute_per_thread, patch_n_in_batch);

        for (int64_t i = permute_start; i < permute_end; ++i) {
            const int64_t p = patch_start_batch + i;
            const int64_t p_in_batch = p % (dst_w * dst_h * dst_d);
            const int64_t p_in_depth = p_in_batch % (dst_w * dst_h);
            const int64_t batch_idx  = p / (dst_w * dst_h * dst_d);
            const int64_t dst_z      = p_in_batch / (dst_w * dst_h);
            const int64_t dst_y      = p_in_depth / dst_w;
            const int64_t dst_x      = p_in_depth % dst_w;

            for (int64_t ioc = 0; ioc < oc; ++ioc) {
                const float value = gemm_output[i * oc + ioc];
                const int64_t ocn_idx = batch_idx * oc + ioc;
                float * dst_ptr = (float *)((char *)dst_data + dst_x*dst->nb[0] + dst_y*dst->nb[1] + dst_z*dst->nb[2] + ocn_idx*dst->nb[3]);
                *dst_ptr = value;
            }
        }
    }
}

void ggml_compute_forward_conv_3d(
        const ggml_compute_params * params,
        ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    ggml_compute_forward_conv_3d_impl(params, src0, src1, dst, src0->type);
}

template <typename kernel_t>
static void ggml_compute_forward_conv_transpose_2d_impl(
    const ggml_compute_params * params,
          ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_BINARY_OP_LOCALS

    const int ith = params->ith;
    const int nth = params->nth;

    const int nk = ne00*ne01*ne02*ne03;

    GGML_ASSERT(nb00 == ggml_type_size(src0->type));
    GGML_ASSERT(nb10 == sizeof(float));

    if (ith == 0) {
        memset(params->wdata, 0, params->wsize);

        // permute kernel data (src0) from (Kw x Kh x Cout x Cin) to (Cin x Kw x Kh x Cout)
        {
            kernel_t * const wdata = (kernel_t *) params->wdata + 0;

            for (int64_t i03 = 0; i03 < ne03; i03++) {
                for (int64_t i02 = 0; i02 < ne02; i02++) {
                    const kernel_t * const src = (kernel_t *)((char *) src0->data + i03*nb03 + i02*nb02);
                    kernel_t * dst_data = wdata + i02*ne01*ne00*ne03;
                    for (int64_t i01 = 0; i01 < ne01; i01++) {
                        for (int64_t i00 = 0; i00 < ne00; i00++) {
                            dst_data[i01*ne00*ne03 + i00*ne03 + i03] = src[i01 * ne00 + i00];
                        }
                    }
                }
            }
        }

        // permute source data (src1) from (Sw x Sh x Cin) to (Cin x Sw x Sh)
        {
            kernel_t * const wdata = (kernel_t *) params->wdata + nk;
            for (int i12 = 0; i12 < ne12; i12++) {
                for (int i11 = 0; i11 < ne11; i11++) {
                    const float * const src = (float *)((char *) src1->data + i12*nb12 + i11*nb11);
                    kernel_t * dst_data = wdata + i11*ne10*ne12;
                    for (int i10 = 0; i10 < ne10; i10++) {
                        if constexpr (std::is_same_v<kernel_t, ggml_fp16_t>) {
                            dst_data[i10*ne12 + i12] = GGML_CPU_FP32_TO_FP16(src[i10]);
                        } else {
                            dst_data[i10*ne12 + i12] = src[i10];
                        }
                    }
                }
            }
        }

        memset(dst->data, 0, ggml_nbytes(dst));
    }
    ggml_barrier(params->threadpool);

    const int32_t stride = ggml_get_op_params_i32(dst, 0);

    // total patches in dst
    const int np = ne2;

    // patches per thread
    const int dp = (np + nth - 1)/nth;

    // patch range for this thread
    const int ip0 = dp*ith;
    const int ip1 = MIN(ip0 + dp, np);

    kernel_t * const wdata = (kernel_t *) params->wdata + 0;
    kernel_t * const wdata_src = wdata + nk;

    for (int i2 = ip0; i2 < ip1; i2++) { // Cout
        float * dst_data = (float *)((char *) dst->data + i2*nb2);
        kernel_t * wdata_kernel = wdata + i2*ne01*ne00*ne03;
        for (int i11 = 0; i11 < ne11; i11++) {
            for (int i10 = 0; i10 < ne10; i10++) {
                const int i1n = i11*ne10*ne12 + i10*ne12;
                for (int i01 = 0; i01 < ne01; i01++) {
                    for (int i00 = 0; i00 < ne00; i00++) {
                        float v = 0;
                        if constexpr (std::is_same_v<kernel_t, ggml_fp16_t>) {
                            ggml_vec_dot_f16(ne03, &v, 0,
                                    wdata_src + i1n, 0,
                                    wdata_kernel + i01*ne00*ne03 + i00*ne03, 0, 1);
                        } else {
                            ggml_vec_dot_f32(ne03, &v, 0,
                                    wdata_src + i1n, 0,
                                    wdata_kernel + i01*ne00*ne03 + i00*ne03, 0, 1);
                        }
                        dst_data[(i11*stride + i01)*ne0 + i10*stride + i00] += v;
                    }
                }
            }
        }
    }
}

void ggml_compute_forward_conv_transpose_2d(
        const ggml_compute_params * params,
              ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F16:
            {
                ggml_compute_forward_conv_transpose_2d_impl<ggml_fp16_t>(params, dst);
            } break;
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_conv_transpose_2d_impl<float>(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_conv_2d_dw

struct ggml_conv_2d_dw_params {
    int64_t channels;
    int64_t batch;
    int64_t src_w;
    int64_t src_h;
    int64_t dst_w;
    int64_t dst_h;
    int64_t knl_w;
    int64_t knl_h;
    int stride_x;
    int stride_y;
    int pad_x;
    int pad_y;
    int dilation_x;
    int dilation_y;
};

static inline float ggml_conv_2d_dw_knl_f32(const char * data, int64_t i, ggml_type type) {
    if (type == GGML_TYPE_F16) {
        return GGML_FP16_TO_FP32(((const ggml_fp16_t *)data)[i]);
    }
    return ((const float *)data)[i];
}

static void ggml_compute_forward_conv_2d_dw_cwhn(
        const ggml_compute_params * params,
        const ggml_tensor * src,
        const ggml_tensor * kernel,
        ggml_tensor * dst,
        const ggml_conv_2d_dw_params & p) {

    const int64_t c = p.channels;
    const char * knl_data = (const char *)kernel->data;
    const ggml_type knl_type = kernel->type;

    const int64_t rows_total = p.dst_h * p.batch;
    const int64_t rows_per_thread = (rows_total + params->nth - 1) / params->nth;
    const int64_t row_start = params->ith * rows_per_thread;
    const int64_t row_end = MIN(row_start + rows_per_thread, rows_total);

#ifdef GGML_SIMD
    int64_t c_pkg_end = 0;
    int64_t pkg_size = GGML_F32_EPR;
    if (knl_type == GGML_TYPE_F32) {
    #if defined(__ARM_FEATURE_SVE)
        pkg_size = svcntw();
    #else
        pkg_size = GGML_F32_EPR;
    #endif
        c_pkg_end = (c / pkg_size) * pkg_size;
    }
#else
    const int64_t c_pkg_end = 0;
#endif

    for (int64_t row = row_start; row < row_end; ++row) {
        const int64_t dst_y = row % p.dst_h;
        const float * src_data = (const float *)src->data + (row / p.dst_h) * p.src_w * p.src_h * c;
        for (int64_t dst_x = 0; dst_x < p.dst_w; ++dst_x) {
            float * dst_data = (float *)dst->data + (row * p.dst_w + dst_x) * c;
            const int64_t src_y_base = dst_y * p.stride_y - p.pad_y;
            const int64_t src_x_base = dst_x * p.stride_x - p.pad_x;

#ifdef GGML_SIMD
            for (int64_t c_i = 0; c_i < c_pkg_end; c_i += pkg_size) {
                GGML_F32_VEC sum = GGML_F32_VEC_ZERO;
                for (int64_t knl_y = 0; knl_y < p.knl_h; ++knl_y) {
                    const int64_t src_y = src_y_base + knl_y * p.dilation_y;
                    if (src_y < 0 || src_y >= p.src_h) {
                        continue;
                    }
                    for (int64_t knl_x = 0; knl_x < p.knl_w; ++knl_x) {
                        const int64_t src_x = src_x_base + knl_x * p.dilation_x;
                        if (src_x < 0 || src_x >= p.src_w) {
                            continue;
                        }
                        const float * kp = (const float *)knl_data + (knl_y * p.knl_w + knl_x) * c + c_i;
                        GGML_F32_VEC k = GGML_F32_VEC_LOAD(kp);
                        GGML_F32_VEC s = GGML_F32_VEC_LOAD(src_data + (src_y * p.src_w + src_x) * c + c_i);
                        sum = GGML_F32_VEC_FMA(sum, k, s);
                    }
                }
                GGML_F32_VEC_STORE(dst_data + c_i, sum);
            }
#endif
            for (int64_t c_i = c_pkg_end; c_i < c; ++c_i) {
                float sum = 0.0f;
                for (int64_t knl_y = 0; knl_y < p.knl_h; ++knl_y) {
                    const int64_t src_y = src_y_base + knl_y * p.dilation_y;
                    if (src_y < 0 || src_y >= p.src_h) {
                        continue;
                    }
                    for (int64_t knl_x = 0; knl_x < p.knl_w; ++knl_x) {
                        const int64_t src_x = src_x_base + knl_x * p.dilation_x;
                        if (src_x < 0 || src_x >= p.src_w) {
                            continue;
                        }
                        sum += ggml_conv_2d_dw_knl_f32(knl_data, (knl_y * p.knl_w + knl_x) * c + c_i, knl_type)
                             * src_data[(src_y * p.src_w + src_x) * c + c_i];
                    }
                }
                dst_data[c_i] = sum;
            }
        }
    }
}

static void ggml_compute_forward_conv_2d_dw_whcn(
        const ggml_compute_params * params,
        const ggml_tensor * src,
        const ggml_tensor * kernel,
        ggml_tensor * dst,
        const ggml_conv_2d_dw_params & p) {

    const int64_t n = p.channels * p.batch;
    const int64_t per_thread = (n + params->nth - 1) / params->nth;
    const int64_t start = params->ith * per_thread;
    const int64_t end = MIN(start + per_thread, n);
    const char * knl_base = (const char *)kernel->data;
    const ggml_type knl_type = kernel->type;

    for (int64_t i = start; i < end; ++i) {
        const int64_t knl_offset = (i % p.channels) * p.knl_w * p.knl_h;
        const float * src_data = (const float *)src->data + i * p.src_w * p.src_h;
        float * dst_data = (float *)dst->data + i * p.dst_w * p.dst_h;

        for (int64_t dst_y = 0; dst_y < p.dst_h; ++dst_y) {
            for (int64_t dst_x = 0; dst_x < p.dst_w; ++dst_x) {

                float sum = 0.0f;
                for (int64_t knl_y = 0; knl_y < p.knl_h; ++knl_y) {
                    const int64_t src_y = dst_y * p.stride_y + knl_y * p.dilation_y - p.pad_y;
                    if (src_y < 0 || src_y >= p.src_h) {
                        continue;
                    }
                    for (int64_t knl_x = 0; knl_x < p.knl_w; ++knl_x) {
                        const int64_t src_x = dst_x * p.stride_x + knl_x * p.dilation_x - p.pad_x;
                        if (src_x < 0 || src_x >= p.src_w) {
                            continue;
                        }
                        sum += ggml_conv_2d_dw_knl_f32(knl_base, knl_offset + knl_y * p.knl_w + knl_x, knl_type)
                             * src_data[src_y * p.src_w + src_x];
                    }
                }
                dst_data[dst_y * p.dst_w + dst_x] = sum;
            }
        }
    }
}

void ggml_compute_forward_conv_2d_dw(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * kernel = dst->src[0];
    const ggml_tensor * src = dst->src[1];
    ggml_conv_2d_dw_params p;
    p.channels = src->ne[2];
    p.batch = src->ne[3];
    p.src_w = src->ne[0];
    p.src_h = src->ne[1];
    p.dst_w = dst->ne[0];
    p.dst_h = dst->ne[1];
    p.knl_w = kernel->ne[0];
    p.knl_h = kernel->ne[1];
    p.stride_x = dst->op_params[0];
    p.stride_y = dst->op_params[1];
    p.pad_x = dst->op_params[2];
    p.pad_y = dst->op_params[3];
    p.dilation_x = dst->op_params[4];
    p.dilation_y = dst->op_params[5];

    GGML_ASSERT(kernel->type == GGML_TYPE_F32 || kernel->type == GGML_TYPE_F16);
    GGML_ASSERT(kernel->ne[3] == p.channels);
    GGML_ASSERT(dst->ne[3] == p.batch);

    if (ggml_is_contiguous(src)) {
        ggml_compute_forward_conv_2d_dw_whcn(params, src, kernel, dst, p);
    } else if (ggml_is_contiguous_channels(src)) {
        GGML_ASSERT(kernel->nb[0] >= kernel->nb[2] && kernel->nb[1] >= kernel->nb[0]);
        ggml_compute_forward_conv_2d_dw_cwhn(params, src, kernel, dst, p);
    } else {
        GGML_ABORT("non-contiguous memory layout not supported");
    }
}

// ggml_compute_forward_pool_1d_ksp
static void ggml_compute_forward_pool_1d_ksp(
        const ggml_compute_params * params,
        const ggml_op_pool op,
        const int k,
        const int s,
        const int p,
        ggml_tensor * dst) {

    const ggml_tensor * src = dst->src[0];

    assert(src->type == GGML_TYPE_F32 || src->type == GGML_TYPE_F16);

    if (params->ith != 0) {
        return;
    }

    const int64_t IW = src->ne[0];
    const int64_t OW = dst->ne[0];

    const int64_t nr = ggml_nrows(src);

    for (int64_t ir = 0; ir < nr; ++ir) {
        const char * srow_bytes =            (const char *) src->data + ir * src->nb[1];
        float      * drow       = (float *) ((      char *) dst->data + ir * dst->nb[1]);

        for (int64_t ow = 0; ow < OW; ++ow) {
            float res = 0;
            switch (op) {
                case GGML_OP_POOL_AVG: res = 0.0f;     break;
                case GGML_OP_POOL_MAX: res = -FLT_MAX; break;
                case GGML_OP_POOL_COUNT: GGML_ABORT("fatal error");
            }

            int count = 0;
            const int base = (int) ow * s - p;

            for (int ki = 0; ki < k; ++ki) {
                const int j = base + ki;
                if (j < 0 || j >= (int) IW) {
                    continue;
                }

                float v;
                if (src->type == GGML_TYPE_F32) {
                    v = ((const float *) srow_bytes)[j];
                } else {
                    v = GGML_CPU_FP16_TO_FP32(((const ggml_fp16_t *) srow_bytes)[j]);
                }

                switch (op) {
                    case GGML_OP_POOL_AVG: res += v;                break;
                    case GGML_OP_POOL_MAX: res =  std::max(v, res); break;
                    case GGML_OP_POOL_COUNT: GGML_ABORT("fatal error");
                }

                ++count;
            }

            switch (op) {
                case GGML_OP_POOL_AVG: res = (count > 0) ? (res / count) : 0.0f; break;
                case GGML_OP_POOL_MAX:                                           break;
                case GGML_OP_POOL_COUNT: GGML_ABORT("fatal error");
            }

            drow[ow] = res;
        }
    }
}

// ggml_compute_forward_pool_1d

void ggml_compute_forward_pool_1d(
        const ggml_compute_params * params,
              ggml_tensor * dst) {

    const int32_t * opts = (const int32_t *)dst->op_params;
    ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    const int k0 = opts[1];
    const int s0 = opts[2];
    const int p0 = opts[3];

    ggml_compute_forward_pool_1d_ksp(params, op, k0, s0, p0, dst);
}

// ggml_compute_forward_pool_2d

void ggml_compute_forward_pool_2d(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src = dst->src[0];

    assert(src->type == GGML_TYPE_F32 || src->type == GGML_TYPE_F16);

    if (params->ith != 0) {
        return;
    }

    const int32_t * opts = (const int32_t *)dst->op_params;

    ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    const int k0 = opts[1];
    const int k1 = opts[2];
    const int s0 = opts[3];
    const int s1 = opts[4];
    const int p0 = opts[5];
    const int p1 = opts[6];
    const char * cdata = (const char*)src->data;
    const char * const data_end = cdata + ggml_nbytes(src);

    const int64_t px = dst->ne[0];
    const int64_t py = dst->ne[1];
    const int64_t pa = px * py;

    float * dplane = (float *)dst->data;

    const int ka = k0 * k1;
    const int offset0 = -p0;
    const int offset1 = -p1;

    while (cdata < data_end) {
        for (int oy = 0; oy < py; ++oy) {
            float * const drow = dplane + oy * px;
            float * const out  = drow;

            for (int ox = 0; ox < px; ++ox) {
                float res = 0;
                switch (op) {
                    case GGML_OP_POOL_AVG: res = 0;        break;
                    case GGML_OP_POOL_MAX: res = -FLT_MAX; break;
                    case GGML_OP_POOL_COUNT: GGML_ABORT("fatal error");
                }

                const int ix = offset0 + ox * s0;
                const int iy = offset1 + oy * s1;

                for (int ky = 0; ky < k1; ++ky) {
                    if (iy + ky < 0 || iy + ky >= src->ne[1]) {
                        continue;
                    }

                    const void * srow = (const void *)(cdata + src->nb[1] * (iy + ky));
                    for (int kx = 0; kx < k0; ++kx) {
                        int j = ix + kx;
                        if (j < 0 || j >= src->ne[0]) {
                            continue;
                        }

                        const float srow_j = (src->type == GGML_TYPE_F32) ? ((const float*)srow)[j] : GGML_CPU_FP16_TO_FP32(((const ggml_fp16_t*)srow)[j]);
                        switch (op) {
                            case GGML_OP_POOL_AVG: res += srow_j;                break;
                            case GGML_OP_POOL_MAX: res =  std::max(srow_j, res); break;
                            case GGML_OP_POOL_COUNT:               GGML_ABORT("fatal error");
                        }
                    }
                }
                switch (op) {
                    case GGML_OP_POOL_AVG:           res /= ka; break;
                    case GGML_OP_POOL_MAX:                      break;
                    case GGML_OP_POOL_COUNT: GGML_ABORT("fatal error");
                }

                out[ox] = res;
            }
        }

        cdata  += src->nb[2];
        dplane += pa;
    }
}

// ggml_compute_forward_pool_2d_back

void ggml_compute_forward_pool_2d_back(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src  = dst->src[0];
    const ggml_tensor * dstf = dst->src[1]; // forward tensor of dst

    assert(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);

    if (params->ith != 0) {
        return;
    }

    const int32_t * opts = (const int32_t *)dst->op_params;
    ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    const int k0 = opts[1];
    const int k1 = opts[2];
    const int s0 = opts[3];
    const int s1 = opts[4];
    const int p0 = opts[5];
    const int p1 = opts[6];

    char       * cdata  = (char       *) dst->data;
    const char * cdataf = (const char *) dstf->data;
    const char * const data_end = cdata + ggml_nbytes(dst);

    GGML_ASSERT(params->ith == 0);
    memset(cdata, 0, ggml_nbytes(dst));

    const int64_t px = src->ne[0];
    const int64_t py = src->ne[1];
    const int64_t pa = px * py;

    const float * splane = (const float *) src->data;

    const int ka = k0 * k1;
    const int offset0 = -p0;
    const int offset1 = -p1;

    while (cdata < data_end) {
        for (int oy = 0; oy < py; ++oy) {
            const float * const srow = splane + oy * px;
            for (int ox = 0; ox < px; ++ox) {
                const float grad0 = srow[ox];

                const int ix = offset0 + ox * s0;
                const int iy = offset1 + oy * s1;

                if (op == GGML_OP_POOL_MAX) {
                    float maxval = -FLT_MAX;
                    int kxmax = -1;
                    int kymax = -1;

                    for (int ky = 0; ky < k1; ++ky) {
                        if (iy + ky < 0 || iy + ky >= dst->ne[1]) {
                            continue;
                        }
                        const void * drowf = (const void *)(cdataf + dst->nb[1] * (iy + ky));
                        for (int kx = 0; kx < k0; ++kx) {
                            int j = ix + kx;
                            if (j < 0 || j >= dst->ne[0]) {
                                continue;
                            }

                            const float val = dst->type == GGML_TYPE_F32 ?
                                ((const float *) drowf)[j] : GGML_CPU_FP16_TO_FP32(((const ggml_fp16_t *) drowf)[j]);
                            if (val <= maxval) {
                                continue;
                            }

                            maxval = val;
                            kxmax = kx;
                            kymax = ky;
                        }
                    }

                    if (kxmax == -1 || kymax == -1) {
                        continue;
                    }

                    void * drow = (void *)(cdata + dst->nb[1] * (iy + kymax));
                    const int j = ix + kxmax;
                    if (dst->type == GGML_TYPE_F32) {
                        ((float *) drow)[j] += grad0;
                    } else {
                        ((ggml_fp16_t *) drow)[j] = GGML_CPU_FP32_TO_FP16(grad0 + GGML_CPU_FP16_TO_FP32(((const ggml_fp16_t *) drow)[j]));
                    }
                } else if (op == GGML_OP_POOL_AVG) {
                    const float grad = grad0 / ka;

                    for (int ky = 0; ky < k1; ++ky) {
                        if (iy + ky < 0 || iy + ky >= dst->ne[1]) {
                            continue;
                        }
                        void * drow = (void *)(cdata + dst->nb[1] * (iy + ky));
                        for (int kx = 0; kx < k0; ++kx) {
                            int j = ix + kx;
                            if (j < 0 || j >= dst->ne[0]) {
                                continue;
                            }

                            if (dst->type == GGML_TYPE_F32) {
                                ((float *) drow)[j] += grad;
                            } else {
                                ((ggml_fp16_t *) drow)[j] += GGML_CPU_FP32_TO_FP16(grad);
                            }
                        }
                    }
                } else {
                    GGML_ASSERT(false);
                }
            }
        }

        cdata  += dst->nb[2];
        cdataf += dst->nb[2];
        splane += pa;
    }
}

// ggml_compute_forward_upscale

static void ggml_compute_forward_upscale_f32(
    const ggml_compute_params * params,
    ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);

    const int ith = params->ith;
    const int nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    float sf0 = (float)ne0/src0->ne[0];
    float sf1 = (float)ne1/src0->ne[1];
    float sf2 = (float)ne2/src0->ne[2];
    float sf3 = (float)ne3/src0->ne[3];
    float pixel_offset = 0.5f;

    const int32_t mode_flags = ggml_get_op_params_i32(dst, 0);
    const ggml_scale_mode mode = (ggml_scale_mode) (mode_flags & 0xFF);

    if (mode_flags & GGML_SCALE_FLAG_ALIGN_CORNERS) {
        pixel_offset = 0.0f;
        sf0 = ne0 > 1 && ne00 > 1 ? (float)(ne0 - 1) / (ne00 - 1) : sf0;
        sf1 = ne1 > 1 && ne01 > 1 ? (float)(ne1 - 1) / (ne01 - 1) : sf1;
    }

    if (mode == GGML_SCALE_MODE_NEAREST) {
        for (int64_t i3 = 0; i3 < ne3; i3++) {
            const int64_t i03 = i3 / sf3;
            for (int64_t i2 = ith; i2 < ne2; i2 += nth) {
                const int64_t i02 = i2 / sf2;
                for (int64_t i1 = 0; i1 < ne1; i1++) {
                    const int64_t i01 = i1 / sf1;
                    for (int64_t i0 = 0; i0 < ne0; i0++) {
                        const int64_t i00 = i0 / sf0;

                        const float * x = (float *)((char *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                              float * y = (float *)((char *)  dst->data +  i0*nb0  +  i1*nb1  +  i2*nb2  +  i3*nb3);

                        *y = *x;
                    }
                }
            }
        }
    } else if (mode == GGML_SCALE_MODE_BILINEAR && (mode_flags & GGML_SCALE_FLAG_ANTIALIAS)) {
        // Similar to F.interpolate(..., mode="bilinear", align_corners=False, antialias=True)
        // https://github.com/pytorch/pytorch/blob/8871ff29b743948d1225389d5b7068f37b22750b/aten/src/ATen/native/cpu/UpSampleKernel.cpp
        auto triangle_filter = [](float x) -> float {
            return std::max(1.0f - fabsf(x), 0.0f);
        };

        // support and invscale, minimum 1 pixel for bilinear
        const float support1  = std::max(1.0f, 1.0f / sf1);
        const float invscale1 = 1.0f / support1;
        const float support0  = std::max(1.0f, 1.0f / sf0);
        const float invscale0 = 1.0f / support0;

        for (int64_t i3 = 0; i3 < ne3; i3++) {
            const int64_t i03 = i3 / sf3;
            for (int64_t i2 = ith; i2 < ne2; i2 += nth) {
                const int64_t i02 = i2 / sf2;
                for (int64_t i1 = 0; i1 < ne1; i1++) {
                    const float y = ((float) i1 + pixel_offset) / sf1;
                    for (int64_t i0 = 0; i0 < ne0; i0++) {
                        const float x = ((float) i0 + pixel_offset) / sf0;

                        // the range of source pixels that contribute
                        const int64_t x_min = std::max<int64_t>(x - support0 + pixel_offset, 0);
                        const int64_t x_max = std::min<int64_t>(x + support0 + pixel_offset, ne00);
                        const int64_t y_min = std::max<int64_t>(y - support1 + pixel_offset, 0);
                        const int64_t y_max = std::min<int64_t>(y + support1 + pixel_offset, ne01);

                        // bilinear filter with antialiasing
                        float val = 0.0f;
                        float total_weight = 0.0f;

                        for (int64_t sy = y_min; sy < y_max; sy++) {
                            const float weight_y = triangle_filter((sy - y + pixel_offset) * invscale1);

                            for (int64_t sx = x_min; sx < x_max; sx++) {
                                const float weight_x = triangle_filter((sx - x + pixel_offset) * invscale0);
                                const float weight = weight_x * weight_y;

                                if (weight <= 0.0f) {
                                    continue;
                                }

                                const float pixel = *(const float *)((const char *)src0->data + sx*nb00 + sy*nb01 + i02*nb02 + i03*nb03);
                                val += pixel * weight;
                                total_weight += weight;
                            }
                        }

                        if (total_weight > 0.0f) {
                            val /= total_weight;
                        }

                        float * dst_ptr = (float *)((char *)dst->data + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);
                        *dst_ptr = val;
                    }
                }
            }
        }
    } else if (mode == GGML_SCALE_MODE_BILINEAR) {
        for (int64_t i3 = 0; i3 < ne3; i3++) {
            const int64_t i03 = i3 / sf3;
            for (int64_t i2 = ith; i2 < ne2; i2 += nth) {
                const int64_t i02 = i2 / sf2;
                for (int64_t i1 = 0; i1 < ne1; i1++) {
                    const float y = ((float)i1 + pixel_offset) / sf1 - pixel_offset;
                    int64_t y0 = (int64_t)floorf(y);
                    int64_t y1 = y0 + 1;

                    y0 = std::max(int64_t(0), std::min(y0, ne01 - 1));
                    y1 = std::max(int64_t(0), std::min(y1, ne01 - 1));

                    float dy = y - (float)y0;
                    dy = std::max(0.0f, std::min(dy, 1.0f));

                    for (int64_t i0 = 0; i0 < ne0; i0++) {
                        const float x = ((float)i0 + pixel_offset) / sf0 - pixel_offset;
                        int64_t x0 = (int64_t)floorf(x);
                        int64_t x1 = x0 + 1;

                        x0 = std::max(int64_t(0), std::min(x0, ne00 - 1));
                        x1 = std::max(int64_t(0), std::min(x1, ne00 - 1));

                        float dx = x - (float)x0;
                        dx = std::max(0.0f, std::min(dx, 1.0f));

                        // fetch the four surrounding pixel values and interpolate
                        const float a = *(const float *)((const char *)src0->data + x0*nb00 + y0*nb01 + i02*nb02 + i03*nb03);
                        const float b = *(const float *)((const char *)src0->data + x1*nb00 + y0*nb01 + i02*nb02 + i03*nb03);
                        const float c = *(const float *)((const char *)src0->data + x0*nb00 + y1*nb01 + i02*nb02 + i03*nb03);
                        const float d = *(const float *)((const char *)src0->data + x1*nb00 + y1*nb01 + i02*nb02 + i03*nb03);

                        const float val = a*(1 - dx)*(1 - dy) + b*dx*(1 - dy) + c*(1 - dx)*dy + d*dx*dy;

                        float * y_dst = (float *)((char *)dst->data + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);
                        *y_dst = val;
                    }
                }
            }
        }
    } else if (mode == GGML_SCALE_MODE_BICUBIC) {
        // https://en.wikipedia.org/wiki/Bicubic_interpolation#Bicubic_convolution_algorithm
        const float a = -0.75f; // use alpha = -0.75 (same as PyTorch)
        auto weight1 = [a](float x) { return ((a + 2) * x - (a + 3)) * x * x + 1; };
        auto weight2 = [a](float x) { return ((a * x - 5 * a) * x + 8 * a) * x - 4 * a; };
        auto bicubic = [=](float p0, float p1, float p2, float p3, float x) {
            const float w0 = weight2(x + 1);
            const float w1 = weight1(x + 0);
            const float w2 = weight1(1 - x);
            const float w3 = weight2(2 - x);
            return p0*w0 + p1*w1 + p2*w2 + p3*w3;
        };

        for (int64_t i3 = 0; i3 < ne3; i3++) {
            const int64_t i03 = i3 / sf3;
            for (int64_t i2 = ith; i2 < ne2; i2 += nth) {
                const int64_t i02 = i2 / sf2;
                for (int64_t i1 = 0; i1 < ne1; i1++) {
                    const float y = ((float)i1 + pixel_offset) / sf1 - pixel_offset;
                    const int64_t y0 = (int64_t)floorf(y);
                    const float dy = y - (float)y0;

                    for (int64_t i0 = 0; i0 < ne0; i0++) {
                        const float x = ((float)i0 + pixel_offset) / sf0 - pixel_offset;
                        const int64_t x0 = (int64_t)floorf(x);
                        const float dx = x - (float)x0;

                        auto p = [=](int64_t x_off, int64_t y_off) -> float {
                            int64_t i00 = std::max(int64_t(0), std::min(x0 + x_off, ne00 - 1));
                            int64_t i01 = std::max(int64_t(0), std::min(y0 + y_off, ne01 - 1));
                            return *(const float *)((const char *)src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                        };

                        const float val = bicubic(
                            bicubic(p(-1,-1), p(0,-1), p(1,-1), p(2,-1), dx),
                            bicubic(p(-1, 0), p(0, 0), p(1, 0), p(2, 0), dx),
                            bicubic(p(-1, 1), p(0, 1), p(1, 1), p(2, 1), dx),
                            bicubic(p(-1, 2), p(0, 2), p(1, 2), p(2, 2), dx), dy);

                        float * y_dst = (float *)((char *)dst->data + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);
                        *y_dst = val;
                    }
                }
            }
        }
    } else {
        GGML_ABORT("unsupported upscale mode");
    }
}

void ggml_compute_forward_upscale(
    const ggml_compute_params * params,
    ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_upscale_f32(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}


// ggml_compute_forward_pad

template<bool circular_t>
static void ggml_compute_forward_pad_f32(
    const ggml_compute_params * params,
          ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    assert(dst->nb[0] == sizeof(float));

    const int ith = params->ith;
    const int nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    float * dst_ptr = (float *) dst->data;
    const int32_t lp0 = ggml_get_op_params_i32(dst, 0);
    const int32_t rp0 = ggml_get_op_params_i32(dst, 1);
    const int32_t lp1 = ggml_get_op_params_i32(dst, 2);
    const int32_t rp1 = ggml_get_op_params_i32(dst, 3);
    const int32_t lp2 = ggml_get_op_params_i32(dst, 4);
    const int32_t rp2 = ggml_get_op_params_i32(dst, 5);
    const int32_t lp3 = ggml_get_op_params_i32(dst, 6);
    const int32_t rp3 = ggml_get_op_params_i32(dst, 7);

    // TODO: optimize

    for (int64_t i2 = 0; i2 < ne2; ++i2) {
        for (int64_t i1 = ith; i1 < ne1; i1 += nth) {
            for (int64_t i0 = 0; i0 < ne0; ++i0) {
                for (int64_t i3 = 0; i3 < ne3; ++i3) {
                    // circular means wrap around on a torus, so x and y loop around
                    if constexpr (circular_t) {
                        const int64_t dst_idx = i3*(ne0*ne1*ne2) + i2*(ne0*ne1) + i1*ne0 + i0;
                        const int64_t src_i0 = ggml_wrap_around(i0 - lp0, ne00);
                        const int64_t src_i1 = ggml_wrap_around(i1 - lp1, ne01);
                        const int64_t src_i2 = ggml_wrap_around(i2 - lp2, ne02);
                        const int64_t src_i3 = ggml_wrap_around(i3 - lp3, ne03);

                        const int64_t src_idx =
                            src_i3*nb03 +
                            src_i2*nb02 +
                            src_i1*nb01 +
                            src_i0*nb00;

                        const float * src_ptr = (const float *)((char *) src0->data + src_idx);
                        dst_ptr[dst_idx] = *src_ptr;
                    } else {
                        const int64_t dst_idx = i3*(ne0*ne1*ne2) + i2*(ne0*ne1) + i1*ne0 + i0;
                        if ((i0 >= lp0 && i0 < ne0 - rp0) \
                            && (i1 >= lp1 && i1 < ne1 - rp1) \
                            && (i2 >= lp2 && i2 < ne2 - rp2) \
                            && (i3 >= lp3 && i3 < ne3 - rp3)) {
                            const int64_t src_idx = (i3 - lp3)*nb03 + (i2 - lp2)*nb02 + (i1 - lp1)*nb01 + (i0 - lp0)*nb00;
                            const float * src_ptr = (const float *)((char *) src0->data + src_idx);
                            dst_ptr[dst_idx] = *src_ptr;
                        } else {
                            dst_ptr[dst_idx] = 0;
                        }
                    }
                }
            }
        }
    }
}


void ggml_compute_forward_pad(
    const ggml_compute_params * params,
    ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const bool circular = (bool) ggml_get_op_params_i32(dst, 8);
    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                if (circular) {
                    ggml_compute_forward_pad_f32<true>(params, dst);
                } else {
                    ggml_compute_forward_pad_f32<false>(params, dst);
                }
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_pad_reflect_1d

void ggml_compute_forward_pad_reflect_1d(
        const ggml_compute_params * params,
              ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    const int ith = params->ith;
    const int nth = params->nth;

    const int32_t * opts = (const int32_t *) dst->op_params;
    const int p0 = opts[0];
    const int p1 = opts[1];

    GGML_TENSOR_UNARY_OP_LOCALS

    for (int64_t i3 = 0; i3 < ne3; i3++) {
        for (int64_t i2 = 0; i2 < ne2; i2++) {
            for (int64_t i1 = ith; i1 < ne1; i1 += nth) {
                float * left  = (float *) ((char *) dst->data + i3*nb3 + i2*nb2 + i1*nb1 +         p0*nb0);
                float * right = (float *) ((char *) dst->data + i3*nb3 + i2*nb2 + i1*nb1 + (ne0-p1-1)*nb0);

                ggml_vec_cpy_f32(ne00, left, (float *) ((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01));

                for (int i0 = 1; i0 <= p0; i0++) { left[-i0] = left[i0];   }
                for (int i0 = 1; i0 <= p1; i0++) { right[i0] = right[-i0]; }
            }
        }
    }
}

// ggml_compute_forward_roll

static int64_t ggml_wrap_index(int64_t i, int64_t ne) {
    if (i < 0) {
        return i + ne;
    } else if (i >= ne) {
        return i - ne;
    }
    return i;
}

static void ggml_compute_forward_roll_f32(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];
    const float * src_data = (const float *) src0->data;
    float * dst_data = (float *) dst->data;

    GGML_TENSOR_UNARY_OP_LOCALS

    const int s0 = ggml_get_op_params_i32(dst, 0);
    const int s1 = ggml_get_op_params_i32(dst, 1);
    const int s2 = ggml_get_op_params_i32(dst, 2);
    const int s3 = ggml_get_op_params_i32(dst, 3);

    const int64_t total = ne1 * ne2 * ne3;
    const int64_t per_thread = (total + params->nth) / params->nth;
    const int64_t start = params->ith * per_thread;
    const int64_t end   = std::min(start + per_thread, total);

    for (int64_t i = start; i < end; ++i) {
        const int64_t i1 = i % ne1;
        const int64_t i2 = (i / ne1) % ne2;
        const int64_t i3 = i / (ne2 * ne1);
        float * dst_row = dst_data + (i3*nb3 + i2*nb2 + i1*nb1) / sizeof(float);

        const int64_t i01 = ggml_wrap_index(i1 - s1, ne01);
        const int64_t i02 = ggml_wrap_index(i2 - s2, ne02);
        const int64_t i03 = ggml_wrap_index(i3 - s3, ne03);
        const float * src_row = src_data + (i03*nb03 + i02*nb02 + i01*nb01) / sizeof(float);

        const int64_t s = ggml_wrap_index(-s0, ne00);
        const int64_t n = ne00 - s;
        ggml_vec_cpy_f32(n, dst_row,     src_row + s);
        ggml_vec_cpy_f32(s, dst_row + n, src_row);
    }
}

void ggml_compute_forward_roll(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_roll_f32(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_arange

static void ggml_compute_forward_arange_f32(
    const ggml_compute_params * params,
    ggml_tensor * dst) {

    GGML_ASSERT(dst->nb[0] == sizeof(float));

    const int ith = params->ith;
    const int nth = params->nth;

    const float start = ggml_get_op_params_f32(dst, 0);
    const float stop  = ggml_get_op_params_f32(dst, 1);
    const float step  = ggml_get_op_params_f32(dst, 2);

    const int64_t steps = (int64_t) ceilf((stop - start) / step);

    GGML_ASSERT(ggml_nelements(dst) == steps);

    for (int64_t i = ith; i < steps; i+= nth) {
        float value = start + step * i;
        ((float *)dst->data)[i] = value;
    }
}

void ggml_compute_forward_arange(
    const ggml_compute_params * params,
    ggml_tensor * dst) {
    switch (dst->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_arange_f32(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

static void ggml_compute_forward_timestep_embedding_f32(
    const ggml_compute_params * params,
    ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->nb[0] == sizeof(float));

    const int ith = params->ith;
    const int nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    const int dim = ggml_get_op_params_i32(dst, 0);
    const int max_period = ggml_get_op_params_i32(dst, 1);

    int half = dim / 2;

    for (int64_t i = 0; i < ne00; i++) {
        float * embed_data = (float *)((char *)  dst->data +  i*nb1);
        for (int64_t j = ith; j < half; j += nth) {
            float timestep = ((float *)src0->data)[i];
            float freq = (float)expf(-logf(max_period) * j / half);
            float arg = timestep * freq;
            embed_data[j] = cosf(arg);
            embed_data[j + half] = sinf(arg);
        }
        if (dim % 2 != 0 && ith == 0) {
            embed_data[2 * half] = 0.f;
        }
    }
}

void ggml_compute_forward_timestep_embedding(
    const ggml_compute_params * params,
    ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_timestep_embedding_f32(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_argsort

template<enum ggml_sort_order order>
struct cmp_argsort {
    const float * data;
    bool operator()(int32_t a, int32_t b) const {
        if constexpr (order == GGML_SORT_ORDER_ASC) {
            return data[a] < data[b];
        } else {
            return data[a] > data[b];
        }
    }
};

static void ggml_compute_forward_argsort_f32(
    const ggml_compute_params * params,
    ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(nb0 == sizeof(float));

    const int ith = params->ith;
    const int nth = params->nth;

    const int64_t nr = ggml_nrows(src0);

    ggml_sort_order order = (ggml_sort_order) ggml_get_op_params_i32(dst, 0);

    for (int64_t i = ith; i < nr; i += nth) {
        const float * src_data = (float *)((char *) src0->data + i*nb01);

        int32_t * dst_data = (int32_t *)((char *) dst->data + i*nb1);

        for (int64_t j = 0; j < ne0; j++) {
            dst_data[j] = j;
        }

        switch (order) {
            case GGML_SORT_ORDER_ASC:
                std::sort(dst_data, dst_data + ne0, cmp_argsort<GGML_SORT_ORDER_ASC>{src_data});
                break;

            case GGML_SORT_ORDER_DESC:
                std::sort(dst_data, dst_data + ne0, cmp_argsort<GGML_SORT_ORDER_DESC>{src_data});
                break;

            default:
                GGML_ABORT("invalid sort order");
        }
    }
}

void ggml_compute_forward_argsort(
    const ggml_compute_params * params,
    ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_argsort_f32(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_top_k

struct cmp_top_k {
    const float * data;
    bool operator()(int32_t a, int32_t b) const {
        return data[a] > data[b];
    }
};

static void ggml_compute_forward_top_k_f32(
    const ggml_compute_params * params,
    ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(nb0 == sizeof(float));

    const int ith = params->ith;
    const int nth = params->nth;

    const int64_t nr = ggml_nrows(src0);

    const int top_k = ne0;

    int32_t * tmp = (int32_t *) params->wdata + (ne00 + CACHE_LINE_SIZE_F32) * ith;

    for (int64_t i = ith; i < nr; i += nth) {
        const float * src_data = (float *)((char *) src0->data + i*nb01);

        for (int64_t j = 0; j < ne00; j++) {
            tmp[j] = j;
        }

        std::partial_sort(tmp, tmp + top_k, tmp + ne00, cmp_top_k{src_data});

        int32_t * dst_data = (int32_t *)((char *) dst->data + i*nb1);

        std::copy(tmp, tmp + top_k, dst_data);

        // emphasize that the order is not important
        if (top_k > 1) {
            std::swap(dst_data[0], dst_data[1]);
        }
    }
}

void ggml_compute_forward_top_k(
    const ggml_compute_params * params,
    ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_top_k_f32(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

static void ggml_compute_forward_flash_attn_ext_f16_one_chunk(
        const ggml_compute_params * params,
        ggml_tensor * dst,
        int ir0, int ir1,
        int64_t ic_start, int64_t ic_end,
        float * partials, int64_t partial_stride) {

    const bool write_partials = (partials != nullptr);
    const ggml_tensor * q     = dst->src[0];
    const ggml_tensor * k     = dst->src[1];
    const ggml_tensor * v     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    GGML_TENSOR_LOCALS(int64_t, neq, q,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbq, q,   nb)
    GGML_TENSOR_LOCALS(int64_t, nek, k,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbk, k,   nb)
    GGML_TENSOR_LOCALS(int64_t, nev, v,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbv, v,   nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst, ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst, nb)

    const int64_t DK = nek0;
    const int64_t DV = nev0;
    const int64_t N  = neq1;

    GGML_ASSERT(ne0 == DV);
    GGML_ASSERT(ne2 == N);

    // input tensor rows must be contiguous
    GGML_ASSERT(nbq0 == ggml_type_size(q->type));
    GGML_ASSERT(nbk0 == ggml_type_size(k->type));
    GGML_ASSERT(nbv0 == ggml_type_size(v->type));

    GGML_ASSERT(neq0 == DK);
    GGML_ASSERT(nek0 == DK);
    GGML_ASSERT(nev0 == DV);

    GGML_ASSERT(neq1 == N);

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // broadcast factors
    const int64_t rk2 = neq2/nek2;
    const int64_t rk3 = neq3/nek3;

    const int64_t rv2 = neq2/nev2;
    const int64_t rv3 = neq3/nev3;

    // parallelize by q rows using ggml_vec_dot_f32

    float scale         = 1.0f;
    float max_bias      = 0.0f;
    float logit_softcap = 0.0f;

    memcpy(&scale,         (float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias,      (float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (float *) dst->op_params + 2, sizeof(float));

    if (logit_softcap != 0) {
        scale /= logit_softcap;
    }

    const uint32_t n_head      = neq2;
    const uint32_t n_head_log2 = 1u << (uint32_t) floor(log2(n_head));

    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    ggml_type         const k_vec_dot_type = ggml_get_type_traits_cpu(k->type)->vec_dot_type;
    ggml_from_float_t const q_to_vec_dot   = ggml_get_type_traits_cpu(k_vec_dot_type)->from_float;
    ggml_vec_dot_t    const kq_vec_dot     = ggml_get_type_traits_cpu(k->type)->vec_dot;
    ggml_to_float_t   const v_to_float     = ggml_get_type_traits(v->type)->to_float;

    GGML_ASSERT((                            q_to_vec_dot) && "fattn: unsupported K-type");
    GGML_ASSERT((v->type == GGML_TYPE_F32 || v_to_float  ) && "fattn: unsupported V-type");

    int ith = params->ith;

    for (int ir = ir0; ir < ir1; ++ir) {
        // q indices
        const int iq3 = ir/(neq2*neq1);
        const int iq2 = (ir - iq3*neq2*neq1)/neq1;
        const int iq1 = (ir - iq3*neq2*neq1 - iq2*neq1);

        const uint32_t h = iq2; // head index
        const float slope = (max_bias > 0.0f) ? h < n_head_log2 ? powf(m0, h + 1) : powf(m1, 2*(h - n_head_log2) + 1) : 1.0f;

        float S = 0.0f;      // sum
        float M = -INFINITY; // maximum KQ value

        float       * VKQ32 = (float       *) params->wdata + ith*(1*DK + 2*DV + CACHE_LINE_SIZE_F32); // FP32 VKQ accumulator
        float       * V32   =                 (VKQ32 + 1*DV); // (temporary) FP32 V buffer
        ggml_fp16_t * VKQ16 = (ggml_fp16_t *) (VKQ32 + 1*DV); // (temporary) FP16 VKQ accumulator
        ggml_fp16_t * Q_q   = (ggml_fp16_t *) (VKQ32 + 2*DV); // (temporary) buffer for Q converted to quantized/FP16

        if (v->type == GGML_TYPE_F16) {
            memset(VKQ16, 0, DV*sizeof(ggml_fp16_t));
        } else {
            memset(VKQ32, 0, DV*sizeof(float));
        }

        const ggml_fp16_t * mp = mask ? (ggml_fp16_t *)((char *) mask->data + iq1*mask->nb[1] + (iq2%mask->ne[2])*mask->nb[2] + (iq3%mask->ne[3])*mask->nb[3]) : NULL;

        // k indices
        const int ik3 = iq3 / rk3;
        const int ik2 = iq2 / rk2;

        // v indices
        const int iv3 = iq3 / rv3;
        const int iv2 = iq2 / rv2;

        const float * pq = (const float *) ((char *) q->data + (iq1*nbq1 + iq2*nbq2 + iq3*nbq3));
        q_to_vec_dot(pq, Q_q, DK);

        // online softmax / attention
        // loop over n_kv and n_head_kv
        // ref: https://arxiv.org/pdf/2112.05682.pdf

        for (int64_t ic = ic_start; ic < ic_end; ++ic) {
            const float mv = mp ? slope*GGML_CPU_FP16_TO_FP32(mp[ic]) : 0.0f;
            if (mv == -INFINITY) {
                continue;
            }

            float s; // KQ value

            const char * k_data = (const char *) k->data + ( ic*nbk1 + ik2*nbk2 + ik3*nbk3);
            kq_vec_dot(DK, &s, 0, k_data, 0, Q_q, 0, 1);

            s = s*scale; // scale KQ value

            if (logit_softcap != 0.0f) {
                s = logit_softcap*tanhf(s);
            }

            s += mv; // apply mask

            const float Mold = M;

            float ms = 1.0f; // upon new higher max val, scale VKQ and KQ sum with this value
            float vs = 1.0f; // post-softmax KQ value, expf(s - M)

            const char * v_data = ((const char *) v->data + (ic*nbv1 + iv2*nbv2 + iv3*nbv3));

            if (v->type == GGML_TYPE_F16) {
                if (s > M) {
                    // s is new maximum, ms < 1.0f, vs == expf(s - s) == 1.0f
                    M = s;
                    ms = expf(Mold - M);

                    // V = V*expf(Mold - M)
                    ggml_vec_scale_f16(DV, VKQ16, ms);
                } else {
                    // no new maximum, ms == 1.0f, vs != 1.0f
                    vs = expf(s - M);
                }

                // V += v*expf(s - M)
                ggml_vec_mad_f16(DV, VKQ16, (const ggml_fp16_t *) v_data, vs);
            } else {
                if (s > M) {
                    // s is new maximum, ms < 1.0f, vs == expf(s - s) == 1.0f
                    M = s;
                    ms = expf(Mold - M);

                    // V = V*expf(Mold - M)
                    ggml_vec_scale_f32(DV, VKQ32, ms);
                } else {
                    // no new maximum, ms == 1.0f, vs != 1.0f
                    vs = expf(s - M);
                }

                // V += v*expf(s - M)
                if (v_to_float) {
                    v_to_float(v_data, V32, DV);
                    ggml_vec_mad_f32(DV, VKQ32, V32, vs);
                } else {
                    // V is F32
                    ggml_vec_mad_f32(DV, VKQ32, (const float *) v_data, vs);
                }
            }

            S = S*ms + vs; // scale and increment sum with partial sum
        }

        if (v->type == GGML_TYPE_F16) {
            for (int64_t d = 0; d < DV; ++d) {
                VKQ32[d] = GGML_CPU_FP16_TO_FP32(VKQ16[d]);
            }
        }

        // sinks - apply only on the first kv-chunk
        if (sinks && ic_start == 0) {
            const float s = ((float *)((char *) sinks->data))[h];

            float ms = 1.0f;
            float vs = 1.0f;

            if (s > M) {
                ms = expf(M - s);
                M = s;
                ggml_vec_scale_f32(DV, VKQ32, ms);
            } else {
                vs = expf(s - M);
            }

            S = S*ms + vs;
        }

        if (write_partials) {
            // Write M, S, VKQ to partials for later reduction
            // partials layout: [M, S, VKQ[DV]] per query head
            float * partial = partials + ir * partial_stride;
            partial[0] = M;
            partial[1] = S;
            memcpy(partial + 2, VKQ32, DV * sizeof(float));
        } else {
            // V /= S
            const float S_inv = S == 0.0f ? 0.0f : 1.0f/S;
            ggml_vec_scale_f32(DV, VKQ32, S_inv);

            // dst indices
            const int i1 = iq1;
            const int i2 = iq2;
            const int i3 = iq3;

            // permute(0, 2, 1, 3)
            memcpy((char *) dst->data + (i3*ne2*ne1 + i2 + i1*ne1)*nb1, VKQ32, nb1);
        }
    }
}

static void ggml_compute_forward_flash_attn_ext_tiled(
        const ggml_compute_params * params,
        ggml_tensor * dst,
        int ir0, int ir1) {
    const ggml_tensor * q     = dst->src[0];
    const ggml_tensor * k     = dst->src[1];
    const ggml_tensor * v     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    GGML_TENSOR_LOCALS(int64_t, neq, q,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbq, q,   nb)
    GGML_TENSOR_LOCALS(int64_t, nek, k,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbk, k,   nb)
    GGML_TENSOR_LOCALS(int64_t, nev, v,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbv, v,   nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst, ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst, nb)

    const int64_t DK = nek0;
    const int64_t DV = nev0;
    const int64_t N  = neq1;

    GGML_ASSERT(ne0 == DV);
    GGML_ASSERT(ne2 == N);

    // input tensor rows must be contiguous
    GGML_ASSERT(nbq0 == ggml_type_size(q->type));
    GGML_ASSERT(nbk0 == ggml_type_size(k->type));
    GGML_ASSERT(nbv0 == ggml_type_size(v->type));

    GGML_ASSERT(neq0 == DK);
    GGML_ASSERT(nek0 == DK);
    GGML_ASSERT(nev0 == DV);

    GGML_ASSERT(neq1 == N);

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    GGML_ASSERT(k->type == v->type);
    const ggml_type kv_type = k->type;


    // broadcast factors
    const int64_t rk2 = neq2/nek2;
    const int64_t rk3 = neq3/nek3;

    const int64_t rv2 = neq2/nev2;
    const int64_t rv3 = neq3/nev3;

    float scale         = 1.0f;
    float max_bias      = 0.0f;
    float logit_softcap = 0.0f;

    memcpy(&scale,         (float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias,      (float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (float *) dst->op_params + 2, sizeof(float));

    if (logit_softcap != 0) {
        scale /= logit_softcap;
    }

    const uint32_t n_head      = neq2;
    const uint32_t n_head_log2 = 1u << (uint32_t) floor(log2(n_head));

    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    int ith = params->ith;

    static constexpr int Q_TILE_SZ  = ggml_fa_tile_config::Q;
    static constexpr int KV_TILE_SZ = ggml_fa_tile_config::KV;

    int ir = ir0;
    while (ir < ir1) {
        // q indices for the start of this tile
        const int iq3 = ir/(neq2*neq1);
        const int iq2 = (ir - iq3*neq2*neq1)/neq1;
        const int iq1 = (ir - iq3*neq2*neq1 - iq2*neq1);

        // Number of valid rows in this tile:
        // - limited by tile size (Q_TILE_SZ)
        // - limited by chunk boundary (ir1 - ir)
        // - limited by head boundary (neq1 - iq1) to avoid crossing into next head
        const int tile_rows = MIN(Q_TILE_SZ, MIN((int)(ir1 - ir), (int)(neq1 - iq1)));
        GGML_ASSERT(tile_rows > 0);

        const uint32_t h = iq2; // head index
        const float slope = (max_bias > 0.0f) ? h < n_head_log2 ? powf(m0, h + 1) : powf(m1, 2*(h - n_head_log2) + 1) : 1.0f;

        float S[Q_TILE_SZ];
        float M[Q_TILE_SZ];

        for (int i = 0 ; i < Q_TILE_SZ; ++i) {
            S[i] = 0.;
            M[i] = -INFINITY;
        }

        // Per-thread scratch layout:
        // Q_q:    Q_TILE_SZ * DK (converted Q tile — F32 for GEMM, KV type for scalar)
        // KQ:     Q_TILE_SZ * KV_TILE_SZ (attention scores in float)
        // mask:   Q_TILE_SZ * KV_TILE_SZ (mask in float)
        // VKQ32:  Q_TILE_SZ * DV (FP32 output accumulator)
        // V32:    KV_TILE_SZ * DV (F32 buffer for V tile)
        // K_f32:  KV_TILE_SZ * DK (F32 buffer for K tile — GEMM path)
        float * base  = (float *) params->wdata + ith*(Q_TILE_SZ*DK + 2*Q_TILE_SZ*KV_TILE_SZ + Q_TILE_SZ*DV + KV_TILE_SZ*DV + KV_TILE_SZ*DK + CACHE_LINE_SIZE_F32);

        void  * Q_q    = base;
        float * KQ     = (float *)((char *)base + Q_TILE_SZ * DK * sizeof(float));
        float * mask32 = KQ + Q_TILE_SZ * KV_TILE_SZ;
        float * VKQ32  = mask32 + Q_TILE_SZ * KV_TILE_SZ;
        float * V32    = VKQ32 + Q_TILE_SZ * DV;
        float * K_f32  = V32 + KV_TILE_SZ * DV;

        memset(VKQ32, 0, Q_TILE_SZ * DV * sizeof(float));
        memset(mask32, 0, Q_TILE_SZ * KV_TILE_SZ * sizeof(float));

        // k indices
        const int ik3 = iq3 / rk3;
        const int ik2 = iq2 / rk2;

        // v indices
        const int iv3 = iq3 / rv3;
        const int iv2 = iq2 / rv2;

        {
            float * Q_f32 = (float *)Q_q;
            for (int tq = 0; tq < tile_rows; tq++) {
                const float * pq = (const float *) ((char *) q->data + ((iq1 + tq)*nbq1 + iq2*nbq2 + iq3*nbq3));
                memcpy(Q_f32 + tq * DK, pq, DK * sizeof(float));
            }
            for (int tq = tile_rows; tq < Q_TILE_SZ; tq++) {
                memset(Q_f32 + tq * DK, 0, DK * sizeof(float));
            }
        }

        memset(K_f32, 0, DK * KV_TILE_SZ * sizeof(float));
        memset(V32,   0, KV_TILE_SZ * DV * sizeof(float));

        for (int64_t ic = 0; ic < nek1; ic += KV_TILE_SZ) {
            const int kv_tile = (int)std::min((int64_t)KV_TILE_SZ, nek1 - ic);

            // skip the tile entirely if all the masks are -inf
            if (mask) {
                bool can_skip = true;
                for (int tq = 0; tq < tile_rows; tq++) {
                    const ggml_fp16_t * mp_row = (const ggml_fp16_t *)((const char *) mask->data + (iq1 + tq)*mask->nb[1] + (iq2%mask->ne[2])*mask->nb[2] + (iq3%mask->ne[3])*mask->nb[3]);
                    for (int tk = 0; tk < kv_tile; tk++) {
                        mask32[tq * KV_TILE_SZ + tk] = slope * GGML_CPU_FP16_TO_FP32(mp_row[ic + tk]);
                        if (mask32[tq * KV_TILE_SZ + tk] != -INFINITY) {
                            can_skip = false;
                        }
                    }
                    // Pad remaining mask entries with -inf
                    for (int tk = kv_tile; tk < KV_TILE_SZ; tk++) {
                        mask32[tq * KV_TILE_SZ + tk] = -INFINITY;
                    }
                }

                if (can_skip) {
                    continue;
                }
            }

            // Pack K tile transposed: K_f32[dk][kv] so KV_TILE is contiguous (SIMD dim)
            // Zero-pad the last tile so the GEMM always operates on KV_TILE_SZ columns
            for (int tk = 0; tk < kv_tile; tk++) {
                const char * k_data = (const char *)k->data + (ic + tk)*nbk1 + ik2*nbk2 + ik3*nbk3;
                if (kv_type == GGML_TYPE_F16) {
                    const ggml_fp16_t * k_f16 = (const ggml_fp16_t *)k_data;
                    for (int64_t dk = 0; dk < DK; dk++) {
                        K_f32[dk * KV_TILE_SZ + tk] = GGML_CPU_FP16_TO_FP32(k_f16[dk]);
                    }
                } else {
                    const float * k_f32_src = (const float *)k_data;
                    for (int64_t dk = 0; dk < DK; dk++) {
                        K_f32[dk * KV_TILE_SZ + tk] = k_f32_src[dk];
                    }
                }
            }
            memset(KQ, 0, Q_TILE_SZ * KV_TILE_SZ * sizeof(float));
            simd_gemm(KQ, (const float *)Q_q, K_f32, Q_TILE_SZ, DK, KV_TILE_SZ);
            ggml_vec_scale_f32(Q_TILE_SZ * KV_TILE_SZ, KQ, scale);

            // Set padded KQ entries to -inf so softmax gives them zero weight
            if (kv_tile < KV_TILE_SZ) {
                for (int tq = 0; tq < Q_TILE_SZ; tq++) {
                    for (int tk = kv_tile; tk < KV_TILE_SZ; tk++) {
                        KQ[tq * KV_TILE_SZ + tk] = -INFINITY;
                    }
                }
            }

            if (logit_softcap != 0.0f) {
                ggml_vec_tanh_f32(Q_TILE_SZ * KV_TILE_SZ, KQ, KQ);
                ggml_vec_scale_f32(Q_TILE_SZ * KV_TILE_SZ, KQ, logit_softcap);
            }

            if (mask) {
                ggml_vec_add_f32(tile_rows * KV_TILE_SZ, KQ, KQ, mask32);
            }

            bool skip[Q_TILE_SZ] = {};

            for (int tq = 0; tq < Q_TILE_SZ; tq++) {
                float * kq_row = KQ + tq * KV_TILE_SZ;

                float tile_max;
                ggml_vec_max_f32(KV_TILE_SZ, &tile_max, kq_row);

                if (tile_max == -INFINITY) {
                    skip[tq] = true;
                    continue;
                }

                const float Mold = M[tq];
                const float Mnew = fmaxf(Mold, tile_max);

                if (Mnew > Mold) {
                    const float ms = expf(Mold - Mnew);
                    ggml_vec_scale_f32(DV, VKQ32 + tq * DV, ms);
                    S[tq] *= ms;
                }
                M[tq] = Mnew;


                S[tq] += ggml_vec_soft_max_f32(KV_TILE_SZ, kq_row, kq_row, Mnew);
            }

            // V accumulation: VKQ32 += softmax(KQ) * V
            // Pack V tile to contiguous F32, zero-padded
            for (int tk = 0; tk < kv_tile; tk++) {
                const char * v_data = (const char *)v->data + (ic + tk)*nbv1 + iv2*nbv2 + iv3*nbv3;
                if (kv_type == GGML_TYPE_F16) {
                    ggml_fp16_to_fp32_row((const ggml_fp16_t *)v_data, V32 + tk * DV, DV);
                } else {
                    memcpy(V32 + tk * DV, v_data, DV * sizeof(float));
                }
            }
            for (int tq = 0; tq < Q_TILE_SZ; tq++) {
                if (skip[tq]) {
                    memset(KQ + tq * KV_TILE_SZ, 0, KV_TILE_SZ * sizeof(float));
                }
            }
            simd_gemm(VKQ32, KQ, V32, Q_TILE_SZ, KV_TILE_SZ, DV);
        }

        // sinks (apply only to valid rows in the tile)
        if (sinks) {
            const float s = ((float *)((char *) sinks->data))[h];

            for (int tq = 0; tq < tile_rows; tq++) {
                float ms = 1.0f;
                float vs = 1.0f;

                if (s > M[tq]) {
                    ms = expf(M[tq] - s);
                    ggml_vec_scale_f32(DV, VKQ32 + tq * DV, ms);
                } else {
                    vs = expf(s - M[tq]);
                }

                S[tq] = S[tq] * ms + vs;
            }
        }

        for (int tq = 0; tq < tile_rows; tq++) {
            // V /= S
            const float S_inv = S[tq] == 0.0f ? 0.0f : 1.0f / S[tq];
            ggml_vec_scale_f32(DV, VKQ32 + tq * DV, S_inv);

            // dst indices
            const int i1 = iq1 + tq;
            const int i2 = iq2;
            const int i3 = iq3;

            // permute(0, 2, 1, 3)
            memcpy((char *) dst->data + (i3*ne2*ne1 + i2 + i1*ne1)*nb1, VKQ32 + tq * DV, nb1);
        }

        ir += tile_rows;
    }
}

// Reduction function: combines partial results across KV chunks
// Partials layout in wdata: [n_q_heads][n_chunks][2 + DV]
static void ggml_flash_attn_ext_reduce_partials(
        const ggml_compute_params * params,
        ggml_tensor * dst,
        const int64_t n_chunks,
        const int64_t chunk_size) {

    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * v = dst->src[2];

    const int64_t DK        = k->ne[0];
    const int64_t DV        = v->ne[0];
    const int64_t nek1      = k->ne[1];
    const int64_t n_q_heads = q->ne[2];

    const int ith = params->ith;
    const int nth = params->nth;

    const int64_t wdata_per_thread = DK + 2*DV + CACHE_LINE_SIZE_F32;
    float *       thread_wdata     = (float *) params->wdata + ith * wdata_per_thread;

    const int64_t partials_offset  = nth * (DK + 2*DV + CACHE_LINE_SIZE_F32);
    const int64_t partial_size     = 2 + DV;
    const float * partials_base    = (const float *) params->wdata + partials_offset;

    // Output layout
    const int64_t ne1 = dst->ne[1];
    const int64_t ne2 = dst->ne[2];
    const size_t  nb1 = dst->nb[1];

    // Each thread reduces a subset of query heads
    for (int64_t q_head = ith; q_head < n_q_heads; q_head += nth) {
        float   M_final   = -INFINITY;
        float   S_final   = 0.0f;
        float * VKQ_final = thread_wdata;
        memset(VKQ_final, 0, DV * sizeof(float));

        // Combine partials from all chunks
        for (int64_t chunk_idx = 0; chunk_idx < n_chunks; ++chunk_idx) {
            const int64_t ic_start = chunk_idx * chunk_size;
            if (ic_start >= nek1) continue;

            const float * partial   = partials_base + (q_head * n_chunks + chunk_idx) * partial_size;
            const float   M_chunk   = partial[0];
            const float   S_chunk   = partial[1];
            const float * VKQ_chunk = partial + 2;

            if (S_chunk == 0.0f) continue;

            const float M_new     = fmaxf(M_final, M_chunk);
            const float scale_old = expf(M_final - M_new);
            const float scale_new = expf(M_chunk - M_new);

            for (int64_t d = 0; d < DV; ++d) {
                VKQ_final[d] = VKQ_final[d] * scale_old + VKQ_chunk[d] * scale_new;
            }
            S_final = S_final * scale_old + S_chunk * scale_new;
            M_final = M_new;
        }

        // Normalize and write to output
        if (S_final != 0.0f) {
            const float S_inv = 1.0f / S_final;
            ggml_vec_scale_f32(DV, VKQ_final, S_inv);
        }
        // iq1=0, iq3=0 for decode
        memcpy((char *) dst->data + (0*ne2*ne1 + q_head + 0*ne1)*nb1, VKQ_final, nb1);
    }
}

static void ggml_compute_forward_flash_attn_ext_f16(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * q     = dst->src[0];
    const ggml_tensor * k     = dst->src[1];
    const ggml_tensor * v     = dst->src[2];

    GGML_TENSOR_LOCALS(int64_t, neq, q,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbq, q,   nb)
    GGML_TENSOR_LOCALS(int64_t, nek, k,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbk, k,   nb)
    GGML_TENSOR_LOCALS(int64_t, nev, v,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbv, v,   nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst, ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst, nb)

    const int64_t DK = nek0;
    const int64_t DV = nev0;
    const int64_t N  = neq1;


    GGML_ASSERT(ne0 == DV);
    GGML_ASSERT(ne2 == N);

    // input tensor rows must be contiguous
    GGML_ASSERT(nbq0 == ggml_type_size(q->type));
    GGML_ASSERT(nbk0 == ggml_type_size(k->type));
    GGML_ASSERT(nbv0 == ggml_type_size(v->type));

    GGML_ASSERT(neq0 == DK);
    GGML_ASSERT(nek0 == DK);
    GGML_ASSERT(nev0 == DV);

    GGML_ASSERT(neq1 == N);

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    const int ith = params->ith;
    const int nth = params->nth;

    // When use_ref is set, force the vec-only reference implementation (no tiling, no KV-chunking)
    const bool use_ref = params->use_ref;

    const bool kv_is_f32_or_f16 = (k->type == GGML_TYPE_F32 || k->type == GGML_TYPE_F16);
    const bool use_split_kv_path = !use_ref && (neq1 == 1 && neq3 == 1) && kv_is_f32_or_f16 && (k->type == v->type) && q->type == GGML_TYPE_F32 && nek1 >= 512;

    if (use_split_kv_path) {
        const int64_t chunk_size = (nek1 + nth - 1) / nth;

        // Partials buffer layout: [q_head][kv_chunk][M, S, VKQ]
        const int64_t partial_size  = 2 + DV;
        float *       partials_base = (float *) params->wdata + nth * (DK + 2*DV + CACHE_LINE_SIZE_F32);

        const int64_t ic_start = ith * chunk_size;
        const int64_t ic_end   = std::min(ic_start + chunk_size, nek1);

        const int64_t partial_stride = nth * partial_size;
        float *       chunk_partials = partials_base + ith * partial_size;

        if (ic_start < nek1) {
            for (int64_t q_head = 0; q_head < neq2; q_head++) {
                ggml_compute_forward_flash_attn_ext_f16_one_chunk(
                    params, dst, q_head, q_head + 1, ic_start, ic_end,
                    chunk_partials, partial_stride);
            }
        } else {
            for (int64_t q_head = 0; q_head < neq2; q_head++) {
                float * q_partials = chunk_partials + q_head * partial_stride;
                q_partials[0] = -INFINITY;  // M
                q_partials[1] = 0.0f;       // S
            }
        }

        ggml_barrier(params->threadpool);
        ggml_flash_attn_ext_reduce_partials(params, dst, nth, chunk_size);
    } else {

        // total rows in q
        const int64_t nr = neq1*neq2*neq3;

        // disable for NUMA
        const bool disable_chunking = ggml_is_numa();

        // 4x chunks per thread
        int nth_scaled = nth * 4;
        int64_t chunk_size = (nr + nth_scaled - 1) / nth_scaled;
        int64_t nchunk     = (nr + chunk_size - 1) / chunk_size;

        if (nth == 1 || nchunk < nth || disable_chunking) {
            nchunk = nth;
        }

        if (ith == 0) {
            ggml_threadpool_chunk_set(params->threadpool, nth);
        }

        ggml_barrier(params->threadpool);

        const int64_t dr = (nr + nchunk - 1) / nchunk;

        static constexpr int64_t Q_TILE_SZ  = ggml_fa_tile_config::Q;
        bool use_tiled = !use_ref &&
                               (q->type == GGML_TYPE_F32 &&
                                kv_is_f32_or_f16 &&
                                k->type == v->type &&
                                neq1 >= Q_TILE_SZ);
#ifdef GGML_SIMD
#if defined(__ARM_FEATURE_SVE)
        const int64_t f32_epr = svcntw();
#else
        const int64_t f32_epr = GGML_F32_EPR;
#endif
        use_tiled &= (DV % f32_epr == 0);
#endif
        int current_chunk = ith;

        while (current_chunk < nchunk) {
            const int64_t ir0 = dr * current_chunk;
            const int64_t ir1 = MIN(ir0 + dr, nr);

            if (use_tiled) {
                ggml_compute_forward_flash_attn_ext_tiled(params, dst, ir0, ir1);
            } else {
                ggml_compute_forward_flash_attn_ext_f16_one_chunk(params, dst, ir0, ir1, 0, nek1, nullptr, 0);
            }

            current_chunk = ggml_threadpool_chunk_add(params->threadpool, 1);
        }
    }
}

void ggml_compute_forward_flash_attn_ext(
        const ggml_compute_params * params,
        ggml_tensor * dst) {
    switch (dst->op_params[3]) {
        case GGML_PREC_DEFAULT:
        case GGML_PREC_F32:
            {
                // uses F32 accumulators
                ggml_compute_forward_flash_attn_ext_f16(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_flash_attn_back

static void ggml_compute_forward_flash_attn_back_f32(
        const ggml_compute_params * params,
        const bool masked,
              ggml_tensor * dst) {

    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * v = dst->src[2];
    const ggml_tensor * d = dst->src[3];

    GGML_TENSOR_LOCALS(int64_t, neq, q,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbq, q,   nb)
    GGML_TENSOR_LOCALS(int64_t, nek, k,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbk, k,   nb)
    GGML_TENSOR_LOCALS(int64_t, nev, v,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbv, v,   nb)
    GGML_TENSOR_LOCALS(int64_t, ned, d,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbd, d,   nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst, ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst, nb)

    const int ith = params->ith;
    const int nth = params->nth;

    const int64_t D = neq0;
    const int64_t N = neq1;
    const int64_t P = nek1 - N;
    const int64_t M = P + N;

    const int Mup  = ggml_up(M, GGML_SOFT_MAX_UNROLL);
    const int mxDM = MAX(D, Mup);

    // GGML_ASSERT(ne0 == D);
    // GGML_ASSERT(ne1 == N);
    GGML_ASSERT(P >= 0);

    GGML_ASSERT(nbq0 == sizeof(float));
    GGML_ASSERT(nbk0 == sizeof(float));
    GGML_ASSERT(nbv0 == sizeof(float));

    GGML_ASSERT(neq0 == D);
    GGML_ASSERT(nek0 == D);
    GGML_ASSERT(nev1 == D);
    GGML_ASSERT(ned0 == D);

    GGML_ASSERT(neq1 == N);
    GGML_ASSERT(nek1 == N + P);
    GGML_ASSERT(nev1 == D);
    GGML_ASSERT(ned1 == N);

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    if (ith == 0) {
        memset(dst->data, 0, nb0*ne0*ne1*ne2*ne3);
    }
    ggml_barrier(params->threadpool);

    const int64_t elem_q = ggml_nelements(q);
    const int64_t elem_k = ggml_nelements(k);

    ggml_type result_type = dst->type;
    GGML_ASSERT(ggml_blck_size(result_type) == 1);
    const size_t tsize = ggml_type_size(result_type);

    const size_t offs_q = 0;
    const size_t offs_k = offs_q + GGML_PAD(elem_q * tsize, GGML_MEM_ALIGN);
    const size_t offs_v = offs_k + GGML_PAD(elem_k * tsize, GGML_MEM_ALIGN);

    void * grad_q = (char *) dst->data;
    void * grad_k = (char *) dst->data + offs_k;
    void * grad_v = (char *) dst->data + offs_v;

    const size_t nbgq1 = nb0*neq0;
    const size_t nbgq2 = nb0*neq0*neq1;
    const size_t nbgq3 = nb0*neq0*neq1*neq2;

    const size_t nbgk1 = nb0*nek0;
    const size_t nbgk2 = nb0*nek0*nek1;
    const size_t nbgk3 = nb0*nek0*nek1*neq2;

    const size_t nbgv1 = nb0*nev0;
    const size_t nbgv2 = nb0*nev0*nev1;
    const size_t nbgv3 = nb0*nev0*nev1*neq2;

    // parallelize by k rows using ggml_vec_dot_f32

    // total rows in k
    const int nr = nek2*nek3;

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    const float scale = 1.0f/sqrtf(D);

    //printf("P=%d N=%d D=%d ir0=%d ir1=%d scale = %f\n", P, N, D, ir0, ir1, scale);

    // how often k2 (and v2) is repeated in q2
    int nrep = neq2/nek2;

    for (int ir = ir0; ir < ir1; ++ir) {
        // q indices
        const int ik3 = ir/(nek2);
        const int ik2 = ir - ik3*nek2;

        const int iq3 = ik3;
        const int id3 = ik3;
        const int iv3 = ik3;
        const int iv2 = ik2;

        for (int irep = 0; irep < nrep; ++irep) {
            const int iq2 = ik2 + irep*nek2;
            const int id2 = iq2;

            // (ik2 + irep*nek2) % nek2 == ik2
            for (int iq1 = 0; iq1 < neq1; ++iq1) {
                const int id1 = iq1;

                // not sure about CACHE_LINE_SIZE_F32..
                // - maybe it must not be multiplied by 2 and excluded from .. in SM 1*(..) offset?
                float * S  = (float *) params->wdata + ith*2*(mxDM + CACHE_LINE_SIZE_F32) + 0*(mxDM+CACHE_LINE_SIZE_F32);
                float * SM = (float *) params->wdata + ith*2*(mxDM + CACHE_LINE_SIZE_F32) + 1*(mxDM+CACHE_LINE_SIZE_F32);

                for (int i = M; i < Mup; ++i) {
                    S[i] = -INFINITY;
                }

                const int64_t masked_begin = masked ? (P + iq1 + 1) : M;
                for (int64_t ic = 0; ic < masked_begin; ++ic) {
                    // k indices
                    const int ik1 = ic;

                    // S indices
                    const int i1 = ik1;

                    ggml_vec_dot_f32(neq0,
                            S + i1, 0,
                            (float *) ((char *) k->data + (ik1*nbk1 + ik2*nbk2 + ik3*nbk3)), 0,
                            (float *) ((char *) q->data + (iq1*nbq1 + iq2*nbq2 + iq3*nbq3)), 0, 1);
                }

                // scale
                ggml_vec_scale_f32(masked_begin, S, scale);

                for (int64_t i = masked_begin; i < M; i++) {
                    S[i] = -INFINITY;
                }

                // softmax
                // exclude known -INF S[..] values from max and loop
                // dont forget to set their SM values to zero
                {
                    float max = -INFINITY;
                    ggml_vec_max_f32(masked_begin, &max, S);

                    ggml_float sum = 0.0;
                    {
#ifdef GGML_SOFT_MAX_ACCELERATE
                        max = -max;
                        vDSP_vsadd(SM, 1, &max, SM, 1, Mup);
                        vvexpf(SM, SM, &Mup);
                        ggml_vec_sum_f32(Mup, &sum, SM);
#else
                        sum = ggml_vec_soft_max_f32(Mup, SM, S, max);
#endif
                    }

                    assert(sum > 0.0);

                    sum = 1.0/sum;
                    ggml_vec_scale_f32(masked_begin, SM, sum);

                }

                // step-by-step explanation
                {
                    // forward-process                    shape      grads from backward process
                    // parallel_for ik2,ik3:
                    //  for irep:
                    //   iq2 = ik2 + irep*nek2
                    //   k[:D,:M,:,:]                     [D,M,:,:]  grad[k][:D,:M,ik2,ik3]  += grad[kcur]
                    //   q[:D,:N,:,:]                     [D,N,:,:]  grad[q][:D,iq1,iq2,iq3] += grad[qcur]
                    //   v[:M,:D,:,:]                     [M,D,:,:]  grad[v][:M,:D,iv2,iv3]  += grad[vcur]
                    //   for iq1:
                    //    kcur   = k[:D,:M,ik2,ik3]       [D,M,1,1]  grad[kcur] = grad[S1].T @ qcur
                    //    qcur   = q[:D,iq1,iq2,iq3]      [D,1,1,1]  grad[qcur] = grad[S1]   @ kcur
                    //    vcur   = v[:M,:D,iv2,iv3]       [M,D,1,1]  grad[vcur] = grad[S5].T @ S4
                    //    S0     = -Inf                   [D,1,1,1]
                    //   ~S1[i]  = dot(kcur[:D,i], qcur)
                    //    S1     = qcur @ kcur.T          [M,1,1,1]  grad[S1]   = grad[S2] * scale
                    //    S2     = S1 * scale             [M,1,1,1]  grad[S2]   = diag_mask_zero(grad[S3], P)
                    //    S3     = diag_mask_inf(S2, P)   [M,1,1,1]  grad[S3]   = S4 * (grad[S4] - dot(S4, grad[S4]))
                    //    S4     = softmax(S3)            [M,1,1,1]  grad[S4]   = grad[S5] @ vcur
                    //   ~S5[i]  = dot(vcur[:,i], S4)
                    //    S5     = S4 @ vcur.T            [D,1,1,1]  grad[S5]   = d[:D,id1,id2,id3]
                    //   ~dst[i,iq1,iq2,iq3]  = S5[i]              ^
                    //    dst[:D,iq1,iq2,iq3] = S5                 | grad[dst[:D,iq1,iq2,iq3]] = d[:D,id1,id2,id3]
                    // dst                               backward-/ grad[dst]                 = d
                    //
                    // output gradients with their dependencies:
                    //
                    // grad[kcur] = grad[S1].T @ qcur
                    // grad[S1]   = diag_mask_zero(grad[S3], P) * scale
                    // grad[S3]   = S4 * (grad[S4] - dot(S4, grad[S4]))
                    // grad[S4]   = grad[S5] @ vcur
                    // grad[S4]   = d[:D,id1,id2,id3] @ vcur
                    // grad[qcur] = grad[S1]   @ kcur
                    // grad[vcur] = grad[S5].T @ S4
                    // grad[vcur] = d[:D,id1,id2,id3].T @ S4
                    //
                    // in post-order:
                    //
                    // S1         = qcur @ kcur.T
                    // S2         = S1 * scale
                    // S3         = diag_mask_inf(S2, P)
                    // S4         = softmax(S3)
                    // grad[S4]   = d[:D,id1,id2,id3] @ vcur
                    // grad[S3]   = S4 * (grad[S4] - dot(S4, grad[S4]))
                    // grad[S1]   = diag_mask_zero(grad[S3], P) * scale
                    // grad[qcur] = grad[S1]   @ kcur
                    // grad[kcur] = grad[S1].T @ qcur
                    // grad[vcur] = d[:D,id1,id2,id3].T @ S4
                    //
                    // using less variables (SM=S4):
                    //
                    // S             = diag_mask_inf(qcur @ kcur.T * scale, P)
                    // SM            = softmax(S)
                    // S             = d[:D,iq1,iq2,iq3] @ vcur
                    // dot_SM_gradSM = dot(SM, S)
                    // S             = SM * (S - dot(SM, S))
                    // S             = diag_mask_zero(S, P) * scale
                    //
                    // grad[q][:D,iq1,iq2,iq3] += S   @ kcur
                    // grad[k][:D,:M,ik2,ik3]  += S.T @ qcur
                    // grad[v][:M,:D,iv2,iv3]  += d[:D,id1,id2,id3].T @ SM
                }

                // S = gradSM = d[:D,id1,id2,id3] @ vcur[:,:,iv2,iv3]
                // S = d[:D,id1,id2,id3] @ vcur[:,:,iv2,iv3]
                // for ic:
                //   S[:M] += vcur[:M,ic,iv2,iv3] * d[ic,id1,id2,id3]
                // exclude known future zero S[..] values from operation
                ggml_vec_set_f32(masked_begin, S, 0);
                for (int64_t ic = 0; ic < D; ++ic) {
                    ggml_vec_mad_f32(masked_begin,
                            S,
                             (float *) ((char *) v->data + (          ic*nbv1  + iv2*nbv2 + iv3*nbv3)),
                            *(float *) ((char *) d->data + (ic*nbd0 + id1*nbd1 + id2*nbd2 + id3*nbd3)));
                }

                // S = SM * (S - dot(SM, S))
                float dot_SM_gradSM = 0;
                ggml_vec_dot_f32 (masked_begin, &dot_SM_gradSM, 0, SM, 0, S, 0, 1);
                ggml_vec_acc1_f32(M, S, -dot_SM_gradSM);
                ggml_vec_mul_f32 (masked_begin, S, S, SM);

                // S = diag_mask_zero(S, P) * scale
                // already done by above ggml_vec_set_f32

                // exclude known zero S[..] values from operation
                ggml_vec_scale_f32(masked_begin, S, scale);

                // S    shape [M,1]
                // SM   shape [M,1]
                // kcur shape [D,M]
                // qcur shape [D,1]
                // vcur shape [M,D]

                // grad[q][:D,iq1,iq2,iq3] += S @ kcur
                // grad[q][:D,iq1,iq2,iq3] += shape[M,1] @ shape[D,M]
                // for ic:
                //  grad[q][:D,iq1,iq2,iq3] += S[ic] * kcur[:D,ic,ik2,ik3]
                // exclude known zero S[..] values from loop
                for (int64_t ic = 0; ic < masked_begin; ++ic) {
                    ggml_vec_mad_f32(D,
                            (float *) ((char *) grad_q  + (iq1*nbgq1 + iq2*nbgq2  + iq3*nbgq3)),
                            (float *) ((char *) k->data + (ic*nbk1   + ik2*nbk2   + ik3*nbk3)),
                            S[ic]);
                }

                // grad[k][:D,:M,iq2,iq3] += S.T @ qcur
                // for ic:
                //  grad[k][:D,ic,iq2,iq3] += S.T[0,ic] * qcur[:D,0]
                //  grad[k][:D,ic,iq2,iq3] += S[ic]     * qcur[:D,0]
                // exclude known zero S[..] values from loop
                for (int64_t ic = 0; ic < masked_begin; ++ic) {
                    ggml_vec_mad_f32(D,
                            (float *) ((char *) grad_k  + (ic*nbgk1  + ik2*nbgk2  + ik3*nbgk3)),
                            (float *) ((char *) q->data + (iq1*nbq1  + iq2*nbq2   + iq3*nbq3)),
                            S[ic]);
                }

                // grad[v][:M,:D,iv2,iv3] += d[:D,id1,id2,id3].T       @ SM
                // for ic:
                //  grad[v][:M,ic,iv2,iv3] += d[:D,id1,id2,id3].T[0,ic] * SM[:M]
                //  grad[v][:M,ic,iv2,iv3] += d[ic,id1,id2,id3]         * SM[:M]
                // exclude known zero SM[..] values from mad
                for (int64_t ic = 0; ic < D; ++ic) {
                    ggml_vec_mad_f32(masked_begin,
                            (float *) ((char *) grad_v   + (          ic*nbgv1 + iv2*nbgv2 + iv3*nbgv3)),
                            SM,
                            *(float *) ((char *) d->data + (ic*nbd0 + id1*nbd1 + id2*nbd2  + id3*nbd3)));
                }
            }
        }
    }
}

void ggml_compute_forward_flash_attn_back(
        const ggml_compute_params * params,
        const bool masked,
        ggml_tensor * dst) {

    const ggml_tensor * q = dst->src[0];

    switch (q->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_flash_attn_back_f32(params, masked, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_ssm_conv

static void ggml_compute_forward_ssm_conv_f32(
        const ggml_compute_params * params,
        ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0]; // conv_x
    const ggml_tensor * src1 = dst->src[1]; // conv1d.weight

    const int ith = params->ith;
    const int nth = params->nth;

    const int nc  = src1->ne[0]; // d_conv
    const int ncs = src0->ne[0]; // d_conv - 1 + n_t
    const int nr  = src0->ne[1]; // d_inner
    const int n_t =  dst->ne[1]; // tokens per sequence
    const int n_s =  dst->ne[2]; // number of sequences in the batch

    GGML_ASSERT( dst->ne[0] == nr);
    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(src1->nb[0] == sizeof(float));
    GGML_ASSERT(src0->nb[1] == src0->ne[0]*sizeof(float));

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);
    const int ir  = ir1 - ir0;

    for (int i3 = 0; i3 < n_s; ++i3) {
        for (int i2 = 0; i2 < n_t; ++i2) {
            // {d_conv - 1 + n_t, d_inner, n_seqs}
            // sliding window
            const float * s = (const float *) ((const char *) src0->data + ir0*(src0->nb[1]) + i2*(src0->nb[0]) + i3*(src0->nb[2])); // {d_conv, d_inner, n_s}
            const float * c = (const float *) ((const char *) src1->data + ir0*(src1->nb[1])); // {d_conv, d_inner}
            float * x = (float *) ((char *) dst->data + ir0*(dst->nb[0]) + i2*(dst->nb[1]) + i3*(dst->nb[2])); // {d_inner, n_t, n_s}

            // TODO: transpose the output for smaller strides for big batches?
            // d_inner
            for (int i1 = 0; i1 < ir; ++i1) {
                // rowwise dot product
                // NOTE: not using ggml_vec_dot_f32, because its sum is in double precision
                float sumf = 0.0f;

                // d_conv
                for (int i0 = 0; i0 < nc; ++i0) {
                    sumf += s[i0 + i1*ncs] * c[i0 + i1*nc];
                }
                x[i1] = sumf;
            }
        }
    }
}

void ggml_compute_forward_ssm_conv(
        const ggml_compute_params * params,
        ggml_tensor * dst) {
    switch (dst->src[0]->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_ssm_conv_f32(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_ssm_scan

static void ggml_compute_forward_ssm_scan_f32(
        const ggml_compute_params * params,
        ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0]; // s  {d_state, dim, n_head, n_seqs+}
    const ggml_tensor * src1 = dst->src[1]; // x  {dim, n_head, n_seq_tokens, n_seqs}
    const ggml_tensor * src2 = dst->src[2]; // dt {n_head, n_seq_tokens, n_seqs}
    const ggml_tensor * src3 = dst->src[3]; // A  {d_state, n_head} or {1, n_head}
    const ggml_tensor * src4 = dst->src[4]; // B  {d_state, n_group, n_seq_tokens, n_seqs}
    const ggml_tensor * src5 = dst->src[5]; // C  {d_state, n_group, n_seq_tokens, n_seqs}
    const ggml_tensor * src6 = dst->src[6]; // ids {n_seqs}

    const int ith = params->ith;
    const int nth = params->nth;

    const int64_t nc = src0->ne[0]; // d_state
    const int64_t nr = src0->ne[1]; // dim
    const int64_t nh = src1->ne[1]; // n_head
    const int64_t ng = src4->ne[1];
    const int64_t nt = src1->ne[2]; // number of tokens per sequence
    const int64_t ns = src1->ne[3]; // number of sequences in the batch

    // can't use ggml_nbytes because src1 is not necessarily contiguous
    const int64_t s_off = ggml_nelements(src1) * ggml_element_size(src1);

    GGML_ASSERT(ggml_nelements(src1) + nc*nr*nh*ns == ggml_nelements(dst));
    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(src1->nb[0] == sizeof(float));
    GGML_ASSERT(src2->nb[0] == sizeof(float));
    GGML_ASSERT(src3->nb[0] == sizeof(float));
    GGML_ASSERT(src4->nb[0] == sizeof(float));
    GGML_ASSERT(src5->nb[0] == sizeof(float));
    GGML_ASSERT(src6->nb[0] == sizeof(int32_t));
    GGML_ASSERT(nh % ng == 0);

    // heads per thread
    const int dh = (nh + nth - 1)/nth;

    // head range for this thread
    const int ih0 = dh*ith;
    const int ih1 = MIN(ih0 + dh, nh);

    const int32_t * ids = (const int32_t *) src6->data;

    for (int i3 = 0; i3 < ns; ++i3) {
        const float * s0 = (const float *) ((const char *) src0->data + ids[i3]*(src0->nb[3])); // {d_state, dim, nh, ns}
              float * s  = (      float *) ((      char *) dst->data  + i3*(src0->nb[3]) + s_off); // {d_state, dim, nh, ns}

        for (int i2 = 0; i2 < nt; ++i2) {
            const float * x  = (const float *) ((const char *) src1->data + i2*(src1->nb[2]) + i3*(src1->nb[3])); // {dim, nh, nt, ns}
            const float * dt = (const float *) ((const char *) src2->data + i2*(src2->nb[1]) + i3*(src2->nb[2])); // {nh, nt, ns}
            const float * A  = (const float *) ((const char *) src3->data); // {d_state, nh} or {1, nh}
            const float * B  = (const float *) ((const char *) src4->data + i2*(src4->nb[2]) + i3*(src4->nb[3])); // {d_state, ng, nt, ns}
            const float * C  = (const float *) ((const char *) src5->data + i2*(src5->nb[2]) + i3*(src5->nb[3])); // {d_state, ng, nt, ns}
                  float * y  = (      float *) ((      char *) dst->data + i2*(nh*nr*sizeof(float)) + i3*(nt*nh*nr*sizeof(float))); // {dim, nh, nt, ns}

            if (src3->ne[0] == 1) {
                // Mamba-2 has a scalar decay factor per head; dA can be outside the state-wise loop

                // n_head
                for (int h = ih0; h < ih1; ++h) {
                    // ref: https://github.com/state-spaces/mamba/blob/62db608da60f6fc790b8ed9f4b3225e95ca15fde/mamba_ssm/ops/triton/softplus.py#L16
                    const float dt_soft_plus = ggml_compute_softplus_f32(dt[h]);
                    const float dA = expf(dt_soft_plus * A[h]);
                    const int g = h / (nh / ng); // repeat_interleave

                    // dim
                    for (int i1 = 0; i1 < nr; ++i1) {
                        const int ii = i1 + h*nr;
                        const float x_dt = x[ii] * dt_soft_plus;
                        float sumf = 0.0f;
#if defined(GGML_SIMD)
    #if defined(__ARM_FEATURE_SVE)
                        const int ggml_f32_epr = svcntw();
                        const int ggml_f32_step = 1 * ggml_f32_epr;

                        const int np = (nc & ~(ggml_f32_step - 1));

                        GGML_F32_VEC sum = GGML_F32_VEC_ZERO;

                        GGML_F32_VEC adA = GGML_F32_VEC_SET1(dA);
                        GGML_F32_VEC axdt = GGML_F32_VEC_SET1(x_dt);

                        for (int i = 0; i < np; i += ggml_f32_step) {
                            // TODO: maybe unroll more?
                            for (int j = 0; j < 1; j++) {
                                GGML_F32_VEC t0 = GGML_F32_VEC_LOAD(s0 + i + j*ggml_f32_epr + ii*nc);
                                GGML_F32_VEC t1 = GGML_F32_VEC_LOAD(B + i + j*ggml_f32_epr + g*nc);
                                GGML_F32_VEC t2 = GGML_F32_VEC_LOAD(C + i + j*ggml_f32_epr + g*nc);

                                t0 = GGML_F32_VEC_MUL(t0, adA);
                                t1 = GGML_F32_VEC_MUL(t1, axdt);

                                t0 = GGML_F32_VEC_ADD(t0, t1);

                                sum = GGML_F32_VEC_FMA(sum, t0, t2);

                                GGML_F32_VEC_STORE(s + i + j*ggml_f32_epr + ii*nc, t0);
                            }
                        }

                        sumf = GGML_F32xt_REDUCE_ONE(sum);
    #elif defined(__riscv_v_intrinsic)
                        // todo: RVV implementation
                        const int np = 0;
    #else
                        const int np = (nc & ~(GGML_F32_STEP - 1));

                        GGML_F32_VEC sum[GGML_F32_ARR] = { GGML_F32_VEC_ZERO };

                        GGML_F32_VEC adA = GGML_F32_VEC_SET1(dA);
                        GGML_F32_VEC axdt = GGML_F32_VEC_SET1(x_dt);

                        GGML_F32_VEC ax[GGML_F32_ARR];
                        GGML_F32_VEC ay[GGML_F32_ARR];
                        GGML_F32_VEC az[GGML_F32_ARR];

                        for (int i = 0; i < np; i += GGML_F32_STEP) {
                            for (int j = 0; j < GGML_F32_ARR; j++) {
                                ax[j] = GGML_F32_VEC_LOAD(s0 + i + j*GGML_F32_EPR + ii*nc);
                                ay[j] = GGML_F32_VEC_LOAD(B + i + j*GGML_F32_EPR + g*nc);
                                az[j] = GGML_F32_VEC_LOAD(C + i + j*GGML_F32_EPR + g*nc);

                                ax[j] = GGML_F32_VEC_MUL(ax[j], adA);
                                ay[j] = GGML_F32_VEC_MUL(ay[j], axdt);

                                ax[j] = GGML_F32_VEC_ADD(ax[j], ay[j]);

                                sum[j] = GGML_F32_VEC_FMA(sum[j], ax[j], az[j]);

                                GGML_F32_VEC_STORE(s + i + j*GGML_F32_EPR + ii*nc, ax[j]);
                            }
                        }

                        // reduce sum0..sum3 to sum0
                        GGML_F32_VEC_REDUCE(sumf, sum);
    #endif
#else
                        const int np = 0;
#endif
                        // d_state
                        for (int i0 = np; i0 < nc; ++i0) {
                            const int i = i0 + ii*nc;
                            const int ig = i0 + g*nc;
                            // state = prev_state * dA + dB * x
                            const float state = (s0[i] * dA) + (B[ig] * x_dt);
                            // y = rowwise_dotprod(state, C)
                            sumf += state * C[ig];
                            s[i] = state;
                        }
                        y[ii] = sumf;
                    }
                }
            } else {
                // Mamba-1 has an element-wise decay factor for the states

                // n_head
                for (int h = ih0; h < ih1; ++h) {
                    // ref: https://github.com/state-spaces/mamba/blob/62db608da60f6fc790b8ed9f4b3225e95ca15fde/mamba_ssm/ops/triton/softplus.py#L16
                    const float dt_soft_plus = ggml_compute_softplus_f32(dt[h]);
                    const int g = h / (nh / ng); // repeat_interleave

                    // dim
                    for (int i1 = 0; i1 < nr; ++i1) {
                        const int ii = i1 + h*nr;
                        const float x_dt = x[ii] * dt_soft_plus;
#if defined(__ARM_FEATURE_SVE)
                        svfloat32_t vx_dt = GGML_F32_VEC_SET1(x_dt);
                        svfloat32_t vdt_soft_plus = GGML_F32_VEC_SET1(dt_soft_plus);
                        svfloat32_t r1_vector = GGML_F32_VEC_ZERO;

                        // d_state
                        // TODO: what happens when (d_state % svcntw()) != 0?
                        for (int64_t k = 0; k < nc; k += svcntw()) {
                            svfloat32_t vA = GGML_F32_VEC_LOAD(&A[h*nc + k]);
                            svfloat32_t vB = GGML_F32_VEC_LOAD(&B[k + g*nc]);
                            svfloat32_t vC = GGML_F32_VEC_LOAD(&C[k + g*nc]);
                            svfloat32_t vs0 = GGML_F32_VEC_LOAD(&s0[ii*nc + k]);

                            svfloat32_t t1 = GGML_F32_VEC_MUL(vdt_soft_plus, vA);
                            t1 = exp_ps_sve(svptrue_b32(), t1);
                            svfloat32_t t2 = GGML_F32_VEC_MUL(vx_dt, vB);

                            vs0 = GGML_F32_VEC_FMA(t2, vs0, t1);
                            r1_vector = GGML_F32_VEC_ADD(GGML_F32_VEC_MUL(vs0, vC), r1_vector);

                            GGML_F32_VEC_STORE(&s[ii*nc + k], vs0);
                        }
                        y[ii] = GGML_F32xt_REDUCE_ONE(r1_vector);
#else
                        float sumf = 0.0f;
                        // NOTE: can't really use GGML_SIMD here because d_state is usually 16
                        //       and also because expf is used within the loop.
                        // d_state
                        for (int i0 = 0; i0 < nc; ++i0) {
                            const int i = i0 + ii*nc;
                            const int ig = i0 + g*nc;
                            // state = prev_state * dA + dB * x
                            const float state = (s0[i] * expf(dt_soft_plus * A[i0 + h*nc])) + (B[ig] * x_dt);
                            // y = rowwise_dotprod(state, C)
                            sumf += state * C[ig];
                            s[i] = state;
                        }
                        y[ii] = sumf;
#endif
                    }
                }
            }
            // use the output as the source when it's not the first token-wise iteration
            s0 = s;
        }
    }
}

void ggml_compute_forward_ssm_scan(
        const ggml_compute_params * params,
        ggml_tensor * dst) {
    switch (dst->src[0]->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_ssm_scan_f32(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_win_part

static void ggml_compute_forward_win_part_f32(
        const ggml_compute_params * params,
        ggml_tensor * dst) {
    GGML_UNUSED(params);

    const ggml_tensor * src0 = dst->src[0];

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)

    const int32_t nep0 = ((const int32_t *)(dst->op_params))[0];
    const int32_t nep1 = ((const int32_t *)(dst->op_params))[1];
    const int32_t w    = ((const int32_t *)(dst->op_params))[2];

    assert(ne00 == ne0);
    assert(ne3  == nep0*nep1);

    // TODO: optimize / multi-thread
    for (int py = 0; py < nep1; ++py) {
        for (int px = 0; px < nep0; ++px) {
            const int64_t i3 = py*nep0 + px;
            for (int64_t i2 = 0; i2 < ne2; ++i2) {
                for (int64_t i1 = 0; i1 < ne1; ++i1) {
                    for (int64_t i0 = 0; i0 < ne0; ++i0) {
                        const int64_t i02 = py*w + i2;
                        const int64_t i01 = px*w + i1;
                        const int64_t i00 = i0;

                        const int64_t i = i3*ne2*ne1*ne0 + i2*ne1*ne0    + i1*ne0   + i0;
                        const int64_t j =                  i02*ne01*ne00 + i01*ne00 + i00;

                        if (py*w + i2 >= ne02 || px*w + i1 >= ne01) {
                            ((float *) dst->data)[i] = 0.0f;
                        } else {
                            ((float *) dst->data)[i] = ((float *) src0->data)[j];
                        }
                    }
                }
            }
        }
    }
}

void ggml_compute_forward_win_part(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_win_part_f32(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_win_unpart

static void ggml_compute_forward_win_unpart_f32(
        const ggml_compute_params * params,
        ggml_tensor * dst) {
    GGML_UNUSED(params);

    const ggml_tensor * src0 = dst->src[0];

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)

    const int32_t w = ((const int32_t *)(dst->op_params))[0];

    // padding
    const int px = (w - ne1%w)%w;
    //const int py = (w - ne2%w)%w;

    const int npx = (px + ne1)/w;
    //const int npy = (py + ne2)/w;

    assert(ne0 == ne00);

    // TODO: optimize / multi-thread
    for (int64_t i2 = 0; i2 < ne2; ++i2) {
        for (int64_t i1 = 0; i1 < ne1; ++i1) {
            for (int64_t i0 = 0; i0 < ne0; ++i0) {
                const int ip2 = i2/w;
                const int ip1 = i1/w;

                const int64_t i02 = i2%w;
                const int64_t i01 = i1%w;
                const int64_t i00 = i0;

                const int64_t i = (ip2*npx + ip1)*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00 + i00;
                const int64_t j =                                  i2*ne1*ne0    + i1*ne0   + i0;

                ((float *) dst->data)[j] = ((float *) src0->data)[i];
            }
        }
    }
}

void ggml_compute_forward_win_unpart(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_win_unpart_f32(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

//ggml_compute_forward_unary

void ggml_compute_forward_unary(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_unary_op op = ggml_get_unary_op(dst);

    switch (op) {
        case GGML_UNARY_OP_ABS:
            {
                ggml_compute_forward_abs(params, dst);
            } break;
        case GGML_UNARY_OP_SGN:
            {
                ggml_compute_forward_sgn(params, dst);
            } break;
        case GGML_UNARY_OP_NEG:
            {
                ggml_compute_forward_neg(params, dst);
            } break;
        case GGML_UNARY_OP_STEP:
            {
                ggml_compute_forward_step(params, dst);
            } break;
        case GGML_UNARY_OP_TANH:
            {
                ggml_compute_forward_tanh(params, dst);
            } break;
        case GGML_UNARY_OP_ELU:
            {
                ggml_compute_forward_elu(params, dst);
            } break;
        case GGML_UNARY_OP_RELU:
            {
                ggml_compute_forward_relu(params, dst);
            } break;
        case GGML_UNARY_OP_SIGMOID:
            {
                ggml_compute_forward_sigmoid(params, dst);
            } break;
        case GGML_UNARY_OP_GELU:
            {
                ggml_compute_forward_gelu(params, dst);
            } break;
        case GGML_UNARY_OP_GELU_ERF:
            {
                ggml_compute_forward_gelu_erf(params, dst);
            } break;
        case GGML_UNARY_OP_GELU_QUICK:
            {
                ggml_compute_forward_gelu_quick(params, dst);
            } break;
        case GGML_UNARY_OP_SILU:
            {
                ggml_compute_forward_silu(params, dst);
            } break;
        case GGML_UNARY_OP_HARDSWISH:
            {
                ggml_compute_forward_hardswish(params, dst);
            } break;
        case GGML_UNARY_OP_HARDSIGMOID:
            {
                ggml_compute_forward_hardsigmoid(params, dst);
            } break;
        case GGML_UNARY_OP_EXP:
            {
                ggml_compute_forward_exp(params, dst);
            } break;
        case GGML_UNARY_OP_FLOOR:
            {
                ggml_compute_forward_floor(params, dst);
            } break;
        case GGML_UNARY_OP_CEIL:
            {
                ggml_compute_forward_ceil(params, dst);
            } break;
        case GGML_UNARY_OP_ROUND:
            {
                ggml_compute_forward_round(params, dst);
            } break;
        case GGML_UNARY_OP_TRUNC:
            {
                ggml_compute_forward_trunc(params, dst);
            } break;
        case GGML_UNARY_OP_XIELU:
            {
                ggml_compute_forward_xielu(params, dst);
            } break;
        case GGML_UNARY_OP_EXPM1:
            {
                ggml_compute_forward_expm1(params, dst);
            } break;
        case GGML_UNARY_OP_SOFTPLUS:
            {
                ggml_compute_forward_softplus(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

//ggml_compute_forward_glu

void ggml_compute_forward_glu(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_glu_op op = ggml_get_glu_op(dst);

    switch (op) {
        case GGML_GLU_OP_REGLU:
            {
                ggml_compute_forward_reglu(params, dst);
            } break;
        case GGML_GLU_OP_GEGLU:
            {
                ggml_compute_forward_geglu(params, dst);
            } break;
        case GGML_GLU_OP_SWIGLU:
            {
                ggml_compute_forward_swiglu(params, dst);
            } break;
        case GGML_GLU_OP_SWIGLU_OAI:
            {
                ggml_compute_forward_swiglu_oai(params, dst);
            } break;
        case GGML_GLU_OP_GEGLU_ERF:
            {
                ggml_compute_forward_geglu_erf(params, dst);
            } break;
        case GGML_GLU_OP_GEGLU_QUICK:
            {
                ggml_compute_forward_geglu_quick(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_get_rel_pos

static void ggml_compute_forward_get_rel_pos_f16(
        const ggml_compute_params * params,
        ggml_tensor * dst) {
    GGML_UNUSED(params);

    const ggml_tensor * src0 = dst->src[0];

    // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L292-L322

    GGML_TENSOR_UNARY_OP_LOCALS

    const int64_t w = ne1;

    ggml_fp16_t * src0_data = (ggml_fp16_t *) src0->data;
    ggml_fp16_t * dst_data  = (ggml_fp16_t *) dst->data;

    for (int64_t i2 = 0; i2 < ne2; ++i2) {
        for (int64_t i1 = 0; i1 < ne1; ++i1) {
            const int64_t pos = (w - i1 - 1) + i2;
            for (int64_t i0 = 0; i0 < ne0; ++i0) {
                dst_data[i2*ne1*ne0 + i1*ne0 + i0] = src0_data[pos*ne00 + i0];
            }
        }
    }
}

void ggml_compute_forward_get_rel_pos(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
            {
                ggml_compute_forward_get_rel_pos_f16(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_add_rel_pos

static void ggml_compute_forward_add_rel_pos_f32(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * src2 = dst->src[2];

    const bool inplace = (bool) ((int32_t *) dst->op_params)[0];
    if (!inplace) {
        if (params->ith == 0) {
            memcpy((char *) dst->data, (char *) src0->data, ggml_nbytes(dst));
        }
        ggml_barrier(params->threadpool);
    }
    // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L357-L359

    float * src1_data = (float *) src1->data;
    float * src2_data = (float *) src2->data;
    float * dst_data  = (float *) dst->data;

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];
    const int64_t ne13 = src1->ne[3];

    const int ith = params->ith;
    const int nth = params->nth;

    // total patches in dst
    const int np = ne13;

    // patches per thread
    const int dp = (np + nth - 1)/nth;

    // patch range for this thread
    const int ip0 = dp*ith;
    const int ip1 = MIN(ip0 + dp, np);

    for (int64_t i13 = ip0; i13 < ip1; ++i13) {
        for (int64_t i12 = 0; i12 < ne12; ++i12) {
            for (int64_t i11 = 0; i11 < ne11; ++i11) {
                const int64_t jp1 = i13*ne12*ne11*ne10 + i12*ne11*ne10 + i11*ne10;
                for (int64_t i10 = 0; i10 < ne10; ++i10) {
                    const int64_t jp0  = jp1 + i10;
                    const float src1_e = src1_data[jp0];
                    const float src2_e = src2_data[jp0];

                    const int64_t jdh = jp0 * ne10;
                    const int64_t jdw = jdh - (ne10 - 1) * i10;

                    for (int64_t j = 0; j < ne10; ++j) {
                        dst_data[jdh + j     ] += src2_e;
                        dst_data[jdw + j*ne10] += src1_e;
                    }
                }
            }
        }
    }
}

void ggml_compute_forward_add_rel_pos(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_add_rel_pos_f32(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_rwkv_wkv6

static void ggml_compute_forward_rwkv_wkv6_f32(
        const ggml_compute_params * params,
        ggml_tensor * dst) {
    const int64_t T = dst->src[1]->ne[2];
    const int64_t C = dst->ne[0];
    const int64_t HEADS = dst->src[1]->ne[1];
    const int64_t n_seqs = dst->src[5]->ne[1];
    const int64_t head_size = C / HEADS;

    float * dst_data = (float *) dst->data;
    float * state = ((float *) dst->data) + C * T;

    const int ith = params->ith;
    const int nth = params->nth;

    const int h_start =  (HEADS * (ith    )) / nth;
    const int h_end   = ((HEADS * (ith + 1)) / nth < HEADS) ?
                         (HEADS * (ith + 1)) / nth : HEADS;

    float * k =          (float *) dst->src[0]->data;
    float * v =          (float *) dst->src[1]->data;
    float * r =          (float *) dst->src[2]->data;
    float * time_faaaa = (float *) dst->src[3]->data;
    float * time_decay = (float *) dst->src[4]->data;

    size_t t_stride = HEADS * head_size; // Same to C

    size_t h_stride = C / HEADS;
    GGML_ASSERT(C % HEADS == 0); // C must be divisible by HEADS
    size_t h_stride_2d = head_size * head_size;

    if (ith == 0) {
        memset(dst_data, 0, T * C * sizeof(float));
    }
    ggml_barrier(params->threadpool);


    #if defined(__AVX__) && !defined(__AVX512F__)
        #define GGML_F32X GGML_F32x8
        #define GGML_F32X_SET1 GGML_F32x8_SET1
        #define GGML_F32X_LOAD GGML_F32x8_LOAD
        #define GGML_F32X_STORE GGML_F32x8_STORE
        #define GGML_F32X_MUL GGML_F32x8_MUL
        #define GGML_F32X_FMA GGML_F32x8_FMA
        #define WKV_VECTOR_SIZE 8
    #elif defined(__AVX512F__)
        #define GGML_F32X GGML_F32x16
        #define GGML_F32X_SET1 GGML_F32x16_SET1
        #define GGML_F32X_LOAD GGML_F32x16_LOAD
        #define GGML_F32X_STORE GGML_F32x16_STORE
        #define GGML_F32X_MUL GGML_F32x16_MUL
        #define GGML_F32X_FMA GGML_F32x16_FMA
        #define WKV_VECTOR_SIZE 16
    #elif defined(__ARM_FEATURE_SVE) && defined(__aarch64__)
        #define GGML_F32X GGML_F32xt
        #define GGML_F32X_SET1 GGML_F32xt_SET1
        #define GGML_F32X_LOAD GGML_F32xt_LOAD
        #define GGML_F32X_STORE GGML_F32xt_STORE
        #define GGML_F32X_MUL GGML_F32xt_MUL
        #define GGML_F32X_FMA GGML_F32xt_FMA
        #define WKV_VECTOR_SIZE 8
    #elif defined(__ARM_NEON) && defined(__aarch64__)
        #define GGML_F32X GGML_F32x4
        #define GGML_F32X_SET1 GGML_F32x4_SET1
        #define GGML_F32X_LOAD GGML_F32x4_LOAD
        #define GGML_F32X_STORE GGML_F32x4_STORE
        #define GGML_F32X_MUL GGML_F32x4_MUL
        #define GGML_F32X_FMA GGML_F32x4_FMA
        #define WKV_VECTOR_SIZE 4
    #endif

    #ifdef WKV_VECTOR_SIZE
        int wkv_vector_size;
        #if defined(__ARM_FEATURE_SVE)
            wkv_vector_size = svcntw();
        #else
            wkv_vector_size = WKV_VECTOR_SIZE;
        #endif
        const int64_t vec_count = head_size / wkv_vector_size;

        for (int64_t t = 0; t < T; t++) {
            size_t t_offset = t * t_stride;
            size_t state_offset = head_size * C * (t / (T / n_seqs));
            float * state_cur = state + state_offset;
            float * state_prev = t % (T / n_seqs) ? state_cur : (float*)dst->src[5]->data + state_offset;

            for (int64_t h = h_start; h < h_end; h++) {
                size_t h_offset = h * h_stride;
                size_t t_h_offset = t_offset + h_offset;
                size_t h_2d_offset = h * h_stride_2d;

                for (int64_t i = 0; i < head_size; i++) {
                    size_t t_h_i_offset = t_h_offset + i;
                    size_t h_i_offset = h_offset + i;
                    size_t h_2d_i_offset = h_2d_offset + i * h_stride;

                    float k_val = k[t_h_i_offset];
                    float r_val = r[t_h_i_offset];
                    float time_faaaa_val = time_faaaa[h_i_offset];
                    float time_decay_val = time_decay[t_h_i_offset];

                    // Broadcast scalar values to vectors
                    GGML_F32X k_vec = GGML_F32X_SET1(k_val);
                    GGML_F32X r_vec = GGML_F32X_SET1(r_val);
                    GGML_F32X time_faaaa_vec = GGML_F32X_SET1(time_faaaa_val);
                    GGML_F32X time_decay_vec = GGML_F32X_SET1(time_decay_val);

                    for (int64_t j = 0; j < vec_count; j++) {
                        size_t base_j = j * wkv_vector_size;
                        size_t t_h_j_offset = t_h_offset + base_j;
                        size_t h_2d_i_j_offset = h_2d_i_offset + base_j;

                        // Load x elements at once
                        GGML_F32X v_vec = GGML_F32X_LOAD(&v[t_h_j_offset]);
                        GGML_F32X prev_state_vec = GGML_F32X_LOAD(&state_prev[h_2d_i_j_offset]);
                        GGML_F32X dst_vec = GGML_F32X_LOAD(&dst_data[t_h_j_offset]);

                        // Compute kv = v * k
                        GGML_F32X kv_vec = GGML_F32X_MUL(v_vec, k_vec);

                        // Compute temp = kv * time_faaaa + prev_state
                        GGML_F32X temp_vec = GGML_F32X_FMA(prev_state_vec, kv_vec, time_faaaa_vec);

                        // Update dst: dst += temp * r
                        dst_vec = GGML_F32X_FMA(dst_vec, temp_vec, r_vec);
                        GGML_F32X_STORE(&dst_data[t_h_j_offset], dst_vec);

                        // Update state: state = prev_state * time_decay + kv
                        GGML_F32X new_state_vec = GGML_F32X_FMA(kv_vec, prev_state_vec, time_decay_vec);
                        GGML_F32X_STORE(&state_cur[h_2d_i_j_offset], new_state_vec);
                    }

                    // Handle remaining elements, this will not be used.
                    for (int64_t j = vec_count * wkv_vector_size; j < head_size; j++) {
                        size_t t_h_j_offset = t_h_offset + j;
                        size_t h_2d_i_j_offset = h_2d_i_offset + j;
                        float v_val = v[t_h_j_offset];
                        float kv_val = v_val * k_val;
                        float prev_state_val = state_prev[h_2d_i_j_offset];
                        float temp_val = kv_val * time_faaaa_val + prev_state_val;
                        dst_data[t_h_j_offset] += temp_val * r_val;
                        state_cur[h_2d_i_j_offset] = prev_state_val * time_decay_val + kv_val;
                    }
                }
            }
        }

    #else
        // basically fused operations:
        // dst = r @ (time_faaaa * (k @ v) + state),
        // state = time_decay * state + (k @ v),
        // recursive through each token
        for (int64_t t = 0; t < T; t++) {
            size_t t_offset = t * t_stride;
            size_t state_offset = head_size * C * (t / (T / n_seqs));
            float * state_cur = state + state_offset;
            float * state_prev = t % (T / n_seqs) ? state_cur : (float*)dst->src[5]->data + state_offset;

            for (int64_t h = h_start; h < h_end; h++) {
                size_t h_offset = h * h_stride;
                size_t t_h_offset = t_offset + h_offset;
                size_t h_2d_offset = h * h_stride_2d;

                for (int64_t i = 0; i < head_size; i++) {
                    size_t t_h_i_offset = t_h_offset + i;
                    size_t h_i_offset = h_offset + i;
                    size_t h_2d_i_offset = h_2d_offset + i * h_stride;

                    float k_val = k[t_h_i_offset];
                    float r_val = r[t_h_i_offset];
                    float time_faaaa_val = time_faaaa[h_i_offset];
                    // RWKV v6: different time_decay for each token.
                    float time_decay_val = time_decay[t_h_i_offset];

                    for (int64_t j = 0; j < head_size; j++) {
                        size_t t_h_j_offset = t_h_offset + j;
                        size_t h_2d_i_j_offset = h_2d_i_offset + j;

                        float v_val = v[t_h_j_offset];
                        float kv_val = v_val * k_val;
                        float prev_state_val = state_prev[h_2d_i_j_offset];
                        float temp_val = kv_val * time_faaaa_val + prev_state_val;
                        dst_data[t_h_j_offset] += temp_val * r_val;
                        state_cur[h_2d_i_j_offset] = prev_state_val * time_decay_val + kv_val;
                    }
                }
            }
        }
    #endif
}


void ggml_compute_forward_rwkv_wkv6(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_rwkv_wkv6_f32(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_gla

static void ggml_compute_forward_gla_f32(
        const ggml_compute_params * params,
        ggml_tensor * dst) {
    const int64_t T = dst->src[1]->ne[2];
    const int64_t C = dst->ne[0];
    const int64_t HEADS = dst->src[1]->ne[1];
    const int64_t n_seqs = dst->src[4]->ne[1];
    const int64_t head_size = C / HEADS;
    const float scale = ggml_get_op_params_f32(dst, 0);

    float * dst_data = (float *) dst->data;
    float * state = ((float *) dst->data) + C * T;

    const int ith = params->ith;
    const int nth = params->nth;

    const int h_start =  (HEADS * (ith    )) / nth;
    const int h_end   = ((HEADS * (ith + 1)) / nth < HEADS) ?
                         (HEADS * (ith + 1)) / nth : HEADS;

    float * k = (float *) dst->src[0]->data;
    float * v = (float *) dst->src[1]->data;
    float * q = (float *) dst->src[2]->data;
    float * g = (float *) dst->src[3]->data;

    size_t t_stride = HEADS * head_size; // Same to C

    size_t h_stride = C / HEADS;
    GGML_ASSERT(C % HEADS == 0); // C must be divisible by HEADS
    size_t h_stride_2d = head_size * head_size;

    if (ith == 0) {
        memset(dst_data, 0, T * C * sizeof(float));
    }
    ggml_barrier(params->threadpool);


    #if defined(__AVX__) && !defined(__AVX512F__)
        #define GGML_F32X GGML_F32x8
        #define GGML_F32X_SET1 GGML_F32x8_SET1
        #define GGML_F32X_LOAD GGML_F32x8_LOAD
        #define GGML_F32X_STORE GGML_F32x8_STORE
        #define GGML_F32X_MUL GGML_F32x8_MUL
        #define GGML_F32X_FMA GGML_F32x8_FMA
        #define GLA_VECTOR_SIZE 8
    #elif defined(__AVX512F__)
        #define GGML_F32X GGML_F32x16
        #define GGML_F32X_SET1 GGML_F32x16_SET1
        #define GGML_F32X_LOAD GGML_F32x16_LOAD
        #define GGML_F32X_STORE GGML_F32x16_STORE
        #define GGML_F32X_MUL GGML_F32x16_MUL
        #define GGML_F32X_FMA GGML_F32x16_FMA
        #define GLA_VECTOR_SIZE 16
    #elif defined(__ARM_FEATURE_SVE) && defined(__aarch64__)
        #define GGML_F32X GGML_F32xt
        #define GGML_F32X_SET1 GGML_F32xt_SET1
        #define GGML_F32X_LOAD GGML_F32xt_LOAD
        #define GGML_F32X_STORE GGML_F32xt_STORE
        #define GGML_F32X_MUL GGML_F32xt_MUL
        #define GGML_F32X_FMA GGML_F32xt_FMA
        #define GLA_VECTOR_SIZE 8
    #elif defined(__ARM_NEON) && defined(__aarch64__)
        #define GGML_F32X GGML_F32x4
        #define GGML_F32X_SET1 GGML_F32x4_SET1
        #define GGML_F32X_LOAD GGML_F32x4_LOAD
        #define GGML_F32X_STORE GGML_F32x4_STORE
        #define GGML_F32X_MUL GGML_F32x4_MUL
        #define GGML_F32X_FMA GGML_F32x4_FMA
        #define GLA_VECTOR_SIZE 4
    #endif

    #ifdef GLA_VECTOR_SIZE
        int gla_vector_size;
        #if defined(__ARM_FEATURE_SVE)
            gla_vector_size = svcntw();
        #else
            gla_vector_size = GLA_VECTOR_SIZE;
        #endif
        const int64_t vec_count = head_size / gla_vector_size;

        for (int64_t t = 0; t < T; t++) {
            size_t t_offset = t * t_stride;
            size_t state_offset = head_size * C * (t / (T / n_seqs));
            float * state_cur = state + state_offset;
            float * state_prev = t % (T / n_seqs) ? state_cur : (float*)dst->src[4]->data + state_offset;

            for (int64_t h = h_start; h < h_end; h++) {
                size_t h_offset = h * h_stride;
                size_t t_h_offset = t_offset + h_offset;
                size_t h_2d_offset = h * h_stride_2d;

                for (int64_t i = 0; i < head_size; i++) {
                    size_t t_h_i_offset = t_h_offset + i;
                    size_t h_2d_i_offset = h_2d_offset + i * h_stride;

                    float k_val = k[t_h_i_offset];
                    float q_val = q[t_h_i_offset] * scale;
                    float g_val = g[t_h_i_offset];

                    // Broadcast scalar values to vectors
                    GGML_F32X k_vec = GGML_F32X_SET1(k_val);
                    GGML_F32X q_vec = GGML_F32X_SET1(q_val);
                    GGML_F32X g_vec = GGML_F32X_SET1(g_val);

                    for (int64_t j = 0; j < vec_count; j++) {
                        size_t base_j = j * gla_vector_size;
                        size_t t_h_j_offset = t_h_offset + base_j;
                        size_t h_2d_i_j_offset = h_2d_i_offset + base_j;

                        // Load x elements at once
                        GGML_F32X v_vec = GGML_F32X_LOAD(&v[t_h_j_offset]);
                        GGML_F32X prev_state_vec = GGML_F32X_LOAD(&state_prev[h_2d_i_j_offset]);
                        GGML_F32X dst_vec = GGML_F32X_LOAD(&dst_data[t_h_j_offset]);

                        // Compute kv = v * k
                        GGML_F32X kv_vec = GGML_F32X_MUL(v_vec, k_vec);

                        // Compute temp = prev_state * g + kv
                        GGML_F32X temp_vec = GGML_F32X_FMA(kv_vec, prev_state_vec, g_vec);

                        // Update dst: dst += temp * q
                        dst_vec = GGML_F32X_FMA(dst_vec, temp_vec, q_vec);
                        GGML_F32X_STORE(&dst_data[t_h_j_offset], dst_vec);

                        // Update state
                        GGML_F32X_STORE(&state_cur[h_2d_i_j_offset], temp_vec);
                    }

                    // Handle remaining elements, this will not be used.
                    for (int64_t j = vec_count * gla_vector_size; j < head_size; j++) {
                        size_t t_h_j_offset = t_h_offset + j;
                        size_t h_2d_i_j_offset = h_2d_i_offset + j;
                        float v_val = v[t_h_j_offset];
                        float kv_val = v_val * k_val;
                        float prev_state_val = state_prev[h_2d_i_j_offset];
                        float temp_val = kv_val + prev_state_val * g_val;
                        dst_data[t_h_j_offset] += temp_val * q_val;
                        state_cur[h_2d_i_j_offset] = temp_val;
                    }
                }
            }
        }

    #else
        for (int64_t t = 0; t < T; t++) {
            size_t t_offset = t * t_stride;
            size_t state_offset = head_size * C * (t / (T / n_seqs));
            float * state_cur = state + state_offset;
            float * state_prev = t % (T / n_seqs) ? state_cur : (float*)dst->src[4]->data + state_offset;

            for (int64_t h = h_start; h < h_end; h++) {
                size_t h_offset = h * h_stride;
                size_t t_h_offset = t_offset + h_offset;
                size_t h_2d_offset = h * h_stride_2d;

                for (int64_t i = 0; i < head_size; i++) {
                    size_t t_h_i_offset = t_h_offset + i;
                    size_t h_2d_i_offset = h_2d_offset + i * h_stride;

                    float k_val = k[t_h_i_offset];
                    float q_val = q[t_h_i_offset] * scale;
                    float g_val = g[t_h_i_offset];

                    for (int64_t j = 0; j < head_size; j++) {
                        size_t t_h_j_offset = t_h_offset + j;
                        size_t h_2d_i_j_offset = h_2d_i_offset + j;

                        float v_val = v[t_h_j_offset];
                        float kv_val = v_val * k_val;
                        float prev_state_val = state_prev[h_2d_i_j_offset];
                        float temp_val = prev_state_val * g_val + kv_val;
                        dst_data[t_h_j_offset] += temp_val * q_val;
                        state_cur[h_2d_i_j_offset] = temp_val;
                    }
                }
            }
        }
    #endif
}


void ggml_compute_forward_gla(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_gla_f32(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

static void ggml_compute_forward_solve_tri_f32(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];  // A (lower triangular)
    const struct ggml_tensor * src1 = dst->src[1];  // B (RHS)

    GGML_TENSOR_BINARY_OP_LOCALS;

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    GGML_ASSERT(ne00 == ne01); // A must be square
    GGML_ASSERT(ne0  == ne10); // solution cols == B cols
    GGML_ASSERT(ne1  == ne11); // solution rows == B rows

    GGML_ASSERT(ne02 == ne12 && ne12 == ne2);
    GGML_ASSERT(ne03 == ne13 && ne13 == ne3);

    const int ith = params->ith;
    const int nth = params->nth;

    const int64_t k = ne10;   // number of RHS columns
    const int64_t n = ne11;   // A is n×n
    const int64_t nr = ne02 * ne03 * k; // we're parallelizing on columns here, so seq x token x column will be the unit

    // chunks per thread
    const int64_t dr = (nr + nth - 1)/nth;

    // chunk range for this thread
    const int64_t ir0 = dr*ith;
    const int64_t ir1 = MIN(ir0 + dr, nr);

    const float * A = (const float *) src0->data;  // [n, n, B1, B2]
    const float * B = (const float *) src1->data;  // [n, k, B1, B2]
          float * X = (      float *) dst->data;   // [n, k, B1, B2]

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t i03 = ir/(ne02*k);
        const int64_t i02 = (ir - i03*ne02*k)/k;
        const int64_t i01 = (ir - i03*ne02*k - i02*k);

        const float * A_batch = A + i02 * nb02 / sizeof(float) + i03 * nb03 / sizeof(float);
        const float * B_batch = B + i02 * nb12 / sizeof(float) + i03 * nb13 / sizeof(float);

        float * X_batch = X + i02 * nb2 / sizeof(float) + i03 * nb3 / sizeof(float);

        for (int64_t i00 = 0; i00 < n; ++i00) {
            float sum = 0.0f;
            for (int64_t t = 0; t < i00; ++t) {
                sum += A_batch[i00 * n + t] * X_batch[t * k + i01];
            }

            const float diag = A_batch[i00 * n + i00];
            assert(diag != 0.0f && "Zero diagonal in triangular matrix");

            X_batch[i00 * k + i01] = (B_batch[i00 * k + i01] - sum) / diag;
        }
    }
}

void ggml_compute_forward_solve_tri(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32) {
        ggml_compute_forward_solve_tri_f32(params, dst);
    } else {
        GGML_ABORT("fatal error");
    }
}

// ggml_compute_forward_gated_delta_net
static void ggml_compute_forward_gated_delta_net_one_chunk(
    const ggml_compute_params * params,
    ggml_tensor * dst,
    int64_t ir0,
    int64_t ir1) {

    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    const int64_t S_v      = src_v->ne[0];
    const int64_t H        = src_v->ne[1];
    const int64_t n_tokens = src_v->ne[2];
    const int64_t n_seqs   = src_v->ne[3];

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    GGML_ASSERT(src_g->ne[0] == 1 || src_g->ne[0] == S_v);
    GGML_ASSERT(src_beta->ne[0] == 1);

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t,  nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t,  nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(int64_t, neg, src_g, ne);
    GGML_TENSOR_LOCALS(size_t,  nbg, src_g, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const bool kda = (neg0 == S_v);

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    const int64_t K = ggml_get_op_params_i32(dst, 0);
    GGML_ASSERT(K >= 1);
    // per-seq stride in floats (seq s starts at state + s * seq_stride)
    const int64_t state_seq_stride = src_state->nb[3] / sizeof(float);

    const int64_t per_thread = S_v + (K > 1 ? S_v * S_v : 0);
    const int ith = params->ith;

    float * delta       = (float *)params->wdata + ith * per_thread + CACHE_LINE_SIZE_F32;
    float * state_work  = K > 1 ? (delta + S_v) : nullptr;

    // output layout: [attn_scores | new_states]
    // attn_scores: S_v * H * n_tokens * n_seqs    floats
    // new_states:  S_v * S_v * H * n_seqs * K     floats  (K snapshot slots; last min(n_tokens, K))
    const int64_t attn_score_elems    = S_v * H * n_tokens * n_seqs;
    const int64_t state_size_per_snap = S_v * S_v * H * n_seqs;
    float * attn_out_base  = (float *)dst->data;
    float * state_out_base = (float *)dst->data + attn_score_elems;

    // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
    // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.

    const float * state_in_base = (const float *)src_state->data;

  //const int64_t rq1 = nev1 / neq1;
  //const int64_t rk1 = nev1 / nek1;
    const int64_t rq3 = nev3 / neq3;
    const int64_t rk3 = nev3 / nek3;

    const float scale = 1.0f / sqrtf((float) S_v);

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t iv1 = ir % H; // head_index
        const int64_t iv3 = ir / H; // sequence

        const int64_t iq1 = iv1 % neq1;
        const int64_t ik1 = iv1 % nek1;

        const int64_t iq3 = iv3 / rq3;
        const int64_t ik3 = iv3 / rk3;

        // For K=1, write directly to the single output slot to avoid an extra memcpy at the end.
        // For K>1, work in scratch and copy out per-token when the slot is in range.
        float * s_out = (K > 1)
            ? state_work
            : state_out_base + (iv3 * H + iv1) * S_v * S_v;

        // copy input state into the working buffer and operate in-place
        // state layout [S_v, S_v, H, n_seqs]: seq iv3 starts at iv3 * state_seq_stride.
        const float * s_in = state_in_base + iv3 * state_seq_stride + iv1 * S_v * S_v;
        memcpy(s_out, s_in, S_v * S_v * sizeof(float));

        // attn output pointer for first token of this (head, seq)
        float * attn_data = attn_out_base + (iv3 * n_tokens * H + iv1) * S_v;

        for (int64_t t = 0; t < n_tokens; t++) {
            const float * q_d = (const float *)((const char *)src_q->data + iq3 * nbq3 + t * nbq2 + iq1 * nbq1);
            const float * k_d = (const float *)((const char *)src_k->data + ik3 * nbk3 + t * nbk2 + ik1 * nbk1);
            const float * v_d = (const float *)((const char *)src_v->data + iv3 * nbv3 + t * nbv2 + iv1 * nbv1);

            const float beta_val = *(const float *)((const char *)src_beta->data + iv3 * nbb3 + t * nbb2 + iv1 * nbb1);
            const float * g_d    =  (const float *)((const char *)src_g->data    + iv3 * nbg3 + t * nbg2 + iv1 * nbg1);

            // state is stored transposed: s_out[j*S_v + i] = S[i][j]
            // so row j of s_out = column j of S (contiguous access)

            if (kda) {
                // precompute exp(g) into delta scratch (reused below)
                for (int64_t i = 0; i < S_v; ++i) {
                    delta[i] = expf(g_d[i]);
                }
                // S[i][:] *= exp(g[i]) => for each row j of M: M[j][i] *= exp(g[i])
                for (int64_t j = 0; j < S_v; ++j) {
                    ggml_vec_mul_f32(S_v, &s_out[j * S_v], &s_out[j * S_v], delta);
                }
            } else {
                ggml_vec_scale_f32(S_v * S_v, s_out, expf(g_d[0]));
            }

            // delta[j] = sum_i S[i][j] * k[i] = dot(row j of M, k)
            for (int64_t j = 0; j < S_v; ++j) {
                float sum = 0.0f;
                ggml_vec_dot_f32(S_v, &sum, 0, &s_out[j * S_v], 0, k_d, 0, 1);
                delta[j] = (v_d[j] - sum) * beta_val;
            }

            // outer product: S[i][j] += k[i] * delta[j] => M[j][i] += delta[j] * k[i]
            for (int64_t j = 0; j < S_v; ++j) {
                ggml_vec_mad_f32(S_v, &s_out[j * S_v], k_d, delta[j]);
            }

            // attn_out[j] = sum_i S[i][j] * q[i] = dot(row j of M, q)
            for (int64_t j = 0; j < S_v; ++j) {
                float sum = 0.0f;
                ggml_vec_dot_f32(S_v, &sum, 0, &s_out[j * S_v], 0, q_d, 0, 1);
                attn_data[j] = sum * scale;
            }

            attn_data += S_v * H; // advance to next token

            if (K > 1) {
                const int64_t target_slot = n_tokens - 1 - t;
                if (target_slot >= 0 && target_slot < K) {
                    float * curr_state_o = state_out_base + target_slot * state_size_per_snap +
                                     (iv3 * H + iv1) * S_v * S_v;
                    memcpy(curr_state_o, s_out, S_v * S_v * sizeof(float));
                }
            }
        }
    }
}


static void ggml_compute_forward_gated_delta_net_f32(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    ggml_tensor * V = dst->src[2];
    int64_t nr = V->ne[1] * V->ne[3];

    // disable for NUMA
    const bool disable_chunking = ggml_is_numa();

    int nth = params->nth;
    int ith = params->ith;

    // 4x chunks per thread
    int nth_scaled = nth * 4;
    int64_t chunk_size = (nr + nth_scaled - 1) / nth_scaled;
    int64_t nchunk     = (nr + chunk_size - 1) / chunk_size;

    if (nth == 1 || nchunk < nth || disable_chunking) {
      nchunk = nth;
    }

    if (ith == 0) {
      ggml_threadpool_chunk_set(params->threadpool, nth);
    }

    ggml_barrier(params->threadpool);

    const int64_t dr = (nr + nchunk - 1) / nchunk;

    int current_chunk = ith;

    while (current_chunk < nchunk) {
        const int64_t ir0 = dr * current_chunk;
        const int64_t ir1 = MIN(ir0 + dr, nr);

        ggml_compute_forward_gated_delta_net_one_chunk(params, dst, ir0, ir1);
        current_chunk = ggml_threadpool_chunk_add(params->threadpool, 1);
    }
}

void ggml_compute_forward_gated_delta_net(
        const ggml_compute_params * params,
        ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_gated_delta_net_f32(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}


// ggml_compute_forward_dsv4_hc_comb

static void ggml_dsv4_hc_comb_norm_cols(float * comb, float eps) {
    constexpr int64_t hc = 4;

    for (int64_t idst = 0; idst < hc; ++idst) {
        float sum = eps;
        for (int64_t isrc = 0; isrc < hc; ++isrc) {
            sum += comb[idst + hc*isrc];
        }

        const float inv_sum = 1.0f / sum;
        for (int64_t isrc = 0; isrc < hc; ++isrc) {
            comb[idst + hc*isrc] *= inv_sum;
        }
    }
}

static void ggml_dsv4_hc_comb_norm_rows(float * comb, float eps) {
    constexpr int64_t hc = 4;

    for (int64_t isrc = 0; isrc < hc; ++isrc) {
        float sum = eps;
        for (int64_t idst = 0; idst < hc; ++idst) {
            sum += comb[idst + hc*isrc];
        }

        const float inv_sum = 1.0f / sum;
        for (int64_t idst = 0; idst < hc; ++idst) {
            comb[idst + hc*isrc] *= inv_sum;
        }
    }
}

static void ggml_compute_forward_dsv4_hc_comb_f32(
        const ggml_compute_params * params,
        ggml_tensor * dst) {
    const ggml_tensor * mixes = dst->src[0];
    const ggml_tensor * scale = dst->src[1];
    const ggml_tensor * base  = dst->src[2];

    GGML_ASSERT(mixes->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32);
    GGML_ASSERT(base->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    constexpr int64_t hc = 4;
    constexpr int64_t comb_offset = 2*hc;
    constexpr int64_t hc_mix_dim = (2 + hc)*hc;

    const int64_t n_tokens = mixes->ne[1];

    GGML_ASSERT(mixes->ne[0] == hc_mix_dim);
    GGML_ASSERT(dst->ne[0] == hc);
    GGML_ASSERT(dst->ne[1] == hc);
    GGML_ASSERT(dst->ne[2] == n_tokens);
    GGML_ASSERT(scale->ne[0] >= 3);
    GGML_ASSERT(base->ne[0] == hc_mix_dim);

    GGML_TENSOR_LOCALS(size_t, nbm, mixes, nb);
    GGML_TENSOR_LOCALS(size_t, nbs, scale, nb);
    GGML_TENSOR_LOCALS(size_t, nbb, base,  nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,   nb);

    const float eps = ggml_get_op_params_f32(dst, 0);
    const int32_t n_iter = ggml_get_op_params_i32(dst, 1);
    GGML_ASSERT(n_iter > 0);

    const int ith = params->ith;
    const int nth = params->nth;

    const int64_t dr  = (n_tokens + nth - 1) / nth;
    const int64_t it0 = dr * ith;
    const int64_t it1 = MIN(it0 + dr, n_tokens);

    const float scale_comb = *(const float *) ((const char *) scale->data + 2*nbs0);

    for (int64_t it = it0; it < it1; ++it) {
        float comb[hc*hc];

        for (int64_t isrc = 0; isrc < hc; ++isrc) {
            float max = -INFINITY;
            for (int64_t idst = 0; idst < hc; ++idst) {
                const int64_t idx = idst + hc*isrc;
                const float xv = *(const float *) ((const char *) mixes->data + (comb_offset + idx)*nbm0 + it*nbm1);
                const float bv = *(const float *) ((const char *) base->data  + (comb_offset + idx)*nbb0);
                const float v = xv * scale_comb + bv;
                comb[idx] = v;
                max = MAX(max, v);
            }

            float sum = 0.0f;
            for (int64_t idst = 0; idst < hc; ++idst) {
                const int64_t idx = idst + hc*isrc;
                const float v = expf(comb[idx] - max);
                comb[idx] = v;
                sum += v;
            }

            const float inv_sum = 1.0f / sum;
            for (int64_t idst = 0; idst < hc; ++idst) {
                const int64_t idx = idst + hc*isrc;
                comb[idx] = comb[idx] * inv_sum + eps;
            }
        }

        ggml_dsv4_hc_comb_norm_cols(comb, eps);
        for (int32_t i = 1; i < n_iter; ++i) {
            ggml_dsv4_hc_comb_norm_rows(comb, eps);
            ggml_dsv4_hc_comb_norm_cols(comb, eps);
        }

        for (int64_t isrc = 0; isrc < hc; ++isrc) {
            for (int64_t idst = 0; idst < hc; ++idst) {
                const int64_t idx = idst + hc*isrc;
                *(float *) ((char *) dst->data + idst*nbd0 + isrc*nbd1 + it*nbd2) = comb[idx];
            }
        }
    }
}

void ggml_compute_forward_dsv4_hc_comb(
        const ggml_compute_params * params,
        ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_dsv4_hc_comb_f32(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_dsv4_hc_pre

static void ggml_compute_forward_dsv4_hc_pre_f32(
        const ggml_compute_params * params,
        ggml_tensor * dst) {
    const ggml_tensor * x       = dst->src[0];
    const ggml_tensor * weights = dst->src[1];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int64_t n_embd   = x->ne[0];
    const int64_t hc       = x->ne[1];
    const int64_t n_tokens = x->ne[2];

    GGML_ASSERT(dst->ne[0] == n_embd);
    GGML_ASSERT(dst->ne[1] == n_tokens);
    GGML_ASSERT(weights->ne[0] == hc);
    GGML_ASSERT(weights->ne[1] == n_tokens);

    GGML_TENSOR_LOCALS(size_t, nbx, x,       nb);
    GGML_TENSOR_LOCALS(size_t, nbw, weights, nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,     nb);

    const int ith = params->ith;
    const int nth = params->nth;

    const int64_t nr  = n_embd * n_tokens;
    const int64_t dr  = (nr + nth - 1) / nth;
    const int64_t ir0 = dr * ith;
    const int64_t ir1 = MIN(ir0 + dr, nr);

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t i0 = ir % n_embd;
        const int64_t it = ir / n_embd;

        float sum = 0.0f;
        for (int64_t ih = 0; ih < hc; ++ih) {
            const float xv = *(const float *) ((const char *) x->data       + i0*nbx0 + ih*nbx1 + it*nbx2);
            const float wv = *(const float *) ((const char *) weights->data + ih*nbw0 + it*nbw1);
            sum += xv * wv;
        }

        *(float *) ((char *) dst->data + i0*nbd0 + it*nbd1) = sum;
    }
}

void ggml_compute_forward_dsv4_hc_pre(
        const ggml_compute_params * params,
        ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_dsv4_hc_pre_f32(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_dsv4_hc_post

static void ggml_compute_forward_dsv4_hc_post_f32(
        const ggml_compute_params * params,
        ggml_tensor * dst) {
    const ggml_tensor * x        = dst->src[0];
    const ggml_tensor * residual = dst->src[1];
    const ggml_tensor * post     = dst->src[2];
    const ggml_tensor * comb     = dst->src[3];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(residual->type == GGML_TYPE_F32);
    GGML_ASSERT(post->type == GGML_TYPE_F32);
    GGML_ASSERT(comb->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int64_t n_embd   = x->ne[0];
    const int64_t n_tokens = x->ne[1];
    const int64_t hc       = residual->ne[1];

    GGML_ASSERT(dst->ne[0] == n_embd);
    GGML_ASSERT(dst->ne[1] == hc);
    GGML_ASSERT(dst->ne[2] == n_tokens);
    GGML_ASSERT(residual->ne[0] == n_embd);
    GGML_ASSERT(residual->ne[2] == n_tokens);
    GGML_ASSERT(post->ne[0] == hc);
    GGML_ASSERT(post->ne[1] == n_tokens);
    GGML_ASSERT(comb->ne[0] == hc);
    GGML_ASSERT(comb->ne[1] == hc);
    GGML_ASSERT(comb->ne[2] == n_tokens);

    GGML_TENSOR_LOCALS(size_t, nbx, x,        nb);
    GGML_TENSOR_LOCALS(size_t, nbr, residual, nb);
    GGML_TENSOR_LOCALS(size_t, nbp, post,     nb);
    GGML_TENSOR_LOCALS(size_t, nbc, comb,     nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,      nb);

    const int ith = params->ith;
    const int nth = params->nth;

    const int64_t nr  = n_embd * hc * n_tokens;
    const int64_t dr  = (nr + nth - 1) / nth;
    const int64_t ir0 = dr * ith;
    const int64_t ir1 = MIN(ir0 + dr, nr);

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t i0     = ir % n_embd;
        const int64_t idst   = (ir / n_embd) % hc;
        const int64_t it     = ir / (n_embd * hc);

        const float xv = *(const float *) ((const char *) x->data    + i0*nbx0 + it*nbx1);
        const float pv = *(const float *) ((const char *) post->data + idst*nbp0 + it*nbp1);

        float sum = xv * pv;
        for (int64_t isrc = 0; isrc < hc; ++isrc) {
            const float rv = *(const float *) ((const char *) residual->data + i0*nbr0 + isrc*nbr1 + it*nbr2);
            const float cv = *(const float *) ((const char *) comb->data     + idst*nbc0 + isrc*nbc1 + it*nbc2);
            sum += rv * cv;
        }

        *(float *) ((char *) dst->data + i0*nbd0 + idst*nbd1 + it*nbd2) = sum;
    }
}

void ggml_compute_forward_dsv4_hc_post(
        const ggml_compute_params * params,
        ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_dsv4_hc_post_f32(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_rwkv_wkv7

static void ggml_compute_forward_rwkv_wkv7_f32(
        const ggml_compute_params * params,
        ggml_tensor * dst) {
    const int64_t T = dst->src[1]->ne[2];
    const int64_t C = dst->ne[0];
    const int64_t HEADS = dst->src[1]->ne[1];
    const int64_t n_seqs = dst->src[6]->ne[1];
    const int64_t head_size = C / HEADS;

    float * dst_data = (float *) dst->data;
    float * state = ((float *) dst->data) + C * T;

    const int ith = params->ith;
    const int nth = params->nth;

    const int h_start =  (HEADS * (ith    )) / nth;
    const int h_end   = ((HEADS * (ith + 1)) / nth < HEADS) ?
                         (HEADS * (ith + 1)) / nth : HEADS;

    float * r = (float *) dst->src[0]->data;
    float * w = (float *) dst->src[1]->data;
    float * k = (float *) dst->src[2]->data;
    float * v = (float *) dst->src[3]->data;
    float * a = (float *) dst->src[4]->data;
    float * b = (float *) dst->src[5]->data;

    int64_t t_stride = HEADS * head_size; // Same to C

    int64_t h_stride = C / HEADS;
    GGML_ASSERT(C % HEADS == 0); // C must be divisible by HEADS
    int64_t h_stride_2d = head_size * head_size;

    #if defined(GGML_SIMD)
        #if defined(__ARM_FEATURE_SVE) || defined(__riscv_v_intrinsic)
            // scalar Route to scalar implementation       //TODO: Write SVE code and RVV code
            for (int64_t t = 0; t < T; t++) {
                int64_t t_offset = t * t_stride;
                int64_t state_offset = head_size * C * (t / (T / n_seqs));
                float * state_cur = state + state_offset;
                float * state_prev = t % (T / n_seqs) ? state_cur : (float*)dst->src[6]->data + state_offset;

                for (int64_t h = h_start; h < h_end; h++) {
                    int64_t h_offset = h * h_stride;
                    int64_t t_h_offset = t_offset + h_offset;
                    int64_t h_2d_offset = h * h_stride_2d;

                    for (int64_t i = 0; i < head_size; i++) {
                        int64_t t_h_i_offset = t_h_offset + i;
                        int64_t h_2d_i_offset = h_2d_offset + i * h_stride;

                        float v_val = v[t_h_i_offset];

                        float sa = 0, result = 0;
                        for (int64_t j = 0; j < head_size; j++) {
                            sa += a[t_h_offset + j] * state_prev[h_2d_i_offset + j];
                        }

                        for (int64_t j = 0; j < head_size; j++) {
                            int64_t t_h_j_offset = t_h_offset + j;
                            int64_t h_2d_i_j_offset = h_2d_i_offset + j;

                            float r_val = r[t_h_j_offset];
                            float w_val = w[t_h_j_offset];
                            float k_val = k[t_h_j_offset];
                            float b_val = b[t_h_j_offset];
                            float kv_val = v_val * k_val;
                            float prev_state_val = state_prev[h_2d_i_j_offset];
                            state_cur[h_2d_i_j_offset] = prev_state_val * w_val + kv_val + sa * b_val;
                            result += state_cur[h_2d_i_j_offset] * r_val;
                        }
                        dst_data[t_h_i_offset] = result;
                    }
                }
            }
        #else
            for (int64_t t = 0; t < T; t++) {
                int64_t t_offset = t * t_stride;
                int64_t state_offset = head_size * C * (t / (T / n_seqs));
                float * state_cur = state + state_offset;
                float * state_prev = t % (T / n_seqs) ? state_cur : (float*)dst->src[6]->data + state_offset;

                for (int64_t h = h_start; h < h_end; h++) {
                    int64_t h_offset = h * h_stride;
                    int64_t t_h_offset = t_offset + h_offset;
                    int64_t h_2d_offset = h * h_stride_2d;

                    for (int64_t ii = 0; ii < head_size; ii++) {
                        int64_t t_h_i_offset = t_h_offset + ii;
                        int64_t h_2d_i_offset = h_2d_offset + ii * h_stride;

                        GGML_F32_VEC v_vec = GGML_F32_VEC_SET1(v[t_h_i_offset]);

                        float sa = 0;
                        {
                            GGML_F32_VEC sum[GGML_F32_ARR] = { GGML_F32_VEC_ZERO };
                            GGML_F32_VEC ax[GGML_F32_ARR];
                            GGML_F32_VEC ay[GGML_F32_ARR];
                            for (int64_t j = 0; j < head_size; j += GGML_F32_STEP) {
                                for (int64_t kk = 0; kk < GGML_F32_ARR; kk++) {
                                    ax[kk] = GGML_F32_VEC_LOAD(&a[t_h_offset + j + kk * GGML_F32_EPR]);
                                    ay[kk] = GGML_F32_VEC_LOAD(&state_prev[h_2d_i_offset + j + kk * GGML_F32_EPR]);
                                    sum[kk] = GGML_F32_VEC_FMA(sum[kk], ax[kk], ay[kk]);
                                }
                            }
                            GGML_F32_VEC_REDUCE(sa, sum);
                        }

                        GGML_F32_VEC sa_vec = GGML_F32_VEC_SET1(sa);

                        int64_t j = 0;
                        GGML_F32_VEC result_vec[GGML_F32_ARR] = { GGML_F32_VEC_ZERO };
                        for (; j < head_size; j += GGML_F32_STEP) {
                            for (int64_t kk = 0; kk < GGML_F32_ARR; kk++) {
                                int64_t t_h_j_offset = t_h_offset + j + kk * GGML_F32_EPR;
                                int64_t h_2d_i_j_offset = h_2d_i_offset + j + kk * GGML_F32_EPR;

                                GGML_F32_VEC r_vec = GGML_F32_VEC_LOAD(&r[t_h_j_offset]);
                                GGML_F32_VEC w_vec = GGML_F32_VEC_LOAD(&w[t_h_j_offset]);
                                GGML_F32_VEC k_vec = GGML_F32_VEC_LOAD(&k[t_h_j_offset]);
                                GGML_F32_VEC b_vec = GGML_F32_VEC_LOAD(&b[t_h_j_offset]);

                                k_vec = GGML_F32_VEC_MUL(v_vec, k_vec);

                                GGML_F32_VEC state_vec = GGML_F32_VEC_LOAD(&state_prev[h_2d_i_j_offset]);
                                // kv + s * decay + sa * b
                                state_vec = GGML_F32_VEC_FMA(k_vec, state_vec, w_vec);
                                state_vec = GGML_F32_VEC_FMA(state_vec, sa_vec, b_vec);
                                GGML_F32_VEC_STORE(&state_cur[h_2d_i_j_offset], state_vec);

                                result_vec[kk] = GGML_F32_VEC_FMA(result_vec[kk], state_vec, r_vec);
                            }
                        }
                        GGML_F32_VEC_REDUCE(dst_data[t_h_i_offset], result_vec);

                        // There shouldn't be left-overs though.
                        for (; j < head_size; j++) {
                            int64_t t_h_j_offset = t_h_offset + j;
                            int64_t h_2d_i_j_offset = h_2d_i_offset + j;

                            float r_val = r[t_h_j_offset];
                            float w_val = w[t_h_j_offset];
                            float k_val = k[t_h_j_offset];
                            float b_val = b[t_h_j_offset];
                            float kv_val = v[t_h_i_offset] * k_val;

                            float prev_state_val = state_prev[h_2d_i_j_offset];
                            state_cur[h_2d_i_j_offset] = prev_state_val * w_val + kv_val + sa * b_val;
                            dst_data[t_h_i_offset] += state_cur[h_2d_i_j_offset] * r_val;
                        }
                    }
                }
            }
        #endif
    #else
        for (int64_t t = 0; t < T; t++) {
            int64_t t_offset = t * t_stride;
            int64_t state_offset = head_size * C * (t / (T / n_seqs));
            float * state_cur = state + state_offset;
            float * state_prev = t % (T / n_seqs) ? state_cur : (float*)dst->src[6]->data + state_offset;

            for (int64_t h = h_start; h < h_end; h++) {
                int64_t h_offset = h * h_stride;
                int64_t t_h_offset = t_offset + h_offset;
                int64_t h_2d_offset = h * h_stride_2d;

                for (int64_t i = 0; i < head_size; i++) {
                    int64_t t_h_i_offset = t_h_offset + i;
                    int64_t h_2d_i_offset = h_2d_offset + i * h_stride;

                    float v_val = v[t_h_i_offset];

                    float sa = 0, result = 0;
                    for (int64_t j = 0; j < head_size; j++) {
                        sa += a[t_h_offset + j] * state_prev[h_2d_i_offset + j];
                    }

                    for (int64_t j = 0; j < head_size; j++) {
                        int64_t t_h_j_offset = t_h_offset + j;
                        int64_t h_2d_i_j_offset = h_2d_i_offset + j;

                        float r_val = r[t_h_j_offset];
                        float w_val = w[t_h_j_offset];
                        float k_val = k[t_h_j_offset];
                        float b_val = b[t_h_j_offset];
                        float kv_val = v_val * k_val;
                        float prev_state_val = state_prev[h_2d_i_j_offset];
                        state_cur[h_2d_i_j_offset] = prev_state_val * w_val + kv_val + sa * b_val;
                        result += state_cur[h_2d_i_j_offset] * r_val;
                    }
                    dst_data[t_h_i_offset] = result;
                }
            }
        }
    #endif
}


void ggml_compute_forward_rwkv_wkv7(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_rwkv_wkv7_f32(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_map_custom1

void ggml_compute_forward_map_custom1(
        const ggml_compute_params * params,
              ggml_tensor * dst) {

    const ggml_tensor * a = dst->src[0];

    struct ggml_map_custom1_op_params p;
    memcpy(&p, dst->op_params, sizeof(p));

    p.fun(dst, a, params->ith, params->nth, p.userdata);
}

// ggml_compute_forward_map_custom2

void ggml_compute_forward_map_custom2(
        const ggml_compute_params * params,
              ggml_tensor * dst) {

    const ggml_tensor * a = dst->src[0];
    const ggml_tensor * b = dst->src[1];

    struct ggml_map_custom2_op_params p;
    memcpy(&p, dst->op_params, sizeof(p));

    p.fun(dst, a, b, params->ith, params->nth, p.userdata);
}

// ggml_compute_forward_map_custom3

void ggml_compute_forward_map_custom3(
        const ggml_compute_params * params,
              ggml_tensor * dst) {

    const ggml_tensor * a = dst->src[0];
    const ggml_tensor * b = dst->src[1];
    const ggml_tensor * c = dst->src[2];

    struct ggml_map_custom3_op_params p;
    memcpy(&p, dst->op_params, sizeof(p));

    p.fun(dst, a, b, c, params->ith, params->nth, p.userdata);
}

// ggml_compute_forward_custom

void ggml_compute_forward_custom(
    const struct ggml_compute_params * params,
          struct ggml_tensor * dst) {

    struct ggml_custom_op_params p;
    memcpy(&p, dst->op_params, sizeof(p));

    p.fun(dst, params->ith, params->nth, p.userdata);
}

// ggml_compute_forward_cross_entropy_loss

static void ggml_compute_forward_cross_entropy_loss_f32(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->nb[0] == ggml_type_size(src0->type));
    GGML_ASSERT(src1->nb[0] == ggml_type_size(src1->type));
    GGML_ASSERT(ggml_are_same_shape(src0, src1));
    GGML_ASSERT(ggml_is_scalar(dst));
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    // TODO: handle transposed/permuted matrices
    const int64_t nc = src0->ne[0];
    const int64_t nr = ggml_nrows(src0);

    const int ith = params->ith;
    const int nth = params->nth;

    float * sums =  (float *) params->wdata;
    float * st   = ((float *) params->wdata) + nth + ith*nc;
    float sum_thread = 0.0f;

    GGML_ASSERT(params->wsize >= sizeof(float) * (nth + nth * nc));

    // rows per thread
    const int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int64_t ir0 = dr*ith;
    const int64_t ir1 = MIN(ir0 + dr, nr);

    for (int64_t i1 = ir0; i1 < ir1; ++i1) {
        const float * s0 = (const float *)((const char *) src0->data + i1*src0->nb[1]);
        const float * s1 = (const float *)((const char *) src1->data + i1*src1->nb[1]);

#ifndef NDEBUG
        for (int64_t i = 0; i < nc; ++i) {
            //printf("p[%d] = %f\n", i, p[i]);
            assert(!isnan(s0[i]));
            assert(!isnan(s1[i]));
        }
#endif // NDEBUG

        float max = -INFINITY;
        ggml_vec_max_f32(nc, &max, s0);
        const ggml_float sum_softmax = ggml_vec_log_soft_max_f32(nc, st, s0, max);
        assert(sum_softmax >= 0.0);

        ggml_vec_add1_f32(nc, st, st, -sum_softmax);
        ggml_vec_mul_f32(nc, st, st, s1);

        float sum_st = 0.0f;
        ggml_vec_sum_f32(nc, &sum_st, st);
        sum_thread += sum_st;

#ifndef NDEBUG
        for (int64_t i = 0; i < nc; ++i) {
            assert(!isnan(st[i]));
            assert(!isinf(st[i]));
        }
#endif // NDEBUG
    }
    sums[ith] = sum_thread;
    ggml_barrier(params->threadpool);

    if (ith == 0) {
        float * dp = (float *) dst->data;
        ggml_vec_sum_f32(nth, dp, sums);
        dp[0] *= -1.0f / (float) nr;
    }
}

void ggml_compute_forward_cross_entropy_loss(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_cross_entropy_loss_f32(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_cross_entropy_loss_back

static void ggml_compute_forward_cross_entropy_loss_back_f32(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * grad  = dst->src[0]; // gradient of forward pass output
    const ggml_tensor * src0f = dst->src[1]; // src0 of forward pass
    const ggml_tensor * src1f = dst->src[2]; // src1 of forward pass

    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_is_contiguous(src0f));
    GGML_ASSERT(ggml_is_contiguous(src1f));
    GGML_ASSERT(ggml_is_contiguous(grad));
    GGML_ASSERT(ggml_are_same_shape(src0f, src1f) && ggml_are_same_shape(src0f, dst));

    const int64_t ith = params->ith;
    const int64_t nth = params->nth;

    // TODO: handle transposed/permuted matrices
    const int64_t nc = src0f->ne[0];
    const int64_t nr = ggml_nrows(src0f);

    // rows per thread
    const int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int64_t ir0 = dr*ith;
    const int64_t ir1 = MIN(ir0 + dr, nr);

    const float d_by_nr = ((const float *) grad->data)[0] / (float) nr;

    for (int64_t i1 = ir0; i1 < ir1; i1++) {
        float       * ds0 = (float       *)((char       *) dst->data   + i1*dst->nb[1]);
        const float * s0  = (const float *)((const char *) src0f->data + i1*src0f->nb[1]);
        const float * s1  = (const float *)((const char *) src1f->data + i1*src1f->nb[1]);

#ifndef NDEBUG
        for (int64_t i = 0; i < nc; ++i) {
            //printf("p[%d] = %f\n", i, p[i]);
            assert(!isnan(s0[i]));
            assert(!isnan(s1[i]));
        }
#endif // NDEBUG

        // soft_max
        float max = -INFINITY;
        ggml_vec_max_f32(nc, &max, s0);
        const ggml_float sum = ggml_vec_soft_max_f32(nc, ds0, s0, max);
        assert(sum > 0.0);
        ggml_vec_scale_f32(nc, ds0, 1.0/sum);

        // grad(src0f) = (softmax(src0f) - src1f) * grad(cross_entropy_loss(src0f, src1f)) / nr
        ggml_vec_sub_f32(nc, ds0, ds0, s1);
        ggml_vec_scale_f32(nc, ds0, d_by_nr);

#ifndef NDEBUG
        for (int64_t i = 0; i < nc; ++i) {
            assert(!isnan(ds0[i]));
            assert(!isinf(ds0[i]));
        }
#endif // NDEBUG
    }
}

void ggml_compute_forward_cross_entropy_loss_back(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_cross_entropy_loss_back_f32(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

static void ggml_compute_forward_opt_step_adamw_f32(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0         = dst->src[0];
    const ggml_tensor * src0_grad    = dst->src[1];
    const ggml_tensor * src0_grad_m  = dst->src[2];
    const ggml_tensor * src0_grad_v  = dst->src[3];
    const ggml_tensor * adamw_params = dst->src[4];

    GGML_ASSERT(ggml_are_same_shape(src0, src0_grad));
    GGML_ASSERT(ggml_are_same_shape(src0, src0_grad_m));
    GGML_ASSERT(ggml_are_same_shape(src0, src0_grad_v));
    GGML_ASSERT(ggml_nelements(adamw_params) == 7);

    const int ith = params->ith;
    const int nth = params->nth;

    const int nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS
    GGML_ASSERT(nb00 == sizeof(float));

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    const float * adamw_params_ptr = ggml_get_data_f32(adamw_params);

    const float alpha  = adamw_params_ptr[0];
    const float beta1  = adamw_params_ptr[1];
    const float beta2  = adamw_params_ptr[2];
    const float eps    = adamw_params_ptr[3];
    const float wd     = adamw_params_ptr[4];
    const float beta1h = adamw_params_ptr[5];
    const float beta2h = adamw_params_ptr[6];
    const float keep   = 1.f - alpha * wd;
    for (int ir = ir0; ir < ir1; ++ir) {
        const int64_t i03 = ir/(ne02*ne01);
        const int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        const int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        const size_t offset = i03*nb03 + i02*nb02 + i01*nb01;

        float       * w = (float       *) ((char       *) src0->data        + offset); // weight
        const float * g = (const float *) ((const char *) src0_grad->data   + offset); // grad
        float       * m = (float       *) ((char       *) src0_grad_m->data + offset);
        float       * v = (float       *) ((char       *) src0_grad_v->data + offset);

        for (int i00 = 0; i00 < ne00; ++i00) {
            m[i00] = m[i00]*beta1 +        g[i00]*(1.0f - beta1);
            v[i00] = v[i00]*beta2 + g[i00]*g[i00]*(1.0f - beta2);

            const float mh =       m[i00]*beta1h;
            const float vh = sqrtf(v[i00]*beta2h) + eps;

            // The weight decay is applied independently of the Adam momenta m and v.
            // This is NOT equivalent to l2 regularization that adds w[i00]*w[i00] to the loss.
            // See: https://arxiv.org/pdf/1711.05101v3.pdf
            w[i00] = w[i00] * keep - alpha * mh / vh;
        }
    }
}

void ggml_compute_forward_opt_step_adamw(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_opt_step_adamw_f32(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

static void ggml_compute_forward_opt_step_sgd_f32(const ggml_compute_params * params, ggml_tensor * dst) {
    const ggml_tensor * src0       = dst->src[0];
    const ggml_tensor * src0_grad  = dst->src[1];
    const ggml_tensor * sgd_params = dst->src[2];

    GGML_ASSERT(ggml_are_same_shape(src0, src0_grad));
    GGML_ASSERT(ggml_nelements(sgd_params) == 2);

    const int ith = params->ith;
    const int nth = params->nth;

    const int nr = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS
    GGML_ASSERT(nb00 == sizeof(float));

    // rows per thread
    const int dr = (nr + nth - 1) / nth;

    // row range for this thread
    const int ir0 = dr * ith;
    const int ir1 = MIN(ir0 + dr, nr);

    // using adamw param subset we care about - alpha, wd - could have a separate struct
    const float * sgd_params_ptr   = ggml_get_data_f32(sgd_params);
    const float   alpha            = sgd_params_ptr[0];
    const float   keep             = 1.f - alpha * sgd_params_ptr[1];

    for (int ir = ir0; ir < ir1; ++ir) {
        const int64_t i03 = ir / (ne02 * ne01);
        const int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
        const int64_t i01 = (ir - i03 * ne02 * ne01 - i02 * ne01);

        const size_t offset = i03 * nb03 + i02 * nb02 + i01 * nb01;

        float *       w = (float *) ((char *) src0->data + offset);                   // weight
        const float * g = (const float *) ((const char *) src0_grad->data + offset);  // grad

        for (int i00 = 0; i00 < ne00; ++i00) {
            w[i00] = w[i00] * keep - alpha * g[i00];
        }
    }
}

void ggml_compute_forward_opt_step_sgd(const ggml_compute_params * params, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_opt_step_sgd_f32(params, dst);
            }
            break;
        default:
            {
                GGML_ABORT("fatal error - sgd is F32 only");
            }
    }
}

static void ggml_compute_forward_fwht_f32(const ggml_compute_params * params, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_BINARY_OP_LOCALS

    const int ith = params->ith;
    const int nth = params->nth;

    const int64_t n = ne10;
    GGML_ASSERT((n & (n - 1)) == 0); // must be power of 2

    const int64_t nr = ne11 * ne12 * ne13;
    const int64_t rows_per_thread = (nr + nth - 1) / nth;
    const int64_t start_row = ith * rows_per_thread;
    const int64_t end_row = MIN(start_row + rows_per_thread, nr);

    const float scale = 1.0f / sqrtf((float)n);

#if defined(GGML_SIMD)
    const GGML_F32_VEC v_minus_one = GGML_F32_VEC_SET1(-1.0f);
#endif

    for (int64_t r = start_row; r < end_row; r++) {
        const int64_t i13 = r / (ne11 * ne12);
        const int64_t i12 = (r - i13 * ne11 * ne12) / ne11;
        const int64_t i11 = r - i13 * ne11 * ne12 - i12 * ne11;

        const float * src_row = (const float *) ((const char *) src1->data + i11 * nb11 + i12 * nb12 + i13 * nb13);
        float * dst_row = (float *) ((char *) dst->data + i11 * nb1 + i12 * nb2 + i13 * nb3);

        for (int64_t j = 0; j < n; j++) {
            dst_row[j] = src_row[j] * scale;
        }

        // Scalar passes
#if defined(GGML_SIMD)
#if defined(__ARM_FEATURE_SVE)
        const int step = svcntw();
#else
        const int step = GGML_F32_EPR;
#endif
#else
        const int step = n;
#endif
        for (int64_t len = 1; len < step && len < n; len <<= 1) {
            for (int64_t i = 0; i < n; i += 2 * len) {
                for (int64_t j = 0; j < len; j++) {
                    float u = dst_row[i + j];
                    float v = dst_row[i + len + j];
                    dst_row[i + j] = u + v;
                    dst_row[i + len + j] = u - v;
                }
            }
        }

        // SIMD passes using GGML_F32_VEC_* macros for multi-architecture support
#if defined(GGML_SIMD)
        for (int64_t len = step; len < n; len <<= 1) {
            for (int64_t i = 0; i < n; i += 2 * len) {
                for (int64_t j = 0; j < len; j += step) {
                    GGML_F32_VEC u = GGML_F32_VEC_LOAD(dst_row + i + j);
                    GGML_F32_VEC v = GGML_F32_VEC_LOAD(dst_row + i + len + j);

                    GGML_F32_VEC_STORE(dst_row + i + j,       GGML_F32_VEC_ADD(u, v));
                    GGML_F32_VEC_STORE(dst_row + i + len + j, GGML_F32_VEC_FMA(u, v, v_minus_one));
                }
            }
        }
#endif
    }
}

void ggml_compute_forward_fwht(const ggml_compute_params * params, ggml_tensor * dst) {
    const ggml_tensor * src1 = dst->src[1];

    switch (src1->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_fwht_f32(params, dst);
            }
            break;
        default:
            {
                GGML_ABORT("fatal error - fwht is F32 only");
            }
    }
}

// ggml_compute_forward_lightning_indexer

void ggml_compute_forward_lightning_indexer(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * w = dst->src[2]; // weights
    const ggml_tensor * m = dst->src[3]; // mask

    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(   q->type == GGML_TYPE_F32);
    GGML_ASSERT(   w->type == GGML_TYPE_F32);
    GGML_ASSERT(   m->type == GGML_TYPE_F16);

    GGML_TENSOR_LOCALS(int64_t, neq,  q, ne)
    GGML_TENSOR_LOCALS(size_t,  nbq,  q, nb)
    GGML_TENSOR_LOCALS(int64_t, nek,  k, ne)
    GGML_TENSOR_LOCALS(size_t,  nbk,  k, nb)
    GGML_TENSOR_LOCALS(int64_t, new,  w, ne)
    GGML_TENSOR_LOCALS(size_t,  nbw,  w, nb)
    GGML_TENSOR_LOCALS(int64_t, nem,  m, ne)
    GGML_TENSOR_LOCALS(size_t,  nbm,  m, nb)
    GGML_TENSOR_LOCALS(int64_t, ne, dst, ne)
    GGML_TENSOR_LOCALS(size_t,  nb, dst, nb)

    GGML_ASSERT( nb0 == ggml_type_size(dst->type));
    GGML_ASSERT(nbq0 == ggml_type_size(  q->type));
    GGML_ASSERT(nbk0 == ggml_type_size(  k->type));
    GGML_ASSERT(nbw0 == ggml_type_size(  w->type));
    GGML_ASSERT(nbm0 == ggml_type_size(  m->type));

    const int n_embd    = q->ne[0];
    const int n_head    = q->ne[1];
    const int n_tokens  = q->ne[2];
    const int n_stream  = q->ne[3];
    const int n_kv      = k->ne[2];

    ggml_to_float_t const k_to_float = ggml_get_type_traits(k->type)->to_float;
    GGML_ASSERT((k->type == GGML_TYPE_F32 || k_to_float) && "lightning indexer: unsupported K-type");

    const int nr  = n_kv;
    const int ith = params->ith;
    const int nth = params->nth;

    // (temporary) buffer for K converted to float
    float * k_row_f32 = (float *) params->wdata + ith*(1*n_embd + CACHE_LINE_SIZE_F32);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int s = 0; s < n_stream; ++s) {
        for (int t = 0; t < n_tokens; ++t) {
            const float       *   w_row =       (float *) ((char *)   w->data + t*nbw1 +        s*nbw3);
            const ggml_fp16_t *   m_row = (ggml_fp16_t *) ((char *)   m->data + t*nbm1 + (s%nem3)*nbm3);
            float             * dst_row =       (float *) ((char *) dst->data + t*nb1  +        s*nb3 );
            for (int ik = ir0; ik < ir1; ++ik) {
                char * k_row = (char *) k->data + ik*nbk2 + s*nbk3;
                if (k_to_float) {
                    k_to_float(k_row, k_row_f32, n_embd);
                } else {
                    k_row_f32 = (float *) k_row;
                }
                float score = 0.0f;
                for (int h = 0; h < n_head; ++h) {
                    // dot product of q and k for head h
                    float qk = 0.0f;
                    const float * q_row = (float *) ((char *) q->data + h*nbq1 + t*nbq2 + s*nbq3);
                    ggml_vec_dot_f32(n_embd, &qk, 0, q_row, 0, k_row_f32, 0, 1);
                    // ReLU and weights (prescaled)
                    score += MAX(qk, 0.0f) * w_row[h];
                }
                // apply mask
                dst_row[ik] = score + GGML_CPU_FP16_TO_FP32(m_row[ik]);
            }
        }
    }
}

// ggml_compute_forward_mach1_ne_mm
// Transform-free Lloyd bitshift trellis (Mach-1 NE tier): decode one row into
// a stack buffer (bit-exact with the reference numpy decode: f32(lut) * f32(gscale),
// single fp32 product per weight), then dot with every token column.

void ggml_compute_forward_mach1_ne_mm(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    const struct ggml_tensor * packed = dst->src[0];
    const struct ggml_tensor * gscale = dst->src[1];
    const struct ggml_tensor * lut    = dst->src[2];
    const struct ggml_tensor * x      = dst->src[3];

    GGML_ASSERT(ggml_is_contiguous(x));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int64_t T  = x->ne[0];
    const int64_t B  = packed->ne[1];
    const int64_t nt = x->ne[1]*x->ne[2]*x->ne[3];
    const int     k  = (int)(packed->ne[0]*8 / T);
    const int64_t ng = gscale->ne[0];                    // groups per row (T/128)
    const int64_t group = T / ng;                        // 128
    const int64_t rows_per_chunk = B / lut->ne[1];

    const uint8_t     * pk = (const uint8_t     *) packed->data;
    const ggml_fp16_t * gs = (const ggml_fp16_t *) gscale->data;
    const ggml_fp16_t * lt = (const ggml_fp16_t *) lut->data;
    const float       * xd = (const float       *) x->data;
    float             * yd = (float             *) dst->data;

    const size_t row_bytes = (size_t) packed->ne[0];

    float wbuf[4096];
    GGML_ASSERT(T <= 4096);

    for (int64_t r = params->ith; r < B; r += params->nth) {
        const uint8_t     * pb   = pk + r*row_bytes;
        const ggml_fp16_t * lrow = lt + (r / rows_per_chunk) * 4096;
        const ggml_fp16_t * grow = gs + r*ng;

        uint32_t acc = 0, state = 0;
        int nbits = 0;
        for (int64_t g = 0; g < ng; ++g) {
            const float gsc = GGML_CPU_FP16_TO_FP32(grow[g]);
            float * wg = wbuf + g*group;
            for (int64_t t = 0; t < group; ++t) {
                while (nbits < k) { acc = (acc << 8) | *pb++; nbits += 8; }
                nbits -= k;
                state = ((state << k) | ((acc >> nbits) & ((1u << k) - 1))) & 0xFFFu;
                wg[t] = GGML_CPU_FP16_TO_FP32(lrow[state]) * gsc;
            }
        }
        for (int64_t j = 0; j < nt; ++j) {
            float sum = 0.0f;
            ggml_vec_dot_f32((int) T, &sum, 0, wbuf, 0, (float *)(xd + j*T), 0, 1);
            yd[j*B + r] = sum;
        }
    }
}

// ggml_compute_forward_mach1_embed_rows
// int3 asymmetric grouped embedding gather: W = mn + q * (mx - mn)/7, all fp32.

void ggml_compute_forward_mach1_embed_rows(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    const struct ggml_tensor * q   = dst->src[0];
    const struct ggml_tensor * mn  = dst->src[1];
    const struct ggml_tensor * mx  = dst->src[2];
    const struct ggml_tensor * ids = dst->src[3];

    const int64_t n_embd = dst->ne[0];
    const int64_t nt     = ids->ne[0];
    const int64_t ng     = mn->ne[0];
    const int64_t group  = n_embd / ng;

    const uint8_t     * qd  = (const uint8_t     *) q->data;
    const ggml_fp16_t * mnd = (const ggml_fp16_t *) mn->data;
    const ggml_fp16_t * mxd = (const ggml_fp16_t *) mx->data;
    const int32_t     * id  = (const int32_t     *) ids->data;
    float             * yd  = (float             *) dst->data;

    const size_t row_bytes = (size_t) q->ne[0];

    for (int64_t j = params->ith; j < nt; j += params->nth) {
        const int64_t r = id[j];
        GGML_ASSERT(r >= 0 && r < q->ne[1]);
        const uint8_t * pb = qd + r*row_bytes;
        float * out = yd + j*n_embd;

        uint32_t acc = 0;
        int nbits = 0;
        for (int64_t g = 0; g < ng; ++g) {
            const float mnf  = GGML_CPU_FP16_TO_FP32(mnd[r*ng + g]);
            const float d    = GGML_CPU_FP16_TO_FP32(mxd[r*ng + g]) - mnf;
            // reference: step = max(mx - mn, 1e-8) / 7, all fp32
            const float step = (d > 1e-8f ? d : 1e-8f) / 7.0f;
            for (int64_t t = 0; t < group; ++t) {
                while (nbits < 3) { acc = (acc << 8) | *pb++; nbits += 8; }
                nbits -= 3;
                const uint32_t qv = (acc >> nbits) & 0x7u;
                // two statements on purpose: the reference rounds q*step to fp32
                // BEFORE the add — do not let the compiler contract into an fma
                volatile float prod = (float) qv * step;
                out[g*group + t] = mnf + prod;
            }
        }
    }
}

// ggml_compute_forward_mach1_exp_mm
// Bitshift-trellis routed experts (Mach-1 expert tier). For each expert
// used in the batch, the full weight matrix is materialized in the reference
// decoder's exact fp32 op order (trellis walk -> fp16 round -> row FWHT * su
// -> col FWHT * sv), then applied to every (slot, token) pair routed to it.
// Bit-exact vs the reference decoder given the exported (Wscale-folded) su/sv.

#define GGML_MACH1_DEM_FLAG (1u << 30)   // remap: demoted-tier marker (exporter contract)

// exp_mm: fuse the walk when a group has at most this many (slot, token)
// pairs. See the cost comparison at the branch. The fused path borrows the
// materializing path's scratch, so this bound times (m + n) must stay under
// m*n - it does by three orders of magnitude at every shipped shape.
#define MACH1_EXP_FUSE_MAX_PAIRS 6

// tlut accessor: the table ships either F32 (payload v2) or PRE-ROUNDED F16
// (v3). hatWr is defined at fp16 precision, so both forms yield the same
// weight — rounding an already-rounded value is a no-op, and the sign flip
// is exact in fp16.
static inline float mach1_tlut_at(const void * td, bool f16, int64_t i) {
    return f16 ? GGML_CPU_FP16_TO_FP32(((const ggml_fp16_t *) td)[i])
               : GGML_CPU_FP16_TO_FP32(GGML_CPU_FP32_TO_FP16(((const float *) td)[i]));
}

// ---------------------------------------------------------------------------
// Expanded fp32 lattice table.
//
// The walk indexes the trellis table with a value derived from the shift
// register, then widens it. On x86 that widening is ggml_lookup_fp16_to_fp32 -
// itself a 64 KB table read - so every lattice weight costs TWO dependent
// random loads, the second addressed by the result of the first. The table is
// tiny and per-tensor, so widen it once at first use: the walk then reads V
// CONTIGUOUS floats per state, which is a single vector load, and the
// dependent chain is gone.
//
// Values are exactly what mach1_tlut_at returns, so the weights are unchanged.
// The negated copy makes the trellis sign bit an index rather than a branch;
// negating component 0 of each V-group is exact.
struct mach1_ftab {
    std::vector<float> pos;
    std::vector<float> neg;
    std::vector<uint8_t> src;   // source tlut bytes, to detect a recycled data pointer
};

// Escape hatch to the original scalar walk. Container-to-container variance on
// CPU benchmark hosts is larger than any speedup worth measuring, so the only
// honest A/B is both walks in the SAME process: set GGML_MACH1_CPU_SCALAR=1 to
// take the reference path. It is also the fallback if a payload ever trips the
// vector path's assumptions.
static bool mach1_cpu_scalar() {
    static const bool v = []() {
        const char * e = getenv("GGML_MACH1_CPU_SCALAR");
        return e && atoi(e) != 0;
    }();
    return v;
}

static const mach1_ftab * mach1_get_ftab(const void * td, bool f16, int64_t ne, int V) {
    static std::mutex mtx;
    static std::unordered_map<const void *, mach1_ftab> cache;
    std::lock_guard<std::mutex> lock(mtx);
    const size_t nbytes = (size_t) ne*(f16 ? sizeof(ggml_fp16_t) : sizeof(float));
    auto it = cache.find(td);
    if (it != cache.end() && it->second.src.size() == nbytes &&
        memcmp(it->second.src.data(), td, nbytes) == 0) {
        return &it->second;
    }
    // build, or rebuild in place: a model swap can hand a new tlut the address
    // of a freed one, so a pointer hit alone does not prove the table matches
    mach1_ftab & T = cache[td];
    T.pos.resize((size_t) ne);
    for (int64_t i = 0; i < ne; ++i) {
        T.pos[(size_t) i] = mach1_tlut_at(td, f16, i);
    }
    T.neg = T.pos;
    for (int64_t i = 0; i < ne; i += V) {
        T.neg[(size_t) i] = -T.neg[(size_t) i];
    }
    T.src.assign((const uint8_t *) td, (const uint8_t *) td + nbytes);
    return &T;
}

// DIRECT-WINDOW state at bit offset b2 of a tile stream: the big-endian 16-bit
// window starting there, run through the trellis map p = s*(s + 1). Lifted
// verbatim out of the walk so the row-at-a-time form below decodes identically.
static inline uint32_t mach1_walk_state(const uint16_t * tw, int64_t words, int b2) {
    const int      wi = b2 >> 4;
    const int      o  = b2 & 15;
    const uint32_t w2 = ((uint32_t) tw[wi] << 16) |
                         (uint32_t) tw[wi + 1 < words ? wi + 1 : 0];
    const uint32_t reg = (w2 >> (16 - o)) & 0xFFFFu;
    return reg*(reg + 1);
}

// Row dot for one 16-wide tile row. The scalar walk accumulates weight by
// weight; here the 16 products are summed in a vector tree and the reduction
// happens once per row, which reassociates the sum (and fuses the multiply).
// Both are within the op gate's tolerance and strictly reduce rounding.
// The accumulator lives across the whole column sweep of a row block and is
// reduced once, not once per tile: MACH1_VACC is the widest fp32 vector the
// build has, mach1_vfma16* fold one 16-wide tile row into it, MACH1_VRED
// finishes. Rows are swept eight at a time so eight live accumulators fit the
// register file on AVX2 as well as AVX-512; a row's two decode steps are
// independent of every other row's, so splitting the sweep costs no extra
// decode work.
#if defined(__AVX512F__)
    #define MACH1_HAVE_DOT16 1
    typedef __m512 mach1_vacc;
    #define MACH1_VZERO()  _mm512_setzero_ps()
    #define MACH1_VRED(a)  _mm512_reduce_add_ps(a)
    // w0[0..7] and w1[0..7] against uc[0..15]
    static inline __m512 mach1_vfma16(__m512 a, const float * w0, const float * w1,
                                      const float * uc) {
        const __m512 wv = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_loadu_ps(w0)),
                                             _mm256_loadu_ps(w1), 1);
        return _mm512_fmadd_ps(wv, _mm512_loadu_ps(uc), a);
    }
    // eight 2-float lattice pairs against uc[0..15] (V = 2 tier)
    static inline __m512 mach1_vfma16_p2(__m512 a, const float * const p[8], const float * uc) {
        __m512 wv = _mm512_castps128_ps512(_mm_castpd_ps(
                        _mm_loadh_pd(_mm_load_sd((const double *) p[0]), (const double *) p[1])));
        wv = _mm512_insertf32x4(wv, _mm_castpd_ps(
                        _mm_loadh_pd(_mm_load_sd((const double *) p[2]), (const double *) p[3])), 1);
        wv = _mm512_insertf32x4(wv, _mm_castpd_ps(
                        _mm_loadh_pd(_mm_load_sd((const double *) p[4]), (const double *) p[5])), 2);
        wv = _mm512_insertf32x4(wv, _mm_castpd_ps(
                        _mm_loadh_pd(_mm_load_sd((const double *) p[6]), (const double *) p[7])), 3);
        return _mm512_fmadd_ps(wv, _mm512_loadu_ps(uc), a);
    }
#elif defined(__AVX2__) && defined(__FMA__)
    #define MACH1_HAVE_DOT16 1
    typedef __m256 mach1_vacc;
    #define MACH1_VZERO()  _mm256_setzero_ps()
    #define MACH1_VRED(a)  mach1_hsum256(a)
    static inline float mach1_hsum256(__m256 a) {
        __m128 lo = _mm_add_ps(_mm256_castps256_ps128(a), _mm256_extractf128_ps(a, 1));
        lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
        lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 0x55));
        return _mm_cvtss_f32(lo);
    }
    static inline __m256 mach1_vfma16(__m256 a, const float * w0, const float * w1,
                                      const float * uc) {
        a = _mm256_fmadd_ps(_mm256_loadu_ps(w0), _mm256_loadu_ps(uc),     a);
        return _mm256_fmadd_ps(_mm256_loadu_ps(w1), _mm256_loadu_ps(uc + 8), a);
    }
    static inline __m256 mach1_ld4p(const float * a, const float * b,
                                    const float * c, const float * d) {
        const __m128 lo = _mm_castpd_ps(_mm_loadh_pd(_mm_load_sd((const double *) a),
                                                     (const double *) b));
        const __m128 hi = _mm_castpd_ps(_mm_loadh_pd(_mm_load_sd((const double *) c),
                                                     (const double *) d));
        return _mm256_insertf128_ps(_mm256_castps128_ps256(lo), hi, 1);
    }
    static inline __m256 mach1_vfma16_p2(__m256 a, const float * const p[8], const float * uc) {
        a = _mm256_fmadd_ps(mach1_ld4p(p[0], p[1], p[2], p[3]), _mm256_loadu_ps(uc),     a);
        return _mm256_fmadd_ps(mach1_ld4p(p[4], p[5], p[6], p[7]), _mm256_loadu_ps(uc + 8), a);
    }
#elif defined(__ARM_NEON) && defined(__aarch64__)
    #define MACH1_HAVE_DOT16 1
    typedef float32x4_t mach1_vacc;
    #define MACH1_VZERO()  vdupq_n_f32(0.0f)
    #define MACH1_VRED(a)  vaddvq_f32(a)
    static inline float32x4_t mach1_vfma16(float32x4_t a, const float * w0, const float * w1,
                                           const float * uc) {
        a = vfmaq_f32(a, vld1q_f32(w0),     vld1q_f32(uc));
        a = vfmaq_f32(a, vld1q_f32(w0 + 4), vld1q_f32(uc + 4));
        a = vfmaq_f32(a, vld1q_f32(w1),     vld1q_f32(uc + 8));
        return vfmaq_f32(a, vld1q_f32(w1 + 4), vld1q_f32(uc + 12));
    }
    static inline float32x4_t mach1_vfma16_p2(float32x4_t a, const float * const p[8],
                                              const float * uc) {
        a = vfmaq_f32(a, vcombine_f32(vld1_f32(p[0]), vld1_f32(p[1])), vld1q_f32(uc));
        a = vfmaq_f32(a, vcombine_f32(vld1_f32(p[2]), vld1_f32(p[3])), vld1q_f32(uc + 4));
        a = vfmaq_f32(a, vcombine_f32(vld1_f32(p[4]), vld1_f32(p[5])), vld1q_f32(uc + 8));
        return vfmaq_f32(a, vcombine_f32(vld1_f32(p[6]), vld1_f32(p[7])), vld1q_f32(uc + 12));
    }
#endif

// Accumulators live at once. AVX-512 has the registers to hold all 16 rows, so
// a tile decodes once; the 256-bit builds sweep 16 rows in two passes of 8.
#if defined(__AVX512F__)
    #define MACH1_VROWS 16
    #define MACH1_VW    16
    #define MACH1_VLOADU(p)         _mm512_loadu_ps(p)
    #define MACH1_VSTOREU(p, a)     _mm512_storeu_ps((p), (a))
    #define MACH1_VFMA1(acc, s, v)  _mm512_fmadd_ps(_mm512_set1_ps(s), (v), (acc))
#else
    #define MACH1_VROWS 8
    #if defined(__AVX2__) && defined(__FMA__)
        #define MACH1_VW    8
        #define MACH1_VLOADU(p)        _mm256_loadu_ps(p)
        #define MACH1_VSTOREU(p, a)    _mm256_storeu_ps((p), (a))
        #define MACH1_VFMA1(acc, s, v) _mm256_fmadd_ps(_mm256_set1_ps(s), (v), (acc))
    #elif defined(__ARM_NEON) && defined(__aarch64__)
        #define MACH1_VW    4
        #define MACH1_VLOADU(p)        vld1q_f32(p)
        #define MACH1_VSTOREU(p, a)    vst1q_f32((p), (a))
        #define MACH1_VFMA1(acc, s, v) vfmaq_n_f32((acc), (v), (s))
    #endif
#endif

// Batched (prefill) walk threshold. Below this the per-token walk above wins:
// it keeps the contraction in the vector lanes, which is the right shape when
// there is only one token to contract for.
#define MACH1_RT_BATCH_MIN 8

// The row sweeps MUST unroll: only then does the accumulator array get scalar-
// replaced into registers. Left rolled, GCC keeps it on the stack and every
// FMA becomes a load-modify-store (verified in the generated asm - one
// vfmadd213ps against memory instead of eight against registers).
#if defined(__GNUC__) && MACH1_VROWS == 16
    #define MACH1_UNROLL8 _Pragma("GCC unroll 16")
#elif defined(__GNUC__)
    #define MACH1_UNROLL8 _Pragma("GCC unroll 8")
#else
    #define MACH1_UNROLL8
#endif

// Whole-tile decode.
//
// The walk's step is K*V fresh bits and every shipped tier has step == 8, so
// the states sit at EVERY BYTE OFFSET of the tile's logical stream. That makes
// the tile decodable in two vector passes rather than one shift-and-mask
// sequence per state: the even-offset windows ARE the stored words, and the
// odd-offset windows are those words carrying in the top byte of their
// successor. p = s*(s + 1) only ever feeds a 15-bit row index plus the sign
// bit, so the low 16 bits suffice and one 16-lane multiply does a whole block.
//
// pe[j] is the state at byte offset 2j, po[j] the state at 2j + 1.
#define MACH1_TILE_WORDS_MAX 64

static inline void mach1_decode_tile(const uint16_t * tw, int64_t words,
                                     uint16_t * pe, uint16_t * po) {
    int64_t j = 0;
#if defined(__AVX2__)
    const __m256i ones = _mm256_set1_epi16(1);
    for (; j + 16 <= words; j += 16) {
        const __m256i t = _mm256_loadu_si256((const __m256i *)(tw + j));
        _mm256_storeu_si256((__m256i *)(pe + j), _mm256_mullo_epi16(t, _mm256_add_epi16(t, ones)));
        // one u16 along, wrapping inside the block - the wrapped lane is wrong
        // whenever the block is not the whole tile and is redone below
        const __m256i sh = _mm256_permute2x128_si256(t, t, 0x21);
        const __m256i nx = _mm256_alignr_epi8(sh, t, 2);
        const __m256i o  = _mm256_or_si256(_mm256_slli_epi16(t, 8), _mm256_srli_epi16(nx, 8));
        _mm256_storeu_si256((__m256i *)(po + j), _mm256_mullo_epi16(o, _mm256_add_epi16(o, ones)));
    }
    for (int64_t b = 15; b < j; b += 16) {   // repair each block's carried-in lane
        const uint32_t ro = ((tw[b] & 0xFFu) << 8) |
                             (tw[b + 1 < words ? b + 1 : 0] >> 8);
        po[b] = (uint16_t)(ro*(ro + 1));
    }
#endif
    for (; j < words; ++j) {
        const uint32_t re = tw[j];
        const uint32_t ro = ((tw[j] & 0xFFu) << 8) |
                             (tw[j + 1 < words ? j + 1 : 0] >> 8);
        pe[j] = (uint16_t)(re*(re + 1));
        po[j] = (uint16_t)(ro*(ro + 1));
    }
}

// orthonormal Walsh-Hadamard, Sylvester order: butterfly pairs at stride
// 1,2,4,... then ONE fp32 division by sqrt(d) — the reference op order
static void mach1_fwht_f32(float * v, const int64_t d) {
    for (int64_t span = 1; span < d; span <<= 1) {
#if defined(__AVX__)
        // A butterfly is elementwise over two disjoint contiguous runs, so the
        // vector form performs exactly the same additions on exactly the same
        // operands - bit-identical, not merely within tolerance. span is a
        // power of two, so span >= 8 is always a whole number of vectors.
        if (span >= 8) {
            for (int64_t base = 0; base < d; base += span << 1) {
                for (int64_t i = 0; i < span; i += 8) {
                    const __m256 a = _mm256_loadu_ps(v + base + i);
                    const __m256 b = _mm256_loadu_ps(v + base + span + i);
                    _mm256_storeu_ps(v + base + i,        _mm256_add_ps(a, b));
                    _mm256_storeu_ps(v + base + span + i, _mm256_sub_ps(a, b));
                }
            }
            continue;
        }
#endif
        for (int64_t base = 0; base < d; base += span << 1) {
            for (int64_t i = 0; i < span; ++i) {
                const float a = v[base + i];
                const float b = v[base + span + i];
                v[base + i]        = a + b;
                v[base + span + i] = a - b;
            }
        }
    }
    const float s = sqrtf((float) d);
#if defined(__AVX__)
    if ((d & 7) == 0) {
        const __m256 sv = _mm256_set1_ps(s);
        for (int64_t i = 0; i < d; i += 8) {
            _mm256_storeu_ps(v + i, _mm256_div_ps(_mm256_loadu_ps(v + i), sv));
        }
        return;
    }
#endif
    for (int64_t i = 0; i < d; ++i) {
        v[i] = v[i] / s;
    }
}

void ggml_compute_forward_mach1_exp_mm(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    const struct ggml_tensor * kept  = dst->src[0];
    const struct ggml_tensor * dem   = dst->src[1];
    const struct ggml_tensor * su    = dst->src[2];
    const struct ggml_tensor * sv    = dst->src[3];
    const struct ggml_tensor * tlut  = dst->src[4];
    const struct ggml_tensor * remap = dst->src[5];
    const struct ggml_tensor * ids   = dst->src[6];
    const struct ggml_tensor * x     = dst->src[7];
    const struct ggml_tensor * gamma = dst->src[8];   // optional per-tile wave scale (v3)

    GGML_ASSERT(ggml_is_contiguous(dst));

    const int64_t n        = su->ne[0];
    const int64_t m        = sv->ne[0];
    const int64_t n_expert = remap->ne[0];
    const int64_t n_kept   = kept->ne[2];
    const int64_t n_dem    = dem ? dem->ne[2] : 0;
    const int64_t n_groups = n_kept + n_dem;
    const int64_t n_used   = ids->ne[0];
    const int64_t n_tok    = ids->ne[1];
    const int64_t n_pairs  = n_used*n_tok;
    const int64_t tiles_y  = n/16;
    const int64_t ntiles   = (m/16)*tiles_y;
    const int64_t xne1     = x->ne[1];

    GGML_ASSERT(m <= 4096);   // column scratch bound

    // V and tlut_bits derive from the tlut dims (see ggml_mach1_exp_mm)
    const int V     = (int) tlut->ne[0];
    const int steps = 256 / V;

    const ggml_fp16_t * sud = (const ggml_fp16_t *) su->data;
    const ggml_fp16_t * svd = (const ggml_fp16_t *) sv->data;
    const void        * td  = tlut->data;
    const bool          tf16 = tlut->type == GGML_TYPE_F16;
#ifdef MACH1_HAVE_DOT16
    // widened once per tensor; the walk below reads V contiguous floats
    const mach1_ftab  * ft  = V == 8 && !mach1_cpu_scalar()
        ? mach1_get_ftab(td, tf16, ggml_nelements(tlut), V) : NULL;
#endif
    const int32_t     * rmd = (const int32_t     *) remap->data;
    const float       * xd  = (const float       *) x->data;
    const ggml_fp16_t * gmd = gamma ? (const ggml_fp16_t *) gamma->data : NULL;
    const int64_t       gml = gamma ? gamma->ne[0] : 0;
    float             * yd  = (float             *) dst->data;

    const int ith = params->ith;
    const int nth = params->nth;

    // workspace: W buffer + fp32 su/sv rows + per-group pair lists
    char * w = (char *) params->wdata;
    float   * wbuf     = (float   *) w; w += sizeof(float)*m*n;
    float   * su32     = (float   *) w; w += sizeof(float)*n;
    float   * sv32     = (float   *) w; w += sizeof(float)*m;
    int32_t * grp_cnt  = (int32_t *) w; w += sizeof(int32_t)*n_groups;
    int32_t * grp_off  = (int32_t *) w; w += sizeof(int32_t)*n_groups;
    int32_t * grp_orig = (int32_t *) w; w += sizeof(int32_t)*n_groups;
    int32_t * pairs    = (int32_t *) w; w += sizeof(int32_t)*2*n_pairs;
    GGML_ASSERT((size_t)(w - (char *) params->wdata) <= params->wsize);

    // phase 0: group (slot, token) pairs by storage expert
    if (ith == 0) {
        memset(grp_cnt, 0, sizeof(int32_t)*n_groups);
        for (int64_t t = 0; t < n_tok; ++t) {
            for (int64_t s = 0; s < n_used; ++s) {
                const int32_t id = *(const int32_t *)((const char *) ids->data + s*ids->nb[0] + t*ids->nb[1]);
                GGML_ASSERT(id >= 0 && id < n_expert);
                const int32_t rm = rmd[id];
                const int64_t g  = (rm & GGML_MACH1_DEM_FLAG) ? n_kept + (rm & ~GGML_MACH1_DEM_FLAG) : rm;
                GGML_ASSERT(g >= 0 && g < n_groups);
                grp_orig[g] = id;
                grp_cnt[g]++;
            }
        }
        int32_t off = 0;
        for (int64_t g = 0; g < n_groups; ++g) {
            grp_off[g] = off;
            off += grp_cnt[g];
            grp_cnt[g] = 0;
        }
        for (int64_t t = 0; t < n_tok; ++t) {
            for (int64_t s = 0; s < n_used; ++s) {
                const int32_t id = *(const int32_t *)((const char *) ids->data + s*ids->nb[0] + t*ids->nb[1]);
                const int32_t rm = rmd[id];
                const int64_t g  = (rm & GGML_MACH1_DEM_FLAG) ? n_kept + (rm & ~GGML_MACH1_DEM_FLAG) : rm;
                const int64_t p  = grp_off[g] + grp_cnt[g]++;
                pairs[2*p + 0] = (int32_t) s;
                pairs[2*p + 1] = (int32_t) t;
            }
        }
    }
    ggml_barrier(params->threadpool);

    for (int64_t g = 0; g < n_groups; ++g) {
        if (grp_cnt[g] == 0) {
            continue;   // uniform across threads: wdata is shared
        }
        const bool     is_dem = g >= n_kept;
        const int64_t  ei     = is_dem ? g - n_kept : g;
        const int64_t  words  = is_dem ? 16 : kept->ne[0];   // K*16 codes per tile
        const int      step   = (int)(words*V/16);           // K*V fresh bits per step
        const int64_t  e_orig = grp_orig[g];
        const uint16_t * tr = (const uint16_t *)(is_dem ? dem->data : kept->data) + ei*ntiles*words;

        // FUSED PATH (few pairs on this expert - the decode case, where every
        // expert sees one token). Transform the vectors instead of the matrix,
        // exactly as the CUDA walk does. Cost per group:
        //     fused        cnt*(decode + mac)*m*n
        //     materialize  (decode + log)*m*n + cnt*mac*m*n
        // so fusing wins while (cnt-1)*decode < log, i.e. cnt below ~6 at
        // these shapes - and it also avoids the m*n fp32 scratch entirely.
        // Above the threshold the materializing path below amortizes its
        // decode over the pairs and stays faster, so prefill is untouched.
        // Every thread evaluates the same cnt, so the branch is collective.
        const int64_t cnt_g = grp_cnt[g];
        if (cnt_g > 0 && cnt_g <= MACH1_EXP_FUSE_MAX_PAIRS) {
            float * ubuf = wbuf;                                  // [cnt][n]
            float * vbuf = wbuf + MACH1_EXP_FUSE_MAX_PAIRS*n;     // [cnt][m]
            // gamma-folded copy of u, private to this thread's row block
            float * usc  = vbuf + MACH1_EXP_FUSE_MAX_PAIRS*m + (size_t) ith*n;
            GGML_ASSERT((size_t) MACH1_EXP_FUSE_MAX_PAIRS*(n + m) + (size_t) nth*n <= (size_t) m*n);
            GGML_UNUSED(usc);

            if (ith == 0) {
                for (int64_t j = 0; j < n; ++j) {
                    su32[j] = GGML_CPU_FP16_TO_FP32(sud[e_orig*n + j]);
                }
                for (int64_t i = 0; i < m; ++i) {
                    sv32[i] = GGML_CPU_FP16_TO_FP32(svd[e_orig*m + i]);
                }
            }
            ggml_barrier(params->threadpool);

            // u = H_n(su .* x)/sqrt(n) per pair
            for (int64_t k = ith; k < cnt_g; k += nth) {
                const int64_t pp = grp_off[g] + k;
                const int64_t sI = pairs[2*pp + 0];
                const int64_t tI = pairs[2*pp + 1];
                const float * xc = xd + (xne1 == 1 ? 0 : sI*n) + tI*xne1*n;
                float * u = ubuf + k*n;
                for (int64_t j = 0; j < n; ++j) {
                    u[j] = su32[j]*xc[j];
                }
                mach1_fwht_f32(u, n);
            }
            ggml_barrier(params->threadpool);

            // walk: decode each tile straight into the row-block accumulators
            for (int64_t rb = ith; rb < m/16; rb += nth) {
                for (int64_t k = 0; k < cnt_g; ++k) {
                    const float * u = ubuf + k*n;
                    float acc[16] = { 0.0f };

#ifdef MACH1_HAVE_DOT16
                    // step == 8 is what puts the states at every byte offset,
                    // which mach1_decode_tile relies on; anything else takes the
                    // scalar walk below
                    if (ft && step == 8 && words <= MACH1_TILE_WORDS_MAX) {
                        // The wave gamma is constant over a tile, so fold it into u
                        // once per row block instead of scaling every tile's result.
                        // That leaves nothing tile-local in the accumulator, so it
                        // can live across the whole column sweep and reduce once -
                        // the reduction was the largest non-decode term per row.
                        const float * ur = u;
                        if (gmd) {
                            for (int64_t cb = 0; cb < tiles_y; ++cb) {
                                const int64_t Mb = m/16, Nb = tiles_y;
                                const int64_t wv = (rb + cb <= Nb - 1) ? Mb + Nb - 1 - (rb + cb)
                                                                       : Mb + Nb - 2 - (rb + cb);
                                const float gsc = GGML_CPU_FP16_TO_FP32(gmd[e_orig*gml + wv]);
                                for (int j = 0; j < 16; ++j) {
                                    usc[cb*16 + j] = u[cb*16 + j]*gsc;
                                }
                            }
                            ur = usc;
                        }
                        uint16_t pe[MACH1_TILE_WORDS_MAX], po[MACH1_TILE_WORDS_MAX];
                        const float * fp = ft->pos.data();
                        const float * fn = ft->neg.data();
                        for (int r0 = 0; r0 < 16; r0 += MACH1_VROWS) {
                            mach1_vacc a[MACH1_VROWS];
                            MACH1_UNROLL8
                            for (int q = 0; q < MACH1_VROWS; ++q) {
                                a[q] = MACH1_VZERO();
                            }
                            for (int64_t cb = 0; cb < tiles_y; ++cb) {
                                const uint16_t * tw = tr + (rb*tiles_y + cb)*words;
                                const float    * uc = ur + cb*16;
                                mach1_decode_tile(tw, words, pe, po);
                                MACH1_UNROLL8
                                for (int q = 0; q < MACH1_VROWS; ++q) {
                                    // row r's two steps are the states at byte
                                    // offsets 2r and 2r + 1
                                    const int      r  = r0 + q;
                                    const uint32_t p0 = pe[r];
                                    const uint32_t p1 = po[r];
                                    const float * w0 = ((p0 & 0x8000u) ? fn : fp) + 8*(p0 & 0x7FFFu);
                                    const float * w1 = ((p1 & 0x8000u) ? fn : fp) + 8*(p1 & 0x7FFFu);
                                    a[q] = mach1_vfma16(a[q], w0, w1, uc);
                                }
                            }
                            for (int q = 0; q < MACH1_VROWS; ++q) {
                                acc[r0 + q] = MACH1_VRED(a[q]);
                            }
                        }
                    } else
#endif
                    for (int64_t cb = 0; cb < tiles_y; ++cb) {
                        const int64_t   tid = rb*tiles_y + cb;
                        const uint16_t * tw = tr + tid*words;
                        const float    * uc = u + cb*16;

                        float gsc = 0.0f;
                        if (gmd) {
                            const int64_t a = rb, b = cb;
                            const int64_t Mb = m/16, Nb = tiles_y;
                            const int64_t wv = (a + b <= Nb - 1) ? Mb + Nb - 1 - (a + b)
                                                                 : Mb + Nb - 2 - (a + b);
                            gsc = GGML_CPU_FP16_TO_FP32(gmd[e_orig*gml + wv]);
                        }

                        for (int i = 0; i < steps; ++i) {
                            const uint32_t pz  = mach1_walk_state(tw, words, step*i);
                            const uint32_t row = V == 2 ? (pz >> 6) & 511 : pz & 0x7FFFu;
                            const int64_t  tv  = (int64_t) V*row;
                            const int ri = (V*i) >> 4;
                            const int ci = (V*i) & 15;
                            for (int c = 0; c < V; ++c) {
                                float w16 = mach1_tlut_at(td, tf16, tv + c);
                                if (c == 0 && (pz & 0x8000u)) {
                                    w16 = -w16;
                                }
                                acc[ri] += (gmd ? w16*gsc : w16) * uc[ci + c];
                            }
                        }
                    }

                    float * v = vbuf + k*m + rb*16;
                    for (int i = 0; i < 16; ++i) {
                        v[i] = acc[i];
                    }
                }
            }
            ggml_barrier(params->threadpool);

            // y = sv .* H_m(v)/sqrt(m) per pair
            for (int64_t k = ith; k < cnt_g; k += nth) {
                const int64_t pp = grp_off[g] + k;
                const int64_t sI = pairs[2*pp + 0];
                const int64_t tI = pairs[2*pp + 1];
                float * v = vbuf + k*m;
                mach1_fwht_f32(v, m);
                for (int64_t i = 0; i < m; ++i) {
                    yd[tI*n_used*m + sI*m + i] = v[i]*sv32[i];
                }
            }
            ggml_barrier(params->threadpool);   // wbuf is reused by the next group
            continue;
        }

        // phase A: decode tiles -> hatWr (fp16-rounded) into wbuf
        if (ith == 0) {
            for (int64_t j = 0; j < n; ++j) {
                su32[j] = GGML_CPU_FP16_TO_FP32(sud[e_orig*n + j]);
            }
            for (int64_t i = 0; i < m; ++i) {
                sv32[i] = GGML_CPU_FP16_TO_FP32(svd[e_orig*m + i]);
            }
        }
        for (int64_t tid = ith; tid < ntiles; tid += nth) {
            const uint16_t * tw = tr + tid*words;
            float * wt = wbuf + (tid/tiles_y)*16*n + (tid % tiles_y)*16;

            // per-tile wave gamma (v3): index formula as in ggml_mach1_exp_mm
            float gsc = 0.0f;
            if (gmd) {
                const int64_t a = tid / tiles_y, b = tid % tiles_y;
                const int64_t Mb = m/16, Nb = tiles_y;
                const int64_t wv = (a + b <= Nb - 1) ? Mb + Nb - 1 - (a + b)
                                                     : Mb + Nb - 2 - (a + b);
                gsc = GGML_CPU_FP16_TO_FP32(gmd[e_orig*gml + wv]);
            }

            // MSB-first big-endian 16-bit words, tail-biting. DIRECT-WINDOW
            // states: the shift register after step i holds exactly the 16-bit
            // window of the tile stream starting at bit step*i (wrapping) —
            // verified bit-identical to the serial walk on the shipped
            // payloads. steps*step == words*16 by construction, so the bit
            // offset never needs a modulo; only the word pair wraps. No serial
            // dependency between steps.
            for (int i = 0; i < steps; ++i) {   // T/V steps
                const int      b  = step*i;
                const int      wi = b >> 4;
                const int      o  = b & 15;
                const uint32_t w2 = ((uint32_t) tw[wi] << 16) |
                                     (uint32_t) tw[wi + 1 < words ? wi + 1 : 0];
                const uint32_t reg = (w2 >> (16 - o)) & 0xFFFFu;

                // quantlut_sym: p = s*(s+1) exact, row = (p >> (16-tlut_bits-1)) &
                // (2^tlut_bits - 1), component 0 negated iff bit 15 of p.
                // V=2: tlut_bits 9 -> (p>>6)&511; V=8: tlut_bits 15 -> p&0x7FFF.
                const uint32_t p   = reg*(reg + 1);
                const uint32_t row = V == 2 ? (p >> 6) & 511 : p & 0x7FFFu;
                const int64_t  tv  = (int64_t) V*row;
                const int ri = (V*i) >> 4;
                const int ci = (V*i) & 15;
                float * wo = wt + ri*n + ci;
#ifdef MACH1_HAVE_DOT16
                if (ft) {   // V == 8: the widened table hands over 8 contiguous floats
                    const float * ws = ((p & 0x8000u) ? ft->neg.data() : ft->pos.data()) + tv;
                    if (gmd) {
                        for (int c = 0; c < 8; ++c) {
                            wo[c] = ws[c]*gsc;
                        }
                    } else {
                        memcpy(wo, ws, 8*sizeof(float));
                    }
                } else
#endif
                for (int c = 0; c < V; ++c) {
                    // hatWr is defined at fp16 precision (reference spec); the
                    // v3 wave gamma multiplies AFTER the round, in fp32
                    float w16 = mach1_tlut_at(td, tf16, tv + c);
                    if (c == 0 && (p & 0x8000u)) {
                        w16 = -w16;       // exact in fp16 (sign bit)
                    }
                    wo[c] = gmd ? w16*gsc : w16;
                }
            }
        }
        ggml_barrier(params->threadpool);

        // phase B: FWHT each row over n, scale by su
        for (int64_t i = ith; i < m; i += nth) {
            float * row = wbuf + i*n;
            mach1_fwht_f32(row, n);
            for (int64_t j = 0; j < n; ++j) {
                row[j] *= su32[j];
            }
        }
        ggml_barrier(params->threadpool);

        // phase C: FWHT each column over m, scale by sv
        for (int64_t j = ith; j < n; j += nth) {
            float col[4096];
            for (int64_t i = 0; i < m; ++i) {
                col[i] = wbuf[i*n + j];
            }
            mach1_fwht_f32(col, m);
            for (int64_t i = 0; i < m; ++i) {
                wbuf[i*n + j] = col[i]*sv32[i];
            }
        }
        ggml_barrier(params->threadpool);

        // phase D: y[:, s, t] = W @ x[:, s|0, t] for every pair in the group
        for (int64_t i = ith; i < m; i += nth) {
            const float * wrow = wbuf + i*n;
            for (int64_t p = grp_off[g]; p < grp_off[g] + grp_cnt[g]; ++p) {
                const int64_t s = pairs[2*p + 0];
                const int64_t t = pairs[2*p + 1];
                const float * xc = xd + (xne1 == 1 ? 0 : s*n) + t*xne1*n;
                float sum = 0.0f;
                ggml_vec_dot_f32((int) n, &sum, 0, (float *) wrow, 0, (float *) xc, 0, 1);
                yd[t*n_used*m + s*m + i] = sum;
            }
        }
        ggml_barrier(params->threadpool);   // wbuf is reused by the next group
    }
}

// ggml_compute_forward_mach1_exp_basis
// Shared low-rank residual for demoted experts: y = B (c_e ⊙ (A x)) in fp32.
// Kept-expert slots produce zeros. The B accumulation keeps the reference's
// two-rounding order: p = fl(B_ij * t_j) then acc = fl(acc + p).
// Optional src[6] acc: dst = acc + y (same operand order as the ggml_add it
// replaces); kept slots copy acc through.

void ggml_compute_forward_mach1_exp_basis(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    const struct ggml_tensor * a     = dst->src[0];
    const struct ggml_tensor * b     = dst->src[1];
    const struct ggml_tensor * c     = dst->src[2];
    const struct ggml_tensor * remap = dst->src[3];
    const struct ggml_tensor * ids   = dst->src[4];
    const struct ggml_tensor * x     = dst->src[5];
    const struct ggml_tensor * accs  = dst->src[6];

    GGML_ASSERT(ggml_is_contiguous(dst));

    const int64_t n      = a->ne[0];
    const int64_t r      = a->ne[1];
    const int64_t m      = b->ne[1];
    const int64_t n_dem  = c->ne[1];
    const int64_t n_used = ids->ne[0];
    const int64_t n_tok  = ids->ne[1];
    const int64_t xne1   = x->ne[1];

    const ggml_fp16_t * ad  = (const ggml_fp16_t *) a->data;
    const ggml_fp16_t * bd  = (const ggml_fp16_t *) b->data;
    const ggml_fp16_t * cd  = (const ggml_fp16_t *) c->data;
    const int32_t     * rmd = (const int32_t     *) remap->data;
    const float       * xd  = (const float       *) x->data;
    const float       * accd = accs ? (const float *) accs->data : NULL;
    float             * yd  = (float             *) dst->data;

    for (int64_t pp = params->ith; pp < n_used*n_tok; pp += params->nth) {
        const int64_t s = pp % n_used;
        const int64_t t = pp / n_used;
        const int32_t id = *(const int32_t *)((const char *) ids->data + s*ids->nb[0] + t*ids->nb[1]);
        GGML_ASSERT(id >= 0 && id < remap->ne[0]);
        const int32_t rm = rmd[id];
        float * y = yd + t*n_used*m + s*m;
        const float * ac = accd ? accd + t*n_used*m + s*m : NULL;

        if (!(rm & GGML_MACH1_DEM_FLAG)) {
            if (ac) {
                memcpy(y, ac, sizeof(float)*m);
            } else {
                memset(y, 0, sizeof(float)*m);
            }
            continue;
        }
        const int64_t di = rm & ~GGML_MACH1_DEM_FLAG;
        GGML_ASSERT(di < n_dem);

        const float * xc = xd + (xne1 == 1 ? 0 : s*n) + t*xne1*n;

        float v[256];
        float tv[256];
        for (int64_t j = 0; j < r; ++j) {
            const ggml_fp16_t * ar = ad + j*n;
            float acc = 0.0f;
            for (int64_t i = 0; i < n; ++i) {
                acc += GGML_CPU_FP16_TO_FP32(ar[i]) * xc[i];
            }
            v[j] = acc;
        }
        for (int64_t j = 0; j < r; ++j) {
            // one rounding: t_j = fl(c_j * v_j)
            const float pr = GGML_CPU_FP16_TO_FP32(cd[di*r + j]) * v[j];
            tv[j] = pr;
        }
        for (int64_t i = 0; i < m; ++i) {
            const ggml_fp16_t * br = bd + i*r;
            float acc = 0.0f;
            for (int64_t j = 0; j < r; ++j) {
                // two statements on purpose: reference rounds B_ij*t_j before the
                // add — do not let the compiler contract into an fma
                const float pr = GGML_CPU_FP16_TO_FP32(br[j]) * tv[j];
                acc += pr;
            }
            y[i] = ac ? ac[i] + acc : acc;
        }
    }
}

// ggml_compute_forward_mach1_rt_mm
// Rotated int-lattice trellis dense matmul (payload v3 NE spine):
//   W = diag(sv) . H_m/sqrt(m) . fp16(hatWr) . H_n/sqrt(n) . diag(su)
//
// The Hadamard factors are orthogonal and symmetric, so the transforms ride
// the VECTORS instead of the matrix - the same decomposition the CUDA path
// runs as rt_u / rt_walk / rt_out:
//
//   u = H_n(su .* x)/sqrt(n)      O(n log n)   per token
//   v = hatWr . u                 O(m*n) MACs, decoded on the fly
//   y = sv .* H_m(v)/sqrt(m)      O(m log m)   per token
//
// The earlier form materialized hatWr as a full m*n fp32 matrix (67 MB at the
// GDN qkv shape), Hadamard-transformed it across every row AND every column,
// then did the matvec: O(m*n*log) of transform work and a scratch buffer
// larger than most CPU last-level caches, all repeated per op per batch.
// Walking the trellis straight into the accumulator instead makes the op
// bandwidth-bound on the code stream, which is what the 1.7 bpw payload is
// supposed to buy. Same arithmetic, ~3 orders of magnitude less of it, and
// numerically closer to the CUDA path than the materializing form was.

void ggml_compute_forward_mach1_rt_mm(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    const struct ggml_tensor * trellis = dst->src[0];
    const struct ggml_tensor * su      = dst->src[1];
    const struct ggml_tensor * sv      = dst->src[2];
    const struct ggml_tensor * tlut    = dst->src[3];
    const struct ggml_tensor * x       = dst->src[4];

    GGML_ASSERT(ggml_is_contiguous(dst));

    const int64_t n       = su->ne[0];
    const int64_t m       = sv->ne[0];
    const int64_t words   = trellis->ne[0];
    const int     step    = (int)(words*2/16);      // K*V fresh bits per step (V = 2)
    const int64_t tiles_y = n/16;
    const int64_t nt      = x->ne[1]*x->ne[2]*x->ne[3];

    // (the old m <= 8192 bound was the column-transform scratch; the fused
    // form transforms v in place, so only the trellis geometry constrains m)

    const uint16_t    * trd = (const uint16_t    *) trellis->data;
    const float       * sud = (const float       *) su->data;
    const float       * svd = (const float       *) sv->data;
    const ggml_fp16_t * td  = (const ggml_fp16_t *) tlut->data;   // pre-rounded
    const float       * xd  = (const float       *) x->data;
    float             * yd  = (float             *) dst->data;
#ifdef MACH1_HAVE_DOT16
    // widened once per tensor - see mach1_get_ftab. Gated on F16 so the SIMD
    // and scalar walks read the table the same way.
    const mach1_ftab  * ft  = tlut->type == GGML_TYPE_F16 && !mach1_cpu_scalar()
        ? mach1_get_ftab(tlut->data, true, ggml_nelements(tlut), 2) : NULL;
#endif

    const int ith = params->ith;
    const int nth = params->nth;

    GGML_ASSERT(step == 8);   // the direct-window decode below assumes K*V = 8

    // scratch: u[nt][n] then v[nt][m] - the materialized W buffer is gone.
    // The batched walk adds ut[n][ntp], u transposed so tokens sit in the
    // vector lanes.
    float * ubuf = (float *) params->wdata;
    float * vbuf = ubuf + (size_t) nt*n;
    GGML_ASSERT(params->wsize >= sizeof(float)*(size_t)(nt*n + nt*m));
#ifdef MACH1_VW
    const int64_t ntp  = ((nt + MACH1_VW - 1)/MACH1_VW)*MACH1_VW;
    const bool    bat  = ft && words <= MACH1_TILE_WORDS_MAX && nt >= MACH1_RT_BATCH_MIN &&
                         params->wsize >= sizeof(float)*(size_t)(nt*n + nt*m + n*ntp);
    float * utbuf = vbuf + (size_t) nt*m;
#endif

    // phase 1: u = H_n(su .* x)/sqrt(n), one token per thread turn
    for (int64_t t = ith; t < nt; t += nth) {
        float * u = ubuf + t*n;
        const float * xr = xd + t*n;
        for (int64_t j = 0; j < n; ++j) {
            u[j] = sud[j]*xr[j];
        }
        mach1_fwht_f32(u, n);
    }
    ggml_barrier(params->threadpool);

#ifdef MACH1_VW
    // ---- batched walk (prefill) --------------------------------------------
    //
    // The per-token walk below decodes every tile ONCE PER TOKEN and reduces
    // 16 vectors per tile row. At nt tokens that is nt times the necessary
    // decode, which is why CPU prefill ran at decode speed while the bf16
    // baseline gained 6x from batching.
    //
    // Here the contraction moves out of the lanes and TOKENS move in: a lane
    // is a token, a weight is a broadcast scalar, and out[row][tok] accumulates
    // directly - so a tile decodes once for a whole chunk of tokens and no
    // horizontal reduction happens at all.
    if (bat) {
        // u transposed once, tokens minor
        for (int64_t j = ith; j < n; j += nth) {
            float * dstc = utbuf + j*ntp;
            for (int64_t t = 0; t < nt; ++t) {
                dstc[t] = ubuf[t*n + j];
            }
            for (int64_t t = nt; t < ntp; ++t) {
                dstc[t] = 0.0f;   // padding lanes contribute nothing real
            }
        }
        ggml_barrier(params->threadpool);

        const float * fp = ft->pos.data();
        const float * fn = ft->neg.data();
        const int64_t row_blocks_b = m/16;
        const int64_t chunks       = ntp/MACH1_VW;
        uint16_t pe[MACH1_TILE_WORDS_MAX], po[MACH1_TILE_WORDS_MAX];

        for (int64_t job = ith; job < row_blocks_b*chunks; job += nth) {
            const int64_t rb = job / chunks;
            const int64_t t0 = (job % chunks)*MACH1_VW;
            float outt[16*MACH1_VW];

            for (int r0 = 0; r0 < 16; r0 += MACH1_VROWS) {
                mach1_vacc a[MACH1_VROWS];
                MACH1_UNROLL8
                for (int q = 0; q < MACH1_VROWS; ++q) {
                    a[q] = MACH1_VZERO();
                }
                for (int64_t cb = 0; cb < tiles_y; ++cb) {
                    const uint16_t * tw = trd + (rb*tiles_y + cb)*words;
                    mach1_decode_tile(tw, words, pe, po);
                    const float * wr[MACH1_VROWS][8];
                    MACH1_UNROLL8
                    for (int q = 0; q < MACH1_VROWS; ++q) {
                        const int r = r0 + q;
                        for (int d = 0; d < 4; ++d) {
                            const uint32_t ze = pe[4*r + d];
                            const uint32_t zo = po[4*r + d];
                            wr[q][2*d + 0] = ((ze & 0x8000u) ? fn : fp) + 2*((ze >> 6) & 511);
                            wr[q][2*d + 1] = ((zo & 0x8000u) ? fn : fp) + 2*((zo >> 6) & 511);
                        }
                    }
                    // contract over the tile's 16 columns; the u vector for a
                    // column is shared by all rows, the weight is the scalar
                    for (int c = 0; c < 16; ++c) {
                        const mach1_vacc uv = MACH1_VLOADU(utbuf + (cb*16 + c)*ntp + t0);
                        MACH1_UNROLL8
                        for (int q = 0; q < MACH1_VROWS; ++q) {
                            a[q] = MACH1_VFMA1(a[q], wr[q][c >> 1][c & 1], uv);
                        }
                    }
                }
                MACH1_UNROLL8
                for (int q = 0; q < MACH1_VROWS; ++q) {
                    MACH1_VSTOREU(outt + (r0 + q)*MACH1_VW, a[q]);
                }
            }
            // back to token-major for the column transform
            const int64_t tend = nt - t0 < MACH1_VW ? nt - t0 : MACH1_VW;
            for (int64_t tt = 0; tt < tend; ++tt) {
                float * v = vbuf + (t0 + tt)*m + rb*16;
                for (int r = 0; r < 16; ++r) {
                    v[r] = outt[r*MACH1_VW + tt];
                }
            }
        }
        ggml_barrier(params->threadpool);
    } else
#endif
    {

    // phase 2: the walk. A thread owns whole 16-row blocks; for each column
    // tile it decodes the tile's 256 lattice values (the DIRECT-WINDOW form
    // above - state j is the big-endian u16 at byte offset j, no serial
    // dependency) and MACs them straight into the block's accumulators.
    // Weight bytes are touched once and never written back.
    const int64_t row_blocks = m/16;
    for (int64_t rb = ith; rb < row_blocks; rb += nth) {
        for (int64_t t = 0; t < nt; ++t) {
            const float * u = ubuf + t*n;
            float acc[16] = { 0.0f };

#ifdef MACH1_HAVE_DOT16
            if (ft && words <= MACH1_TILE_WORDS_MAX) {   // eight steps feed one 16-wide row (V = 2)
                const float * fp = ft->pos.data();
                const float * fn = ft->neg.data();
                uint16_t pe[MACH1_TILE_WORDS_MAX], po[MACH1_TILE_WORDS_MAX];
                // no per-tile scale here, so the accumulators simply live across
                // the column sweep and reduce once (see MACH1_VACC)
                for (int r0 = 0; r0 < 16; r0 += MACH1_VROWS) {
                    mach1_vacc a[MACH1_VROWS];
                    MACH1_UNROLL8
                    for (int q = 0; q < MACH1_VROWS; ++q) {
                        a[q] = MACH1_VZERO();
                    }
                    for (int64_t cb = 0; cb < tiles_y; ++cb) {
                        const uint16_t * tw = trd + (rb*tiles_y + cb)*words;
                        const float    * uc = u + cb*16;
                        mach1_decode_tile(tw, words, pe, po);
                        MACH1_UNROLL8
                        for (int q = 0; q < MACH1_VROWS; ++q) {
                            // row r's eight steps are the states at byte offsets
                            // 8r .. 8r + 7, alternating even and odd
                            const int r = r0 + q;
                            const float * p[8];
                            for (int d = 0; d < 4; ++d) {
                                const uint32_t ze = pe[4*r + d];
                                const uint32_t zo = po[4*r + d];
                                p[2*d + 0] = ((ze & 0x8000u) ? fn : fp) + 2*((ze >> 6) & 511);
                                p[2*d + 1] = ((zo & 0x8000u) ? fn : fp) + 2*((zo >> 6) & 511);
                            }
                            a[q] = mach1_vfma16_p2(a[q], p, uc);
                        }
                    }
                    for (int q = 0; q < MACH1_VROWS; ++q) {
                        acc[r0 + q] = MACH1_VRED(a[q]);
                    }
                }
            } else
#endif
            for (int64_t cb = 0; cb < tiles_y; ++cb) {
                const uint16_t * tw = trd + (rb*tiles_y + cb)*words;
                const float    * uc = u + cb*16;

                for (int i = 0; i < 128; ++i) {
                    const uint32_t p   = mach1_walk_state(tw, words, 8*i);
                    const uint32_t row = (p >> 6) & 511;
                    float v0 = GGML_CPU_FP16_TO_FP32(td[2*row + 0]);
                    const float v1 = GGML_CPU_FP16_TO_FP32(td[2*row + 1]);
                    if (p & 0x8000u) {
                        v0 = -v0;                    // exact in fp16 (sign bit)
                    }
                    const int ri = i >> 3;
                    const int ci = (2*i) & 15;
                    acc[ri] += v0*uc[ci] + v1*uc[ci + 1];
                }
            }

            float * v = vbuf + t*m + rb*16;
            for (int i = 0; i < 16; ++i) {
                v[i] = acc[i];
            }
        }
    }
    ggml_barrier(params->threadpool);

    }   // end per-token walk

    // phase 3: y = sv .* H_m(v)/sqrt(m), one token per thread turn
    for (int64_t t = ith; t < nt; t += nth) {
        float * v = vbuf + t*m;
        mach1_fwht_f32(v, m);
        float * y = yd + t*m;
        for (int64_t i = 0; i < m; ++i) {
            y[i] = v[i]*svd[i];
        }
    }
}

// ggml_compute_forward_mach1_head_mm
// int5-g64 lm_head matmul: W[r,j] = q[r,j] * f32(gscale[r][j/64]) with q int5
// unpacked from 5-byte little-endian blocks (8 codes each, stored value q+16).

void ggml_compute_forward_mach1_head_mm(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    const struct ggml_tensor * qp     = dst->src[0];
    const struct ggml_tensor * gscale = dst->src[1];
    const struct ggml_tensor * x      = dst->src[2];

    GGML_ASSERT(ggml_is_contiguous(dst));

    const int64_t n     = x->ne[0];
    const int64_t vocab = qp->ne[1];
    const int64_t ng    = n/64;
    const int64_t nt    = x->ne[1]*x->ne[2]*x->ne[3];
    const int64_t row_bytes = n/8*5;

    GGML_ASSERT(n <= 8192);   // row scratch bound

    const uint8_t     * qd = (const uint8_t     *) qp->data;
    const ggml_fp16_t * gs = (const ggml_fp16_t *) gscale->data;
    const float       * xd = (const float       *) x->data;
    float             * yd = (float             *) dst->data;

    for (int64_t r = params->ith; r < vocab; r += params->nth) {
        const uint8_t     * pb   = qd + r*row_bytes;
        const ggml_fp16_t * grow = gs + r*ng;
        float wrow[8192];

        for (int64_t blk = 0; blk < n/8; ++blk) {
            const uint8_t * b = pb + blk*5;
            const uint64_t word = (uint64_t) b[0] | ((uint64_t) b[1] << 8) |
                                  ((uint64_t) b[2] << 16) | ((uint64_t) b[3] << 24) |
                                  ((uint64_t) b[4] << 32);
            for (int i = 0; i < 8; ++i) {
                const int64_t j = blk*8 + i;
                const float q = (float)((int)((word >> (5*i)) & 31u) - 16);
                // reference decode: q_f32 * gscale_f32, one fp32 rounding
                wrow[j] = q * GGML_CPU_FP16_TO_FP32(grow[j >> 6]);
            }
        }
        for (int64_t t = 0; t < nt; ++t) {
            float sum = 0.0f;
            ggml_vec_dot_f32((int) n, &sum, 0, wrow, 0, (float *)(xd + t*n), 0, 1);
            yd[t*vocab + r] = sum;
        }
    }
}

// ggml_compute_forward_mach1_embed_gather
// Lossless nibble-LUT embedding gather: 4-bit codes (low nibble = even column)
// into per-(row, group-64) bf16 LUT rows; bf16 -> f32 widening is exact.

void ggml_compute_forward_mach1_embed_gather(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    const struct ggml_tensor * codes = dst->src[0];
    const struct ggml_tensor * lut   = dst->src[1];
    const struct ggml_tensor * ids   = dst->src[2];

    const int64_t n_embd = dst->ne[0];
    const int64_t nt     = ids->ne[0];
    const int64_t ng     = n_embd/64;

    const uint8_t  * cd = (const uint8_t  *) codes->data;
    const uint16_t * ld = (const uint16_t *) lut->data;   // bf16 bit patterns
    const int32_t  * id = (const int32_t  *) ids->data;
    float          * yd = (float          *) dst->data;

    for (int64_t t = params->ith; t < nt; t += params->nth) {
        const int64_t r = id[t];
        GGML_ASSERT(r >= 0 && r < codes->ne[1]);
        const uint8_t * crow = cd + r*(n_embd/2);
        float * out = yd + t*n_embd;
        for (int64_t j = 0; j < n_embd; ++j) {
            const uint32_t q = (j & 1) ? (crow[j >> 1] >> 4) : (crow[j >> 1] & 0x0F);
            const uint32_t bits = (uint32_t) ld[(r*ng + (j >> 6))*16 + q] << 16;
            float w;
            memcpy(&w, &bits, sizeof(w));
            out[j] = w;
        }
    }
}
