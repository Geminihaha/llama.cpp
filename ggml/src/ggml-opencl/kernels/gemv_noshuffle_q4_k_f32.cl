#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable

#ifdef cl_qcom_reqd_sub_group_size
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_64 __attribute__((qcom_reqd_sub_group_size("half")))
#endif

#define QK_K  256
#define NSUBGROUPS 4
#define SUBGROUP_SIZE 64

inline void get_scale_min_k4(
    int j,
    global const uchar * q,
    uchar * d,
    uchar * m,
    uchar mask_d6,
    uchar mask_d4,
    uchar mask_hi2
) {
    if (j < 4) {
        *d = q[j]   & mask_d6;
        *m = q[j+4] & mask_d6;
    } else {
        *d = (q[j+4] & mask_d4) | ((q[j-4] & mask_hi2) >> 2);
        *m = ((q[j+4] >> 4) & mask_d4) | ((q[j]   & mask_hi2) >> 2);
    }
}

// Optimized FP16 version of dequantization macro
#define dequantizeBlockAccum_ns_sgbroadcast_1_hi(total_sums, bits4, scale, minv, y) \
    half shared_y; \
    shared_y = sub_group_broadcast(y.s0, 0); \
    total_sums.s0 += ((half)(bits4.s0 & 0x000F) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((half)(bits4.s1 & 0x000F) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s1, 0); \
    total_sums.s0 += ((half)((bits4.s0 & 0x00F0) >> 4) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((half)((bits4.s1 & 0x00F0) >> 4) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s2, 0); \
    total_sums.s0 += ((half)((bits4.s0 & 0x0F00) >> 8) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((half)((bits4.s1 & 0x0F00) >> 8) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s3, 0); \
    total_sums.s0 += ((half)((bits4.s0 & 0xF000) >> 12) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((half)((bits4.s1 & 0xF000) >> 12) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s4, 0); \
    total_sums.s0 += ((half)(bits4.s2 & 0x000F) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((half)(bits4.s3 & 0x000F) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s5, 0); \
    total_sums.s0 += ((half)((bits4.s2 & 0x00F0) >> 4) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((half)((bits4.s3 & 0x00F0) >> 4) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s6, 0); \
    total_sums.s0 += ((half)((bits4.s2 & 0x0F00) >> 8) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((half)((bits4.s3 & 0x0F00) >> 8) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s7, 0); \
    total_sums.s0 += ((half)((bits4.s2 & 0xF000) >> 12) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((half)((bits4.s3 & 0xF000) >> 12) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s0, 1); \
    total_sums.s0 += ((half)(bits4.s4 & 0x000F) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((half)(bits4.s5 & 0x000F) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s1, 1); \
    total_sums.s0 += ((half)((bits4.s4 & 0x00F0) >> 4) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((half)((bits4.s5 & 0x00F0) >> 4) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s2, 1); \
    total_sums.s0 += ((half)((bits4.s4 & 0x0F00) >> 8) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((half)((bits4.s5 & 0x0F00) >> 8) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s3, 1); \
    total_sums.s0 += ((half)((bits4.s4 & 0xF000) >> 12) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((half)((bits4.s5 & 0xF000) >> 12) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s4, 1); \
    total_sums.s0 += ((half)(bits4.s6 & 0x000F) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((half)(bits4.s7 & 0x000F) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s5, 1); \
    total_sums.s0 += ((half)((bits4.s6 & 0x00F0) >> 4) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((half)((bits4.s7 & 0x00F0) >> 4) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s6, 1); \
    total_sums.s0 += ((half)((bits4.s6 & 0x0F00) >> 8) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((half)((bits4.s7 & 0x0F00) >> 8) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s7, 1); \
    total_sums.s0 += ((half)((bits4.s6 & 0xF000) >> 12) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((half)((bits4.s7 & 0xF000) >> 12) * scale.s1 - minv.s1) * shared_y;


#define dequantizeBlockAccum_ns_sgbroadcast_1_lo(total_sums, bits4, scale, minv, y) \
    shared_y = sub_group_broadcast(y.s0, 2); \
    total_sums.s0 += ((half)(bits4.s0 & 0x000F) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((half)(bits4.s1 & 0x000F) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s1, 2); \
    total_sums.s0 += ((half)((bits4.s0 & 0x00F0) >> 4) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((half)((bits4.s0 & 0x00F0) >> 4) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s2, 2); \
    total_sums.s0 += ((half)((bits4.s0 & 0x0F00) >> 8) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((half)((bits4.s1 & 0x0F00) >> 8) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s3, 2); \
    total_sums.s0 += ((half)((bits4.s0 & 0xF000) >> 12) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((half)((bits4.s1 & 0xF000) >> 12) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s4, 2); \
    total_sums.s0 += ((half)(bits4.s2 & 0x000F) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((half)(bits4.s3 & 0x000F) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s5, 2); \
    total_sums.s0 += ((half)((bits4.s2 & 0x00F0) >> 4) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((half)((bits4.s3 & 0x00F0) >> 4) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s6, 2); \
    total_sums.s0 += ((half)((bits4.s2 & 0x0F00) >> 8) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((half)((bits4.s3 & 0x0F00) >> 8) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s7, 2); \
    total_sums.s0 += ((half)((bits4.s2 & 0xF000) >> 12) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((half)((bits4.s3 & 0xF000) >> 12) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s0, 3); \
    total_sums.s0 += ((half)(bits4.s4 & 0x000F) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((half)(bits4.s5 & 0x000F) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s1, 3); \
    total_sums.s0 += ((half)((bits4.s4 & 0x00F0) >> 4) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((half)((bits4.s5 & 0x00F0) >> 4) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s2, 3); \
    total_sums.s0 += ((half)((bits4.s4 & 0x0F00) >> 8) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((half)((bits4.s5 & 0x0F00) >> 8) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s3, 3); \
    total_sums.s0 += ((half)((bits4.s4 & 0xF000) >> 12) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((half)((bits4.s5 & 0xF000) >> 12) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s4, 3); \
    total_sums.s0 += ((half)(bits4.s6 & 0x000F) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((half)(bits4.s7 & 0x000F) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s5, 3); \
    total_sums.s0 += ((half)((bits4.s6 & 0x00F0) >> 4) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((half)((bits4.s7 & 0x00F0) >> 4) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s6, 3); \
    total_sums.s0 += ((half)((bits4.s6 & 0x0F00) >> 8) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((half)((bits4.s7 & 0x0F00) >> 8) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s7, 3); \
    total_sums.s0 += ((half)((bits4.s6 & 0xF000) >> 12) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((half)((bits4.s7 & 0xF000) >> 12) * scale.s1 - minv.s1) * shared_y;


#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_gemv_noshuffle_q4_k_f32(
        read_only  image1d_buffer_t src0_q,
        global half2  * src0_d,
        global half2  * src0_m,
        global uchar  * src0_s,
        read_only  image1d_buffer_t src1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        uchar mask_d6,
        uchar mask_d4,
        uchar mask_hi2)
{
    // Local memory declared at kernel scope
    local half2 reduceLM[SUBGROUP_SIZE * 3];

    uint groupId = get_local_id(1);
    uint gid     = get_global_id(0);
    ushort slid  = get_sub_group_local_id();

    uint K = ne00;
    uint M = ne01;

    uint LINE_STRIDE_A  = M / 2;
    uint BLOCK_STRIDE_A = NSUBGROUPS * M;
    uint scales_per_row = (K / QK_K) * 12;

    private uint4     regA;
    private half2     regS;
    private half2     regM;
    private half8     regB; // Changed to half8 for FP16 texture read

    private half2 totalSum = (half2)(0.0h);

    for (uint k = groupId; k < (K / 32); k += NSUBGROUPS) {
        uint sb = k / 8;
        uint j  = k % 8;

        half2 d   = src0_d[gid + sb * LINE_STRIDE_A];
        half2 dm  = src0_m[gid + sb * LINE_STRIDE_A];

        global const uchar * sc0 = src0_s + 2 * gid * scales_per_row + sb * 12;
        global const uchar * sc1 = src0_s + (2 * gid + 1) * scales_per_row + sb * 12;

        uchar sv0, mn0, sv1, mn1;
        get_scale_min_k4(j, sc0, &sv0, &mn0, mask_d6, mask_d4, mask_hi2);
        get_scale_min_k4(j, sc1, &sv1, &mn1, mask_d6, mask_d4, mask_hi2);

        regS = (half2)( (half)d.s0 * (half)sv0, (half)d.s1 * (half)sv1 );
        regM = (half2)( (half)dm.s0 * (half)mn0, (half)dm.s1 * (half)mn1 );

        if (slid < 4) {
            // Using read_imageh for FP16 texture acceleration
            regB.s0123 = read_imageh(src1, (slid * 2 + k * 8));
            regB.s4567 = read_imageh(src1, (1 + slid * 2 + k * 8));
        }

        // load half weights for two blocks in consecutive rows
        regA.s0 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 0)).x;
        regA.s1 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 1)).x;
        regA.s2 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 2)).x;
        regA.s3 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 3)).x;

        // Perform computation in half precision
        dequantizeBlockAccum_ns_sgbroadcast_1_hi(totalSum, as_ushort8(regA), regS, regM, regB);

        regA.s0 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 4)).x;
        regA.s1 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 5)).x;
        regA.s2 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 6)).x;
        regA.s3 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 7)).x;

        dequantizeBlockAccum_ns_sgbroadcast_1_lo(totalSum, as_ushort8(regA), regS, regM, regB);
    }

    if (groupId == 1) {
        reduceLM[SUBGROUP_SIZE * 0 + slid] = totalSum;
    }
    if (groupId == 2) {
        reduceLM[SUBGROUP_SIZE * 1 + slid] = totalSum;
    }
    if (groupId == 3) {
        reduceLM[SUBGROUP_SIZE * 2 + slid] = totalSum;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (groupId == 0) {
        totalSum += reduceLM[SUBGROUP_SIZE * 0 + slid];
        totalSum += reduceLM[SUBGROUP_SIZE * 1 + slid];
        totalSum += reduceLM[SUBGROUP_SIZE * 2 + slid];
    }

    // 2 outputs per fiber in wave 0
    if (groupId == 0) {
        global float* final_dst = (global float*)((global char*)dst + offsetd);
        vstore2(convert_float2(totalSum), 0, &(final_dst[gid * 2]));
    }
}
