#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifdef cl_intel_subgroups
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#else
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#endif

#ifdef cl_intel_required_subgroup_size
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
#define INTEL_GPU 1
#define REQD_SUBGROUP_SIZE_16 __attribute__((intel_reqd_sub_group_size(16)))
#define REQD_SUBGROUP_SIZE_32 __attribute__((intel_reqd_sub_group_size(32)))
#elif defined(cl_qcom_reqd_sub_group_size)
// Some Adreno compilers crash with this extension even if it is reported as supported
#ifndef GGML_OPENCL_USE_ADRENO_KERNELS
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_64  __attribute__((qcom_reqd_sub_group_size("half")))
#define REQD_SUBGROUP_SIZE_128 __attribute__((qcom_reqd_sub_group_size("full")))
#else
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_64
#define REQD_SUBGROUP_SIZE_128
#endif
#else
#define REQD_SUBGROUP_SIZE_64
#define REQD_SUBGROUP_SIZE_128
#endif

#define QK8_0 32
typedef struct {
    half d;       // delta
    char qs[QK8_0]; // quants
} block_q8_0;

#define NB_Q8_0 8

#ifdef INTEL_GPU
#define N_R0_Q8_0 4 // number of rows each subgroup works on
#define N_SG_Q8_0 2 // number of subgroups in a work group
#define N_SIMDWIDTH 16 // subgroup size
#elif defined (ADRENO_GPU)
#define N_R0_Q8_0 4
#define N_SG_Q8_0 2
#define N_SIMDWIDTH 64
#endif

#ifdef INTEL_GPU
REQD_SUBGROUP_SIZE_16
#elif defined (ADRENO_GPU)
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mv_q8_0_f32(
    global char * src0,
    ulong         offset0,
    global char * src1,
    ulong         offset1,
    global char * dst,
    ulong         offsetd,
    int           ne00,
    int           ne01,
    ulong         nb01,
    ulong         nb02,
    ulong         nb03,
    int           ne12,
    ulong         nb11,
    ulong         nb12,
    ulong         nb13,
    int           ne0,
    int           ne1,
    int           r2,
    int           r3
) {
    src0 = (global char*)((global char*)src0 + offset0);
    src1 = (global char*)((global char*)src1 + offset1);
    dst  = (global char*)((global char*)dst  + offsetd);

    const int lid = get_local_id(0);
    const int lsize = get_local_size(0);

    // Local memory must be declared at kernel scope
    __local float lmem[1024];

#if defined(cl_khr_subgroups) && (__OPENCL_VERSION__ >= 300 || !defined(GGML_OPENCL_USE_ADRENO_KERNELS))
    const uint sg_id = get_sub_group_id();
    const uint sg_lid = get_sub_group_local_id();
#endif

    int nb = ne00/QK8_0;

    int r0 = get_group_id(0);
    int r1 = get_group_id(1);
    int im = get_group_id(2);

#if defined(cl_khr_subgroups) && (__OPENCL_VERSION__ >= 300 || !defined(GGML_OPENCL_USE_ADRENO_KERNELS))
    int first_row = (r0*N_SG_Q8_0 + sg_id) * N_R0_Q8_0;
#else
    int first_row = (r0*N_SG_Q8_0 + (lid / 64)) * N_R0_Q8_0;
#endif

    uint i12 = im%ne12;
    uint i13 = im/ne12;

    ulong offset_src1 = r1*nb11 + i12*nb12 + i13*nb13;
    global float * y  = (global float *) (src1 + offset_src1);

    // pointers to src0 rows
    global block_q8_0 * ax[N_R0_Q8_0];
    for (int row = 0; row < N_R0_Q8_0; ++row) {
        ulong offset_src0 = (first_row + row)*nb01 + (i12/r2)*nb02 + (i13/r3)*nb03;
        ax[row] = (global block_q8_0 *) ((global char *) src0 + offset_src0);
    }

    half yl[NB_Q8_0];
    float sumf[N_R0_Q8_0] = { 0.f };

#if defined(cl_khr_subgroups) && (__OPENCL_VERSION__ >= 300 || !defined(GGML_OPENCL_USE_ADRENO_KERNELS))
    const short ix = sg_lid/4;
    const short il = sg_lid%4;
#else
    const short ix = (lid % 64)/4;
    const short il = (lid % 64)%4;
#endif

    global float * yb = y + ix*QK8_0 + il*NB_Q8_0;

    // each thread handles NB_Q8_0 quants at a time
    for (int ib = ix; ib < nb; ib += N_SIMDWIDTH/4) {
        for (short i = 0; i < NB_Q8_0; ++i) {
            yl[i] = (half)yb[i];
        }

        for (short row = 0; row < N_R0_Q8_0; row++) {
            global char * qs = ax[row][ib].qs + il*NB_Q8_0;
            half sumq = 0.h;
            for (short iq = 0; iq < NB_Q8_0; ++iq) {
                sumq += (half)qs[iq] * yl[iq];
            }
            sumf[row] += (float)sumq * (float)ax[row][ib].d;
        }

        yb += N_SIMDWIDTH*NB_Q8_0;
    }

    global float * dst_f32 = (global float *) dst + (ulong)im*ne0*ne1 + (ulong)r1*ne0;

    for (int row = 0; row < N_R0_Q8_0; ++row) {
#if defined(cl_khr_subgroups) && (__OPENCL_VERSION__ >= 300 || !defined(GGML_OPENCL_USE_ADRENO_KERNELS))
        float tot = sub_group_reduce_add(sumf[row]);
        if (sg_lid == 0 && first_row + row < ne01) {
            dst_f32[first_row + row] = tot;
        }
#else
        // Fallback to local memory reduction for Adreno 6xx
        lmem[lid] = sumf[row];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int i = 32; i > 0; i /= 2) { // 64 threads reduction
            if ((lid % 64) < i) {
                lmem[lid] += lmem[lid + i];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if ((lid % 64) == 0 && first_row + row < ne01) {
            dst_f32[first_row + row] = lmem[lid];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
#endif
    }
}

