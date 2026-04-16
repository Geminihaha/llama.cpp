#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifdef cl_intel_subgroups
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#elif defined(cl_khr_subgroups)
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

#define N_F16_F16 4

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mat_f16_f16(
        global char * src0,
        ulong offset0,
        global char * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne10,
        int ne11,
        int ne12,
        ulong nb10,
        ulong nb11,
        ulong nb12,
        ulong nb13,
        int ne0,
        int ne1,
        int r2,
        int r3)
{
    src0 = (global char*)((global char*)src0 + offset0);
    src1 = (global char*)((global char*)src1 + offset1);
    dst = (global float*)((global char*)dst + offsetd);

    int r0 = get_group_id(0);
    int rb = get_group_id(1)*N_F16_F16;
    int im = get_group_id(2);

    int i12 = im%ne12;
    int i13 = im/ne12;

    ulong offset_src0 = r0*nb01 + (i12/r2)*nb02 + (i13/r3)*nb03;

    global half * x = (global half *) (src0 + offset_src0);

    const int lid = get_local_id(0);
    const int lsize = get_local_size(0);

    // Local memory must be declared at kernel scope
    __local float lmem[1024];

#if defined(cl_khr_subgroups) && (__OPENCL_VERSION__ >= 300 || !defined(GGML_OPENCL_USE_ADRENO_KERNELS))
    const uint sg_id = get_sub_group_id();
    const uint sg_lid = get_sub_group_local_id();
    const uint sg_size = get_max_sub_group_size();
#endif

    if (ne00 < 128) {
        for (int row = 0; row < N_F16_F16; ++row) {
            int r1 = rb + row;
            if (r1 >= ne11) {
                break;
            }

            ulong offset_src1 = r1*nb11 + (i12   )*nb12 + (i13   )*nb13;

            global half * y = (global half *) (src1 + offset_src1);

            float sumf = 0;
#if defined(cl_khr_subgroups) && (__OPENCL_VERSION__ >= 300 || !defined(GGML_OPENCL_USE_ADRENO_KERNELS))
            for (int i = sg_lid; i < ne00; i += sg_size) {
                sumf += (float)((half) x[i] * (half) y[i]);
            }
            float all_sum = sub_group_reduce_add(sumf);
            if (sg_lid == 0) {
                dst[im*ne1*ne0 + r1*ne0 + r0] = all_sum;
            }
#else
            for (int i = lid; i < ne00; i += lsize) {
                sumf += (float)((half) x[i] * (half) y[i]);
            }
            lmem[lid] = sumf;
            barrier(CLK_LOCAL_MEM_FENCE);
            for (int i = lsize / 2; i > 0; i /= 2) {
                if (lid < i) {
                    lmem[lid] += lmem[lid + i];
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (lid == 0) {
                dst[im*ne1*ne0 + r1*ne0 + r0] = lmem[0];
            }
            barrier(CLK_LOCAL_MEM_FENCE); // Sync before next row reuse lmem
#endif
        }
    } else {
        global half4 * x4 = (global half4 *)x;
        for (int row = 0; row < N_F16_F16; ++row) {
            int r1 = rb + row;
            if (r1 >= ne11) {
                break;
            }

            ulong offset_src1 = r1*nb11 + (i12   )*nb12 + (i13   )*nb13;

            global half  * y  = (global half  *) (src1 + offset_src1);
            global half4 * y4 = (global half4 *) y;

            float sumf = 0;
#if defined(cl_khr_subgroups) && (__OPENCL_VERSION__ >= 300 || !defined(GGML_OPENCL_USE_ADRENO_KERNELS))
            for (int i = sg_lid; i < ne00/4; i += sg_size) {
                sumf += (float)((half) x4[i].s0 * (half) y4[i].s0);
                sumf += (float)((half) x4[i].s1 * (half) y4[i].s1);
                sumf += (float)((half) x4[i].s2 * (half) y4[i].s2);
                sumf += (float)((half) x4[i].s3 * (half) y4[i].s3);
            }

            float all_sum = sub_group_reduce_add(sumf);
            if (sg_lid == 0) {
                for (int i = 4*(ne00/4); i < ne00; ++i) {
                    all_sum += (float)((half) x[i] * (half) y[i]);
                }
                dst[im*ne1*ne0 + r1*ne0 + r0] = all_sum;
            }
#else
            for (int i = lid; i < ne00/4; i += lsize) {
                sumf += (float)((half) x4[i].s0 * (half) y4[i].s0);
                sumf += (float)((half) x4[i].s1 * (half) y4[i].s1);
                sumf += (float)((half) x4[i].s2 * (half) y4[i].s2);
                sumf += (float)((half) x4[i].s3 * (half) y4[i].s3);
            }
            lmem[lid] = sumf;
            barrier(CLK_LOCAL_MEM_FENCE);
            for (int i = lsize / 2; i > 0; i /= 2) {
                if (lid < i) {
                    lmem[lid] += lmem[lid + i];
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (lid == 0) {
                float all_sum = lmem[0];
                for (int i = 4*(ne00/4); i < ne00; ++i) {
                    all_sum += (float)((half) x[i] * (half) y[i]);
                }
                dst[im*ne1*ne0 + r1*ne0 + r0] = all_sum;
            }
            barrier(CLK_LOCAL_MEM_FENCE); // Sync before next row reuse lmem
#endif
        }
    }
}
