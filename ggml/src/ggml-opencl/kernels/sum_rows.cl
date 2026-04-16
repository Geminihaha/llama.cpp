#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#ifdef cl_khr_subgroups
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#endif

// Most devices have max workgroup size of 1024, so this is enough for subgroup
// sizes of 16, 32, 64 and 128. Increase this value for smaller subgroups sizes
#define MAX_SUBGROUPS 64
kernel void kernel_sum_rows_f32(
    global char *  src0,
    ulong           offset0,
    global char *  dst,
    ulong           offsetd,
    int             ne00,
    int             ne01,
    int             ne02,
    int             ne03,
    ulong           nb01,
    ulong           nb02,
    ulong           nb03,
    ulong           nb1,
    ulong           nb2,
    ulong           nb3
) {
    src0 = src0 + offset0;
    dst  = dst  + offsetd;

    const int i3 = get_group_id(2);
    const int i2 = get_group_id(1);
    const int i1 = get_group_id(0);

    const int lid = get_local_id(0);
    const int lsize = get_local_size(0);

#if defined(cl_khr_subgroups) && (__OPENCL_VERSION__ >= 300 || !defined(GGML_OPENCL_USE_ADRENO_KERNELS))
    const uint sg_size = get_sub_group_size();
    const uint sg_id = get_sub_group_id();
    const uint sg_lid = get_sub_group_local_id();
#endif

    __local float lmem[1024];

    if (i3 >= ne03 || i2 >= ne02 || i1 >= ne01) {
        return;
    }

#if defined(cl_khr_subgroups) && (__OPENCL_VERSION__ >= 300 || !defined(GGML_OPENCL_USE_ADRENO_KERNELS))
    if(sg_id == 0){
        lmem[sg_lid] = 0.0f;
    }
#else
    if (lid < 1024) {
        lmem[lid] = 0.0f;
    }
#endif

    global float * src_row = (global float *) (src0 + i1*nb01 + i2*nb02 + i3*nb03);
    global float * dst_row = (global float *) (dst  + i1*nb1  + i2*nb2  + i3*nb3);

    float sumf = 0.0f;

    for (int i0 = lid; i0 < ne00; i0 += lsize) {
        sumf += src_row[i0];
    }

#if defined(cl_khr_subgroups) && (__OPENCL_VERSION__ >= 300 || !defined(GGML_OPENCL_USE_ADRENO_KERNELS))
    sumf = sub_group_reduce_add(sumf);

    barrier(CLK_LOCAL_MEM_FENCE);

    if(sg_lid == 0){
        lmem[sg_id] = sumf;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    sumf = lmem[sg_lid];
    sumf = sub_group_reduce_add(sumf);
#else
    // Fallback to local memory reduction for Adreno 6xx or OpenCL < 3.0 to avoid compiler crash
    lmem[lid] = sumf;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int i = lsize / 2; i > 0; i /= 2) {
        if (lid < i) {
            lmem[lid] += lmem[lid + i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    sumf = lmem[0];
#endif

    if (lid == 0) {
        dst_row[0] = sumf;
    }
}

kernel void kernel_sum_rows_f32_4(
    global char *  src0,
    ulong           offset0,
    global char *  dst,
    ulong           offsetd,
    int             ne00,
    int             ne01,
    int             ne02,
    int             ne03,
    ulong           nb01,
    ulong           nb02,
    ulong           nb03,
    ulong           nb1,
    ulong           nb2,
    ulong           nb3
) {
    src0 = src0 + offset0;
    dst  = dst  + offsetd;

    const int i3 = get_group_id(2);
    const int i2 = get_group_id(1);
    const int i1 = get_group_id(0);

    const int lid = get_local_id(0);
    const int lsize = get_local_size(0);

#if defined(cl_khr_subgroups) && (__OPENCL_VERSION__ >= 300 || !defined(GGML_OPENCL_USE_ADRENO_KERNELS))
    const uint sg_size = get_sub_group_size();
    const uint sg_id = get_sub_group_id();
    const uint sg_lid = get_sub_group_local_id();
#endif

    __local float lmem[1024];

    if (i3 >= ne03 || i2 >= ne02 || i1 >= ne01) {
        return;
    }

#if defined(cl_khr_subgroups) && (__OPENCL_VERSION__ >= 300 || !defined(GGML_OPENCL_USE_ADRENO_KERNELS))
    if(sg_id == 0){
        lmem[sg_lid] = 0.0f;
    }
#else
    if (lid < 1024) {
        lmem[lid] = 0.0f;
    }
#endif

    global float4 * src_row = (global float4 *) (src0 + i1*nb01 + i2*nb02 + i3*nb03);
    global float  * dst_row = (global float  *) (dst  + i1*nb1  + i2*nb2  + i3*nb3);

    float4 sum_vec = (float4)0.0f;

    for (int i0 = lid; i0 < ne00 / 4; i0 += lsize) {
        sum_vec += src_row[i0];
    }

    float sumf = dot(sum_vec, (float4)(1.0f));

#if defined(cl_khr_subgroups) && (__OPENCL_VERSION__ >= 300 || !defined(GGML_OPENCL_USE_ADRENO_KERNELS))
    sumf = sub_group_reduce_add(sumf);

    barrier(CLK_LOCAL_MEM_FENCE);

    if(sg_lid == 0){
        lmem[sg_id] = sumf;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    sumf = lmem[sg_lid];
    sumf = sub_group_reduce_add(sumf);
#else
    // Fallback to local memory reduction for Adreno 6xx or OpenCL < 3.0 to avoid compiler crash
    lmem[lid] = sumf;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int i = lsize / 2; i > 0; i /= 2) {
        if (lid < i) {
            lmem[lid] += lmem[lid + i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    sumf = lmem[0];
#endif

    if (lid == 0) {
        dst_row[0] = sumf;
    }
}

