#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

__kernel void atomic_sum(__global const int* arr, __global ulong* sum){
    local ulong tmpSum[1];
    if(get_local_id(0)==0){
        tmpSum[0]=0;
    }
    int i = get_global_id(0);
    barrier(CLK_LOCAL_MEM_FENCE);
    atomic_add(&tmpSum[0],(ulong)arr[i]);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(get_local_id(0)==(get_local_size(0)-1)){
        atomic_add(sum, tmpSum[0]);
    }
}
