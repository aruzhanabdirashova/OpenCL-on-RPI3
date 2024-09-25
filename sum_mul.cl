__kernel void sum_mul(
    __global const float16* a_g, __global const float16* b_g, 
    __global float16* res_add, __global float16* res_mul)
{
  int gid = get_global_id(0);
  res_add[gid] = a_g[gid] + b_g[gid];
  res_mul[gid] = a_g[gid] * b_g[gid];
}
