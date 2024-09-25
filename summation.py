from __future__ import absolute_import, print_function
import numpy as np
import pyopencl as cl
import time

vector_size = 1000000
vector_type = np.float32

print("Generate {} random numbers of type {}".format(vector_size, 
vector_type))
a_np = np.random.rand(vector_size).astype(vector_type)
b_np = np.random.rand(vector_size).astype(vector_type)
res_np = np.empty_like(a_np)

print("Create OpenCL contect and queue")
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

print("Create buffers")
mf = cl.mem_flags
a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_np)
b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b_np)
res_g = cl.Buffer(ctx, mf.WRITE_ONLY, a_np.nbytes)

print("Reading kernel file")
with open("sum.cl", "r") as f_kernel:
    kernel = f_kernel.read()

print("Compiling kernel")
prg = cl.Program(ctx, kernel).build()

print("Executing computation")
#prg.sum(queue, a_np.shape, None, a_g, b_g, res_g)
sum_knl = prg.sum
sum_knl.set_args(a_g, b_g, res_g)
local_work_size = (8,)
#local_work_size = (10,)
global_work_size = (vector_size//16,)
adjusted_global_work_size = (int(np.ceil(global_work_size[0] / local_work_size[0]))*local_work_size[0],)

t0 = time.process_time_ns()
ev = cl.enqueue_nd_range_kernel(queue=queue, kernel=sum_knl, 
global_work_size=adjusted_global_work_size, local_work_size=local_work_size)
ev.wait()
t1 = time.process_time_ns()

cl.enqueue_copy(queue, res_np, res_g)
t2 = time.process_time_ns()

# Check on CPU with Numpy:
print("Computing on the host using numpy")
t3 = time.process_time_ns()
res_local = a_np + b_np
t4 = time.process_time_ns()

print("Comparing results")
print("Difference :{}".format(res_np - res_local))
print(a_np[0:5])
print(b_np[0:5])
print(res_np[0:5])
print(res_local[0:5])

print("Checking the norm between both: {}".format(np.linalg.norm(res_np - 
res_local)))

print("Checking results are mostly the same", np.allclose(res_np, 
res_local))

print("comparing execution times")
print("openCL: {} ms".format((t1-t0)/1000000))
print("openCL copy from GPU to host: {} ms".format((t2-t1)/1000000))
print("numpy: {} ms".format((t4-t3)/1000000))

