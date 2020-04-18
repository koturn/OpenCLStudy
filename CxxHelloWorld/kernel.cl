__kernel void
hello(void)
{
  printf(
      "Hello World! from OpenCL. gid[i] = %d, gid[1] = %d, gid[2] = %d\n",
      get_global_id(0),
      get_global_id(1),
      get_global_id(2));
}
