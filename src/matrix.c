#define DTYPE float

__global__
void add_k(const DTYPE *a, const DTYPE *b, DTYPE *out, size_t n)
{
  size_t i = threadIdx.x;
  if (i < n)
    out[i] = a[i] + b[i];
}

__host__
void add_cpu(const DTYPE *a, const DTYPE *b, DTYPE *out, size_t n)
{
  for (size_t i = 0; i < n; i++)
    out[i] = a[i] + b[i];
}

__host__
void add_gpu(const DTYPE *a, const DTYPE *b, DTYPE *out, size_t n)
{
  const int size = n * sizeof(DTYPE);
  DTYPE *aG, *bG, *outG;
  cudaMalloc((void **) &aG, size);
  cudaMalloc((void **) &bG, size);
  cudaMalloc((void **) &outG, size);
  cudaMemcpy(aG, a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(bG, b, size, cudaMemcpyHostToDevice);
  cudaMemcpy(outG, out, size, cudaMemcpyHostToDevice);

  // Timer start
  add_k<<<n, 1>>>(aG, bG, outG, n);
  // Timer end

  cudaMemcpy(out, outG, size, cudaMemcpyDeviceToHost);
}

int main()
{
  return 0;
}
