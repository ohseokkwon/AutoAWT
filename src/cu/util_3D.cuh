template <typename T>
__device__ T queryTexBuffer(T* buffer, int w, int h, int tx, int ty, int d = 0, int tz = 0) {
	T result = 0;

	if (tz == 0 || d == 0) {
		if ((0 <= tx && tx < w) && (0 <= ty && ty < h))
			result = buffer[ty*w + tx];
	}
	else {
		if ((0 <= tx && tx < w) && (0 <= ty && ty < h) && (0 <= tz && tz < d))
			result = buffer[tz*w*h + ty*w + tx];
	}
	return result;
}

__global__ void calculate2DHistogram(Uint16* d_HU2D, int* d_histogram, const int w, const int h, int d, int tz)
{
	const int tid = threadIdx.x + blockIdx.x*blockDim.x;
	int tx = threadIdx.x;
	int ty = blockIdx.x;
	{
		int pixelValue = queryTexBuffer(d_HU2D, w, h, tx, ty, d, tz);
		atomicAdd(&d_histogram[pixelValue], 1);
	}
}

__global__ void calculate3DHistogram(uint3 windowSize, Uint16* d_buffer, int* d_histogram)
{
	uint32 tx = blockIdx.x * blockDim.x + threadIdx.x;
	uint32 ty = blockIdx.y * blockDim.y + threadIdx.y;
	uint32 tz = blockIdx.z * blockDim.z + threadIdx.z;
	if (tx >= windowSize.x || ty >= windowSize.y || tz >= windowSize.z)
		return;

	uint32 idx = tz * (windowSize.x*windowSize.y) + ty * windowSize.x + tx;

	int pixelValue = queryTexBuffer(d_buffer, windowSize.x, windowSize.y, tx, ty, windowSize.z, tz);
	atomicAdd(&d_histogram[pixelValue], 1);
}


__global__ void median_3Dvoxels(Uint16* in_buffer, Uint16* out_buffer, uint3 windowSize, const uint kernelSize)
{
	uint32 tx = blockIdx.x * blockDim.x + threadIdx.x;
	uint32 ty = blockIdx.y * blockDim.y + threadIdx.y;
	uint32 tz = blockIdx.z * blockDim.z + threadIdx.z;
	if (tx >= windowSize.x || ty >= windowSize.y || tz >= windowSize.z)
		return;

	const uint32 idx = tz * (windowSize.x*windowSize.y) + ty * windowSize.x + tx;
	uint3 search = make_uint3(tx, ty, tz);
	const int naiveSize = 5;
	float patch[naiveSize*naiveSize*naiveSize] = { 0.0f };
	for (int dk = 0; dk < kernelSize; dk++) {
		for (int hk = 0; hk < kernelSize; hk++) {
			for (int wk = 0; wk < kernelSize; wk++) {
				search = make_uint3((int)tx + (wk - int((kernelSize - 1) / 2)),
					(int)ty + (hk - int((kernelSize - 1) / 2)),
					(int)tz + (dk - int((kernelSize - 1) / 2)));
				if ((search.x >= windowSize.x || search.y >= windowSize.y || search.z >= windowSize.z) ||
					(search.x < 0 || search.y < 0 || search.z < 0))
					continue;
				uint search_idx = search.z * (windowSize.x*windowSize.y) + search.y * windowSize.x + search.x;

				patch[dk*kernelSize*kernelSize + hk * kernelSize + wk] = in_buffer[search_idx];
			}
		}
	}
	int minIdx = 0;
	for (int i = 0; i < kernelSize*kernelSize*kernelSize; i++) {
		for (int j = i + 1; j < kernelSize*kernelSize*kernelSize; j++) {
			if (patch[i] > patch[j]) {
				float tmp = patch[i];
				patch[i] = patch[j];
				patch[j] = tmp;
			}
		}
	}

	float median_value = patch[int((kernelSize*kernelSize*kernelSize - 1) / 2)];
	out_buffer[idx] = (Uint16)median_value;
}

__global__ void normalize_kernel(uint3 windowSize, Uint16* in, uint16* out, int* d_threshold, int k)
{
	uint32 tx = blockIdx.x * blockDim.x + threadIdx.x;
	uint32 ty = blockIdx.y * blockDim.y + threadIdx.y;
	uint32 tz = blockIdx.z * blockDim.z + threadIdx.z;
	if (tx >= windowSize.x || ty >= windowSize.y || tz >= windowSize.z)
		return;

	uint32 idx = tz * (windowSize.x*windowSize.y) + ty * windowSize.x + tx;

	int min = 0;
	int max = 0;

	if (k < 0) {
		max = d_threshold[k + 1];
	}
	else if (k >= 2) {
		min = d_threshold[k];
	}
	else {
		min = d_threshold[k];
		max = d_threshold[k + 1];
	}

	/*min = 0;
	max = 3500;*/
	float value = ((float)(in[idx] - min) / (float)(max - min));
	// float to uint16 변환함수 필요함
	out[idx] = uint16(value * 255);
}

__global__ void fetch_in_voxel_kernel(uint3 windowSize, uint16* input, uint16* output, int depth, int axis = 0)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;

	int tx = threadIdx.x;
	int ty = blockIdx.x;
	{
		int v_tid = tid + depth *windowSize.x*windowSize.y;

		output[tid] = input[v_tid];
	}
}