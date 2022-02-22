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

template <typename T>
__device__ T queryTexBuffer_INTR(T* buffer, int w, int h, int tx, int ty, int d = 0, int tz = 0) {
	T result;

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

template <typename T>
__device__ T trilerp(T* buffer, float3 v, int W, int H, int D)
{
	float3 v000 = make_float3(floor(v.x), floor(v.y), floor(v.z));
	float3 v100 = make_float3(ceil(v.x), floor(v.y), floor(v.z));
	float3 v010 = make_float3(floor(v.x), ceil(v.y), floor(v.z));
	float3 v001 = make_float3(floor(v.x), floor(v.y), ceil(v.z));
	float3 v011 = make_float3(floor(v.x), ceil(v.y), ceil(v.z));
	float3 v110 = make_float3(ceil(v.x), ceil(v.y), floor(v.z));
	float3 v101 = make_float3(ceil(v.x), floor(v.y), ceil(v.z));
	float3 v111 = make_float3(ceil(v.x), ceil(v.y), ceil(v.z));

	T c000 = queryTexBuffer_INTR(buffer, W, H, v000.x, v000.y, D, v000.z);
	T c100 = queryTexBuffer_INTR(buffer, W, H, v100.x, v100.y, D, v100.z);
	T c010 = queryTexBuffer_INTR(buffer, W, H, v010.x, v010.y, D, v010.z);
	T c001 = queryTexBuffer_INTR(buffer, W, H, v001.x, v001.y, D, v001.z);
	T c011 = queryTexBuffer_INTR(buffer, W, H, v011.x, v011.y, D, v011.z);
	T c110 = queryTexBuffer_INTR(buffer, W, H, v110.x, v110.y, D, v110.z);
	T c101 = queryTexBuffer_INTR(buffer, W, H, v101.x, v101.y, D, v101.z);
	T c111 = queryTexBuffer_INTR(buffer, W, H, v111.x, v111.y, D, v111.z);

	if ((v111.x - v000.x) < 1e-6 || (v111.y - v000.y) < 1e-6 || (v111.z - v000.z) < 1e-6) {
		return queryTexBuffer_INTR(buffer, W, H, v.x, v.y, D, v.z);
	}
	float xd = (v.x - v000.x) / (v111.x - v000.x);
	float yd = (v.y - v000.y) / (v111.y - v000.y);
	float zd = (v.z - v000.z) / (v111.z - v000.z);

	T c00 = c000*(1.0f - xd) + c100*xd;
	T c01 = c001*(1.0f - xd) + c101*xd;
	T c10 = c010*(1.0f - xd) + c110*xd;
	T c11 = c011*(1.0f - xd) + c111*xd;

	T c0 = c00*(1.0f - yd) + c10*yd;
	T c1 = c01*(1.0f - yd) + c11*yd;
	
	T c = c0*(1.0f - zd) + c1*zd;
	return c;
}

// out_Mask는 이전에 0으로 초기화 하는 것을 권장합니다.
__global__ void fillupVolumebyMask_kernel(uint3 windowSize, Uint16* in_Mask, float* out_Mask, float setValue, uint16 baseValue)
{
	uint32 tx = blockIdx.x * blockDim.x + threadIdx.x;
	uint32 ty = blockIdx.y * blockDim.y + threadIdx.y;
	uint32 tz = blockIdx.z * blockDim.z + threadIdx.z;
	if (tx >= windowSize.x || ty >= windowSize.y || tz >= windowSize.z)
		return;

	uint32 idx = tz * (windowSize.x*windowSize.y) + ty * windowSize.x + tx;

	if (in_Mask[idx] == baseValue) {
		out_Mask[idx] = setValue;
	}
}

__global__ void copy_uint16_to_float_kernel(uint3 windowSize, float* out_buf, Uint16* in_buf)
{
	uint32 tx = blockIdx.x * blockDim.x + threadIdx.x;
	uint32 ty = blockIdx.y * blockDim.y + threadIdx.y;
	uint32 tz = blockIdx.z * blockDim.z + threadIdx.z;
	if (tx >= windowSize.x || ty >= windowSize.y || tz >= windowSize.z)
		return;

	uint32 idx = tz * (windowSize.x*windowSize.y) + ty * windowSize.x + tx;

	out_buf[idx] = (float)in_buf[idx];
}

template <typename T>
__global__ void inverseMask3D_kernel(uint3 windowSize, T* in_Mask)
{
	uint32 tx = blockIdx.x * blockDim.x + threadIdx.x;
	uint32 ty = blockIdx.y * blockDim.y + threadIdx.y;
	uint32 tz = blockIdx.z * blockDim.z + threadIdx.z;
	if (tx >= windowSize.x || ty >= windowSize.y || tz >= windowSize.z)
		return;

	uint32 idx = tz * (windowSize.x*windowSize.y) + ty * windowSize.x + tx;

	if (in_Mask[idx] < 1) {
		in_Mask[idx] = 1;
	}
	else {
		in_Mask[idx] = 0;
	}
}

__global__ void cutoffVolume_kernel(uint3 windowSize, float* in_vf3D, float cutoff, float setValue)
{
	uint32 tx = blockIdx.x * blockDim.x + threadIdx.x;
	uint32 ty = blockIdx.y * blockDim.y + threadIdx.y;
	uint32 tz = blockIdx.z * blockDim.z + threadIdx.z;
	if (tx >= windowSize.x || ty >= windowSize.y || tz >= windowSize.z)
		return;

	uint32 idx = tz * (windowSize.x*windowSize.y) + ty * windowSize.x + tx;

	if (in_vf3D[idx] <= cutoff) {
		in_vf3D[idx] = setValue;
	}
}


__global__ void binalized_kernel(uint3 windowSize, float* in_vf3D, float cutoff)
{
	uint32 tx = blockIdx.x * blockDim.x + threadIdx.x;
	uint32 ty = blockIdx.y * blockDim.y + threadIdx.y;
	uint32 tz = blockIdx.z * blockDim.z + threadIdx.z;
	if (tx >= windowSize.x || ty >= windowSize.y || tz >= windowSize.z)
		return;

	uint32 idx = tz * (windowSize.x*windowSize.y) + ty * windowSize.x + tx;

	if (in_vf3D[idx] > cutoff) {
		in_vf3D[idx] = 1;
	}
	else {
		in_vf3D[idx] = 0;
	}
}

__global__
void connectivityFiltering_kernel(uint3 windowSize, uint16* in_wall, uint16* in_cond, float* out_cond, float setValue)
{
	uint32 tx = blockIdx.x * blockDim.x + threadIdx.x;
	uint32 ty = blockIdx.y * blockDim.y + threadIdx.y;
	uint32 tz = blockIdx.z * blockDim.z + threadIdx.z;
	if (tx >= windowSize.x || ty >= windowSize.y || tz >= windowSize.z)
		return;

	uint32 idx = tz * (windowSize.x*windowSize.y) + ty * windowSize.x + tx;


	// 0이상이면 1개라도 연결되어있다고 가정함.
	int isConnect = 0;
	if (in_wall[idx] > 0)
	{
		for (int j = -1; j <= 1; j++) {
			for (int i = -1; i <= 1; i++) {
				for (int k = -1; k <= 1; k++) {
					//// + 형태로만 검사 시
					//if ((abs(j) - abs(i) - abs(k)) == 1)
					//	continue;
					isConnect += queryTexBuffer(in_cond, windowSize.x, windowSize.y, tx + i, ty + j, windowSize.z, tz + k);
				}
			}
		}
	}
	if (isConnect > 0) {
		out_cond[idx] = setValue;
	}
}


__global__ 
void connectivityFiltering_kernel(uint3 windowSize, uint16* in_wall, float* in_cond, float* out_cond, float setValue)
{
	uint32 tx = blockIdx.x * blockDim.x + threadIdx.x;
	uint32 ty = blockIdx.y * blockDim.y + threadIdx.y;
	uint32 tz = blockIdx.z * blockDim.z + threadIdx.z;
	if (tx >= windowSize.x || ty >= windowSize.y || tz >= windowSize.z)
		return;

	uint32 idx = tz * (windowSize.x*windowSize.y) + ty * windowSize.x + tx;


	// 0이상이면 1개라도 연결되어있다고 가정함.
	int isConnect = 0;
	if (in_wall[idx] > 0)
	{
		for (int j = -1; j <= 1; j++) {
			for (int i = -1; i <= 1; i++) {
				for (int k = -1; k <= 1; k++) {
					//// + 형태로만 검사 시
					//if ((abs(j) - abs(i) - abs(k)) == 1)
					//	continue;
					isConnect += queryTexBuffer(in_cond, windowSize.x, windowSize.y, tx + i, ty + j, windowSize.z, tz + k);
				}
			}
		}
	}
	if (isConnect > 0) {
		out_cond[idx] = setValue;
	}
}


__global__ void subtract_by_bool_kernel(uint3 windowSize, float* io_bufA, uint16* in_boolBuf)
{
	uint32 tx = blockIdx.x * blockDim.x + threadIdx.x;
	uint32 ty = blockIdx.y * blockDim.y + threadIdx.y;
	uint32 tz = blockIdx.z * blockDim.z + threadIdx.z;
	if (tx >= windowSize.x || ty >= windowSize.y || tz >= windowSize.z)
		return;

	uint32 idx = tz * (windowSize.x*windowSize.y) + ty * windowSize.x + tx;
	
	if (in_boolBuf[idx] > 0) {
		io_bufA[idx] = 0;
	}
}



template <typename T>
__global__ void normalize_floatbuf_kernel(uint3 windowSize, T* in, uint16* out, float* min, float* max)
{
	uint32 tx = blockIdx.x * blockDim.x + threadIdx.x;
	uint32 ty = blockIdx.y * blockDim.y + threadIdx.y;
	uint32 tz = blockIdx.z * blockDim.z + threadIdx.z;
	if (tx >= windowSize.x || ty >= windowSize.y || tz >= windowSize.z)
		return;

	uint32 idx = tz * (windowSize.x*windowSize.y) + ty * windowSize.x + tx;
	float value = ((float)(in[idx] - (uint16)min) / (float)(10.0f - -0.5f));
	out[idx] = T(value * 255);
}




__device__ float fatomicMax(float *addr, float value)
{
	int* address_as_i = (int*)addr;
	int old = *address_as_i, assumed;
	do {
		assumed = old;
		old = ::atomicCAS(address_as_i, assumed,
			__float_as_int(::fmaxf(value, __int_as_float(assumed))));
	} while (assumed != old);
	return __int_as_float(old);
}
__device__ float fatomicMin(float *addr, float value)
{
	float old = *addr, assumed;
	if (old <= value) return old;
	do
	{
		assumed = old;
		old = atomicCAS((unsigned int*)addr, __float_as_int(assumed), __float_as_int(value));
	} while (old != assumed);
	return old;
}

__global__ void reduce_fMinMax(uint3 windowSize, float* in_vf3D, float* min, float* max) 
{
	uint32 tx = blockIdx.x * blockDim.x + threadIdx.x;
	uint32 ty = blockIdx.y * blockDim.y + threadIdx.y;
	uint32 tz = blockIdx.z * blockDim.z + threadIdx.z;
	if (tx >= windowSize.x || ty >= windowSize.y || tz >= windowSize.z)
		return;

	uint32 idx = tz * (windowSize.x*windowSize.y) + ty * windowSize.x + tx;

	fatomicMin(&min[0], (float)in_vf3D[idx]);
	fatomicMax(&max[0], (float)in_vf3D[idx]);
}

__global__ void reduce_MinMax(uint3 windowSize, uint32* in_buffer, uint32* min, uint32* max, bool cond)
{
	uint32 tx = blockIdx.x * blockDim.x + threadIdx.x;
	uint32 ty = blockIdx.y * blockDim.y + threadIdx.y;
	uint32 tz = blockIdx.z * blockDim.z + threadIdx.z;
	if (tx >= windowSize.x || ty >= windowSize.y || tz >= windowSize.z)
		return;

	uint32 idx = tz * (windowSize.x*windowSize.y) + ty * windowSize.x + tx;

	uint32 data = in_buffer[idx];

	// 0번째 히스토그램은 제외합니다.
	if (idx < 1)
		return;

	atomicMin(&min[0], data);
	atomicMax(&max[0], data);
}

__global__
void cufloat_memset_by_value_kernel(uint3 windowSize, float* in_buffer, float setValue)
{
	uint32 tx = blockIdx.x * blockDim.x + threadIdx.x;
	uint32 ty = blockIdx.y * blockDim.y + threadIdx.y;
	uint32 tz = blockIdx.z * blockDim.z + threadIdx.z;
	if (tx >= windowSize.x || ty >= windowSize.y || tz >= windowSize.z)
		return;

	uint32 idx = tz * (windowSize.x*windowSize.y) + ty * windowSize.x + tx;

	in_buffer[idx] = setValue;
}

__global__ 
void cufloat4_memset_by_value_kernel(uint3 windowSize, float4* in_VF4D, int offset, float setValue)
{
	uint32 tx = blockIdx.x * blockDim.x + threadIdx.x;
	uint32 ty = blockIdx.y * blockDim.y + threadIdx.y;
	uint32 tz = blockIdx.z * blockDim.z + threadIdx.z;
	if (tx >= windowSize.x || ty >= windowSize.y || tz >= windowSize.z)
		return;

	uint32 idx = tz * (windowSize.x*windowSize.y) + ty * windowSize.x + tx;

	if (offset == 0)
		in_VF4D[idx].x = setValue;
	else if (offset == 1)
		in_VF4D[idx].y = setValue;
	else if (offset == 2)
		in_VF4D[idx].z = setValue;
	else if (offset == 3)
		in_VF4D[idx].w = setValue;
}


__global__ void computeLaplaceEquation_kernel(uint3 windowSize, float* in_vf3D, float* out_vf3D, uint16* limitCond, float* Epsilon)
{
	uint32 tx = blockIdx.x * blockDim.x + threadIdx.x;
	uint32 ty = blockIdx.y * blockDim.y + threadIdx.y;
	uint32 tz = blockIdx.z * blockDim.z + threadIdx.z;
	if (tx >= windowSize.x || ty >= windowSize.y || tz >= windowSize.z)
		return;

	uint32 idx = tz * (windowSize.x*windowSize.y) + ty * windowSize.x + tx;
	float oosix = 1.0f / 6.0f;
	float voxel = in_vf3D[idx];

	float next_voxel = (
		queryTexBuffer(in_vf3D, windowSize.x, windowSize.y, tx + 1, ty + 0, windowSize.z, tz + 0) +
		queryTexBuffer(in_vf3D, windowSize.x, windowSize.y, tx - 1, ty - 0, windowSize.z, tz - 0) +
		queryTexBuffer(in_vf3D, windowSize.x, windowSize.y, tx + 0, ty + 1, windowSize.z, tz + 0) +
		queryTexBuffer(in_vf3D, windowSize.x, windowSize.y, tx - 0, ty - 1, windowSize.z, tz - 0) +
		queryTexBuffer(in_vf3D, windowSize.x, windowSize.y, tx + 0, ty + 0, windowSize.z, tz + 1) +
		queryTexBuffer(in_vf3D, windowSize.x, windowSize.y, tx - 0, ty - 0, windowSize.z, tz - 1)) * oosix;

	// inner energy conditions = 0, but wall is the 1
	out_vf3D[idx] = next_voxel * limitCond[idx];

	float3 dxyz = make_float3(
		(queryTexBuffer(in_vf3D, windowSize.x, windowSize.y, tx + 1, ty + 0, windowSize.z, tz + 0) - queryTexBuffer(in_vf3D, windowSize.x, windowSize.y, tx - 1, ty - 0, windowSize.z, tz - 0)) / 2.0f,
		(queryTexBuffer(in_vf3D, windowSize.x, windowSize.y, tx + 0, ty + 1, windowSize.z, tz + 0) - queryTexBuffer(in_vf3D, windowSize.x, windowSize.y, tx - 0, ty - 1, windowSize.z, tz - 0)) / 2.0f,
		(queryTexBuffer(in_vf3D, windowSize.x, windowSize.y, tx + 0, ty + 0, windowSize.z, tz + 1) - queryTexBuffer(in_vf3D, windowSize.x, windowSize.y, tx - 0, ty - 0, windowSize.z, tz - 1)) / 2.0f);

	if (length(dxyz) > 0) {
		atomicAdd(&Epsilon[0], length(dxyz));
	}
}

__global__ void computeLaplaceEquation_with_Vector_kernel(uint3 windowSize, float* in_vectorfields3D, float* out, float4* G, float* Epsilon, uint16* mask)
{
	uint32 tx = blockIdx.x * blockDim.x + threadIdx.x;
	uint32 ty = blockIdx.y * blockDim.y + threadIdx.y;
	uint32 tz = blockIdx.z * blockDim.z + threadIdx.z;
	if (tx >= windowSize.x || ty >= windowSize.y || tz >= windowSize.z)
		return;

	uint32 idx = tz * (windowSize.x*windowSize.y) + ty * windowSize.x + tx;
	float oosix = 1.0f / 6.0f;
	float voxel = in_vectorfields3D[idx];

	float next_voxel = (
		queryTexBuffer(in_vectorfields3D, windowSize.x, windowSize.y, tx + 1, ty + 0, windowSize.z, tz + 0) +
		queryTexBuffer(in_vectorfields3D, windowSize.x, windowSize.y, tx - 1, ty - 0, windowSize.z, tz - 0) +
		queryTexBuffer(in_vectorfields3D, windowSize.x, windowSize.y, tx + 0, ty + 1, windowSize.z, tz + 0) +
		queryTexBuffer(in_vectorfields3D, windowSize.x, windowSize.y, tx - 0, ty - 1, windowSize.z, tz - 0) +
		queryTexBuffer(in_vectorfields3D, windowSize.x, windowSize.y, tx + 0, ty + 0, windowSize.z, tz + 1) +
		queryTexBuffer(in_vectorfields3D, windowSize.x, windowSize.y, tx - 0, ty - 0, windowSize.z, tz - 1)) * oosix;

	out[idx] = next_voxel;

	float3 dxyz = make_float3(
		(queryTexBuffer(in_vectorfields3D, windowSize.x, windowSize.y, tx + 1, ty + 0, windowSize.z, tz + 0) - queryTexBuffer(in_vectorfields3D, windowSize.x, windowSize.y, tx - 1, ty - 0, windowSize.z, tz - 0)) / 2.0f,
		(queryTexBuffer(in_vectorfields3D, windowSize.x, windowSize.y, tx + 0, ty + 1, windowSize.z, tz + 0) - queryTexBuffer(in_vectorfields3D, windowSize.x, windowSize.y, tx - 0, ty - 1, windowSize.z, tz - 0)) / 2.0f,
		(queryTexBuffer(in_vectorfields3D, windowSize.x, windowSize.y, tx + 0, ty + 0, windowSize.z, tz + 1) - queryTexBuffer(in_vectorfields3D, windowSize.x, windowSize.y, tx - 0, ty - 0, windowSize.z, tz - 1)) / 2.0f);

	if (length(dxyz) > 0) {
		G[idx] = make_float4(normalize(dxyz), G[idx].w);// / length(dxyz);
		atomicAdd(&Epsilon[0], length(dxyz));
	}
}

//__global__ void computeLaplaceEquation_with_Vector_kernel(uint3 windowSize, float* in_vectorfields3D, float* out, float4* G, float* Epsilon, uint16* mask)
//{
//	uint32 tx = blockIdx.x * blockDim.x + threadIdx.x;
//	uint32 ty = blockIdx.y * blockDim.y + threadIdx.y;
//	uint32 tz = blockIdx.z * blockDim.z + threadIdx.z;
//	if (tx >= windowSize.x || ty >= windowSize.y || tz >= windowSize.z)
//		return;
//
//	uint32 idx = tz * (windowSize.x*windowSize.y) + ty * windowSize.x + tx;
//	float oosix = 1.0f / 6.0f;
//	float voxel = in_vectorfields3D[idx];
//
//	float convKernel[3*3*3] = { 0 };
//	float elemSum = 0.0f;
//	for (int i = -1; i < 2; i++) {
//		for (int j = -1; j < 2; j++) {
//			for (int k = -1; k < 2; k++) {
//				if (i == 0 && j == 0 && k == 0)
//					continue;
//
//				float vv = queryTexBuffer(in_vectorfields3D, windowSize.x, windowSize.y, tx + i, ty + j, windowSize.z, tz + k);
//				if (vv <= 0.0f)
//					vv = voxel;
//				convKernel[(k + 1) * 3 * 3 + (j + 1) * 3 + (i + 1)] = vv;
//				elemSum += vv;
//			}
//		}
//	}
//
//	float v0, v1, v2, v3, v4, v5, v6;
//	v0 = voxel;// queryTexBuffer(in_vectorfields3D, windowSize.x, windowSize.y, tx + 0, ty + 0, windowSize.z, tz + 0);
//	v1 = queryTexBuffer(in_vectorfields3D, windowSize.x, windowSize.y, tx + 1, ty + 0, windowSize.z, tz + 0);
//	v2 = queryTexBuffer(in_vectorfields3D, windowSize.x, windowSize.y, tx - 1, ty - 0, windowSize.z, tz - 0);
//	v3 = queryTexBuffer(in_vectorfields3D, windowSize.x, windowSize.y, tx + 0, ty + 1, windowSize.z, tz + 0);
//	v4 = queryTexBuffer(in_vectorfields3D, windowSize.x, windowSize.y, tx - 0, ty - 1, windowSize.z, tz - 0);
//	v5 = queryTexBuffer(in_vectorfields3D, windowSize.x, windowSize.y, tx + 0, ty + 0, windowSize.z, tz + 1);
//	v6 = queryTexBuffer(in_vectorfields3D, windowSize.x, windowSize.y, tx - 0, ty - 0, windowSize.z, tz - 1);
//
//	if (v1 <= 0.0f)
//		v1 = v0;
//	if (v2 <= 0.0f)
//		v2 = v0;
//	if (v3 <= 0.0f)
//		v3 = v0;
//	if (v4 <= 0.0f)
//		v4 = v0;
//	if (v5 <= 0.0f)
//		v5 = v0;
//	if (v6 <= 0.0f)
//		v6 = v0;
//
//	float next_voxel = (v1 + v2 + v3 + v4 + v5 + v6) * oosix;
//
//	out[idx] = 1.0f/(3*3*3-1) * elemSum * (float)mask[idx];
//
//	float3 dxyz = make_float3((v1 - v2) / 2.0f, 
//		(v3 - v4) / 2.0f, 
//		(v5 - v6) / 2.0f);
//
//		/*(queryTexBuffer(in_vectorfields3D, windowSize.x, windowSize.y, tx + 1, ty + 0, windowSize.z, tz + 0) - 
//			queryTexBuffer(in_vectorfields3D, windowSize.x, windowSize.y, tx - 1, ty - 0, windowSize.z, tz - 0)) / 2.0f,
//		(queryTexBuffer(in_vectorfields3D, windowSize.x, windowSize.y, tx + 0, ty + 1, windowSize.z, tz + 0) - 
//			queryTexBuffer(in_vectorfields3D, windowSize.x, windowSize.y, tx - 0, ty - 1, windowSize.z, tz - 0)) / 2.0f,
//		(queryTexBuffer(in_vectorfields3D, windowSize.x, windowSize.y, tx + 0, ty + 0, windowSize.z, tz + 1) - 
//			queryTexBuffer(in_vectorfields3D, windowSize.x, windowSize.y, tx - 0, ty - 0, windowSize.z, tz - 1)) / 2.0f);*/
//
//	if (length(dxyz) > 0) {
//		G[idx] = make_float4(normalize(dxyz), G[idx].w) * (float)mask[idx];// / length(dxyz);
//		atomicAdd(&Epsilon[0], length(dxyz));
//	}
//}

__global__ void compute_thickness_kernel(uint3 windowSize, uint16* in_mask, float4* in_vectorfields, uint mode, float4* in_Vertices, float3* out_normal, int totalSize, float3 voxel_size)
{
	//uint32 tx = blockIdx.x * blockDim.x + threadIdx.x;
	//uint32 ty = blockIdx.y * blockDim.y + threadIdx.y;
	//uint32 tz = blockIdx.z * blockDim.z + threadIdx.z;
	//if (tx >= windowSize.x || ty >= windowSize.y || tz >= windowSize.z)
	//	return;
	const int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= totalSize)
		return;

	//uint32 idx = tz * (windowSize.x*windowSize.y) + ty * windowSize.x + tx;

	const int MAX_TRAVEL = 256;
	float thickness = 0.0f;

	float3 edgePoint = make_float3(in_Vertices[idx]);
	uint3 surfacePT = make_uint3((edgePoint.x + 1.0f) * 0.5f * windowSize.x,
		windowSize.y - ((edgePoint.y + 1.0f) * 0.5f * windowSize.y), edgePoint.z);

	surfacePT = make_uint3(edgePoint.x, edgePoint.y, edgePoint.z);

	if (surfacePT.x >= windowSize.x || surfacePT.y >= windowSize.y || surfacePT.z >= windowSize.z) {
		in_Vertices[idx].w = 0;
		return;
	}
	uint32 search_idx = surfacePT.z * (windowSize.x*windowSize.y) + surfacePT.y * windowSize.x + surfacePT.x;

	float3 normal = make_float3(in_vectorfields[search_idx]);
	normal = normalize(make_float3(normal.x, -normal.y, normal.z)) * (mode ? -1.0f : 1.0f);

	uint3 searchPT = surfacePT;
	float3 integrate_Path = make_float3(surfacePT.x + 0.5f, surfacePT.y + 0.5f, surfacePT.z + 0.5f);
	float3 dir = make_float3(0.0f);
	int state = true;
	int integrate_cnt = 0;
	float dt = 0.001f;
	for (int i = 0; i < MAX_TRAVEL * (1 / dt); i++) {
		float4 voxElem = in_vectorfields[search_idx];
		//float3 signedVe = make_float3(ve.x > 0 ? 1 : -1, ve.y > 0 ? 1 : -1, ve.z > 0 ? 1 : -1);
		if (voxElem.w > 0) {
			state = false;
			if (dot(dir, make_float3(voxElem)) >= 0) {
				dir = make_float3(voxElem) * (mode ? -1.0f : 1.0f);
				if (length(dir) > 0) {
					float3 prev_pos = integrate_Path;
					integrate_Path += dir * dt;
					//ve = make_float3(ve.x > 0.5f ? 1 : 0, ve.y > 0.5f ? 1 : 0, ve.z > 0.5f ? 1 : 0) * signedVe;
					searchPT = make_uint3(integrate_Path.x, integrate_Path.y, integrate_Path.z);// make_uint3(surfacePT.x + ve.x, surfacePT.y + ve.y, surfacePT.z + ve.z);
					if (searchPT.x >= windowSize.x || searchPT.y >= windowSize.y || searchPT.z >= windowSize.z)
						break;
					//thickness += length(dir * voxel_size) * (float)(in_mask[search_idx] > 0 ? 1.0f : 0.0f);
					thickness += length((prev_pos - integrate_Path) * voxel_size) * (float)(in_mask[search_idx] > 0 ? 1.0f : 0.0f);
					search_idx = searchPT.z * (windowSize.x*windowSize.y) + searchPT.y * windowSize.x + searchPT.x;
				}
				else {
					break;
				}
			}
			else {
				if (i == 0) {
					dir = make_float3(voxElem) * (mode ? -1.0f : 1.0f);
					integrate_Path += dir * dt;
					searchPT = make_uint3(integrate_Path.x, integrate_Path.y, integrate_Path.z);
					if (searchPT.x >= windowSize.x || searchPT.y >= windowSize.y || searchPT.z >= windowSize.z)
						break;
					search_idx = searchPT.z * (windowSize.x*windowSize.y) + searchPT.y * windowSize.x + searchPT.x;
					continue;
				}
				break;
			}
		}
		else {
			if (!state)
				break;
			else {
				dir = make_float3(voxElem) * (mode ? -1.0f : 1.0f);
				integrate_Path += dir * dt;
				searchPT = make_uint3(integrate_Path.x, integrate_Path.y, integrate_Path.z);
				if (searchPT.x >= windowSize.x || searchPT.y >= windowSize.y || searchPT.z >= windowSize.z)
					break;
				search_idx = searchPT.z * (windowSize.x*windowSize.y) + searchPT.y * windowSize.x + searchPT.x;
			}
		}
	}

	out_normal[idx] = normal * 1.0 / make_float3(windowSize);// (normal*(thickness < 1e-5 ? 1 : thickness)) / make_float3(windowSize);
	in_Vertices[idx].w = thickness;
}


__global__ void dilation_kernel(uint3 windowSize, Uint16* d_buffer, Uint16* output, const uint kernelSize)
{
	uint32 tx = blockIdx.x * blockDim.x + threadIdx.x;
	uint32 ty = blockIdx.y * blockDim.y + threadIdx.y;
	uint32 tz = blockIdx.z * blockDim.z + threadIdx.z;
	if (tx >= windowSize.x || ty >= windowSize.y || tz >= windowSize.z)
		return;

	uint32 idx = tz * (windowSize.x*windowSize.y) + ty * windowSize.x + tx;
	
	
	const uint naiveSize = 3;
	float patch[naiveSize*naiveSize*naiveSize] = { 0.0f };
	for (int hk = 0; hk < kernelSize; hk++) {
		for (int wk = 0; wk < kernelSize; wk++) {
			for (int dk = 0; dk < kernelSize; dk++) {
				// dilation
				/*if (KERNEL_SIZE > 3 && hk % 2 == 1 || wk % 2 == 1)
				continue;*/
				patch[dk*kernelSize*kernelSize + hk * kernelSize + wk] = queryTexBuffer(d_buffer, windowSize.x, windowSize.y, 
					tx + (wk - int((kernelSize - 1) / 2)), 
					ty + (hk - int((kernelSize - 1) / 2)), windowSize.z,
					tz + (dk - int((kernelSize - 1) / 2)));
				
			}
		}
	}

	int maxVal = 0;
	for (int i = 0; i < kernelSize * kernelSize * kernelSize; i++) {
		if (maxVal < patch[i])
			maxVal = patch[i];
	}

	output[idx] = (Uint16)maxVal;
}

__global__ void erode_kernel(uint3 windowSize, Uint16* d_buffer, Uint16* output, const uint kernelSize)
{
	uint32 tx = blockIdx.x * blockDim.x + threadIdx.x;
	uint32 ty = blockIdx.y * blockDim.y + threadIdx.y;
	uint32 tz = blockIdx.z * blockDim.z + threadIdx.z;
	if (tx >= windowSize.x || ty >= windowSize.y || tz >= windowSize.z)
		return;

	uint32 idx = tz * (windowSize.x*windowSize.y) + ty * windowSize.x + tx;


	const uint naiveSize = 3;
	float patch[naiveSize*naiveSize*naiveSize] = { 0.0f };
	for (int hk = 0; hk < kernelSize; hk++) {
		for (int wk = 0; wk < kernelSize; wk++) {
			for (int dk = 0; dk < kernelSize; dk++) {
				// dilation
				/*if (KERNEL_SIZE > 3 && hk % 2 == 1 || wk % 2 == 1)
				continue;*/
				patch[dk*kernelSize*kernelSize + hk * kernelSize + wk] = queryTexBuffer(d_buffer, windowSize.x, windowSize.y,
					tx + (wk - int((kernelSize - 1) / 2)),
					ty + (hk - int((kernelSize - 1) / 2)), windowSize.z,
					tz + (dk - int((kernelSize - 1) / 2)));

			}
		}
	}

	int minVal = 0xffff;
	for (int i = 0; i < kernelSize * kernelSize; i++) {
		if (minVal > patch[i])
			minVal = patch[i];
	}

	output[idx] = (Uint16)minVal;
}

//
//__global__ void compute_thickness_kernel(uint3 windowSize, uint16* in_mask, float4* in_vectorfields, uint mode, float4* in_Vertices, float3* out_normal, int totalSize, float3 voxel_size)
//{
//	const int idx = threadIdx.x + blockIdx.x * blockDim.x;
//	if (idx >= totalSize)
//		return;
//
//	const int MAX_TRAVEL = 256;
//	float thickness = 0.0f;
//
//	float3 startPT = make_float3(in_Vertices[idx]);
//	float3 v0 = startPT;
//	float dt = 0.01f;
//	for (int i = 0; i < MAX_TRAVEL * (1 / dt); i++) {
//		float3 dir = make_float3(trilerp(in_vectorfields, v0, windowSize.x, windowSize.y, windowSize.z));
//		if (length(dir) > 0) {
//			float3 v1 = v0 + dir * dt;
//			thickness += length((v0 - v1) * voxel_size)*(float)(trilerp(in_mask, v1, windowSize.x, windowSize.y, windowSize.z) > 0 ? 1.0f : 0.0f);
//			v0 = v1;
//		}
//		else {
//			break;
//		}
//	}
//
//	uint32 search_idx = startPT.z * (windowSize.x*windowSize.y) + startPT.y * windowSize.x + startPT.x;
//	float3 normal = make_float3(in_vectorfields[search_idx]);
//	normal = normalize(make_float3(normal.x, -normal.y, normal.z)) * (mode ? -1.0f : 1.0f);
//
//	out_normal[idx] = normal * 1.0 / make_float3(windowSize);// (normal*(thickness < 1e-5 ? 1 : thickness)) / make_float3(windowSize);
//	in_Vertices[idx].w = thickness;
//
//	
//}


// CCL funcs
__global__ void init_CCL(uint3 windowSize, uint32* L, uint32* R)
{
	uint32 tx = blockIdx.x * blockDim.x + threadIdx.x;
	uint32 ty = blockIdx.y * blockDim.y + threadIdx.y;
	uint32 tz = blockIdx.z * blockDim.z + threadIdx.z;
	if (tx >= windowSize.x || ty >= windowSize.y || tz >= windowSize.z)
		return;

	uint32 idx = tz * (windowSize.x*windowSize.y) + ty * windowSize.x + tx;

	// 512*512*D 정수는 float에 담기 어렵습니다.

	//float k = ((float)idx * 0.1f) + 1.0f;
	L[idx] = R[idx] = idx; //(1~시작)
}

__global__ void scanning(uint3 windowSize, float* D, uint32* L, uint32* R)
{
	uint32 tx = blockIdx.x * blockDim.x + threadIdx.x;
	uint32 ty = blockIdx.y * blockDim.y + threadIdx.y;
	uint32 tz = blockIdx.z * blockDim.z + threadIdx.z;
	if (tx >= windowSize.x || ty >= windowSize.y || tz >= windowSize.z)
		return;

	uint32 idx = tz * (windowSize.x*windowSize.y) + ty * windowSize.x + tx;

	float p = D[idx];
	if (p < 1.0f) {
		R[idx] = 0;
		return;
	}

	uint32 P_label = L[idx];
	if (P_label < 1)
		return;

	uint32 Q_label = P_label;

	for (int i = -1; i <= 1; i++) {
		for (int j = -1; j <= 1; j++) {
			for (int k = -1; k <= 1; k++) {
				if (i == 0 && j == 0 && k == 0)
					continue;
				// 열십자 필터만 허용..
				if (1<(abs(i) + abs(j) + abs(k)))
					continue;

				float q = queryTexBuffer(D, windowSize.x, windowSize.y, tx + i, ty + j, windowSize.z, tz + k);
				if (q < 1.0f)
					continue;

				uint32 label = queryTexBuffer(L, windowSize.x, windowSize.y, tx + i, ty + j, windowSize.z, tz + k);
				if (label < 1)
					continue;
				Q_label = min(label, Q_label);
			}
		}
	}

	R[idx] = Q_label;
}

__global__ void analysis_ccl(uint3 windowSize, uint32* d_buffer, uint32* d_hist)
{
	uint32 tx = blockIdx.x * blockDim.x + threadIdx.x;
	uint32 ty = blockIdx.y * blockDim.y + threadIdx.y;
	uint32 tz = blockIdx.z * blockDim.z + threadIdx.z;
	if (tx >= windowSize.x || ty >= windowSize.y || tz >= windowSize.z)
		return;

	uint32 idx = tz * (windowSize.x*windowSize.y) + ty * windowSize.x + tx;

	uint32 ccl_label = d_buffer[idx];// queryTexBuffer(d_buffer, windowSize.x, windowSize.y, tx, ty, windowSize.z, tz);
	
	if (windowSize.x*windowSize.y*windowSize.z <= ccl_label)
		ccl_label = 0;
	atomicAdd(&d_hist[ccl_label], 1);
}

__global__ void remain_largest_CCL(uint3 windowSize, uint32* d_buffer, uint32* argmax)
{
	uint32 tx = blockIdx.x * blockDim.x + threadIdx.x;
	uint32 ty = blockIdx.y * blockDim.y + threadIdx.y;
	uint32 tz = blockIdx.z * blockDim.z + threadIdx.z;
	if (tx >= windowSize.x || ty >= windowSize.y || tz >= windowSize.z)
		return;

	uint32 idx = tz * (windowSize.x*windowSize.y) + ty * windowSize.x + tx;

	uint32 label = argmax[0];
	//if (abs(d_buffer[idx] - label) < 1e-06)
	if (d_buffer[idx] == label)
		d_buffer[idx] = 1;
	else
		d_buffer[idx] = 0;
}

__global__ void find_argmax(uint3 windowSize, uint32* d_hist, uint32* maxHist, uint32* argmax)
{
	uint32 tx = blockIdx.x * blockDim.x + threadIdx.x;
	uint32 ty = blockIdx.y * blockDim.y + threadIdx.y;
	uint32 tz = blockIdx.z * blockDim.z + threadIdx.z;
	if (tx >= windowSize.x || ty >= windowSize.y || tz >= windowSize.z)
		return;

	uint32 idx = tz * (windowSize.x*windowSize.y) + ty * windowSize.x + tx;

	if (d_hist[idx] == maxHist[0])
		argmax[0] = idx;
}

__global__ void memcpy_uint32_to_float(uint3 windowSize, float* out_buffer, uint32* in_buffer, bool cond = true)
{
	uint32 tx = blockIdx.x * blockDim.x + threadIdx.x;
	uint32 ty = blockIdx.y * blockDim.y + threadIdx.y;
	uint32 tz = blockIdx.z * blockDim.z + threadIdx.z;
	if (tx >= windowSize.x || ty >= windowSize.y || tz >= windowSize.z)
		return;

	uint32 idx = tz * (windowSize.x*windowSize.y) + ty * windowSize.x + tx;

	if (cond) {
		if (in_buffer[idx] > 0.0f)
			out_buffer[idx] = 1.0f;
		else
			out_buffer[idx] = 0.0f;
	}
	else {
		if (in_buffer[idx] > 0.0f)
			out_buffer[idx] = 0.0f;
		else
			out_buffer[idx] = 1.0f;
	}

}