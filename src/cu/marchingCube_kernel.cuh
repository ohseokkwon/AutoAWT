// volume data
texture<unsigned short, 3, cudaReadModeElementType> volumeTex;
// raterized fundamental tables
texture<unsigned int, 1, cudaReadModeElementType> edgeTex;
texture<unsigned int, 1, cudaReadModeElementType> triTex;
texture<unsigned int, 1, cudaReadModeElementType> numVertsTex;

__device__ float isoMin = -400;
__device__ float isoMax = 500;
__device__
float sampleVolume(uint16* data, uint3 p, uint3 gridSize)
{
	p.x = min(p.x, gridSize.x - 1);
	p.y = min(p.y, gridSize.y - 1);
	p.z = min(p.z, gridSize.z - 1);

	if (p.x >= gridSize.x || p.y >= gridSize.y || p.z >= gridSize.z)
		return 0.0f;

	uint idx = p.x + (p.y * blockDim.x * gridDim.x) + (p.z * blockDim.x * gridDim.x * blockDim.y * gridDim.y);

	uint16 intensity = data[idx];
	return (float)intensity;
}

// 노멀 계산
__device__
float3 calcNormal(float3 *v0, float3 *v1, float3 *v2)
{
	float3 edge0 = *v1 - *v0;
	float3 edge1 = *v2 - *v0;

	return cross(edge0, edge1);
}

/**
[7]--------[6]
/          /|
/          / |
[3]--------[2] |
|          |  |
|          | [5]
|          | /
|          |/
[0]--------[1]

[0]-[1]-[4]-[5] 동일한 layer 확장 검사 x
depth는 늘리지 않는 상태에서 x , y 의 확장검사 필요
예시 알고리즘에서는 +1의 boundary만 검사
**/

__device__
void marching_test(float* dst, uint16* volume, uint3 voxel_pos, uint3 gridSize)
{
	const uint32 search_limit = 1;
	uint32 search_count = 1;
	dst[0] = sampleVolume(volume, voxel_pos, gridSize);
	dst[1] = sampleVolume(volume, voxel_pos + make_uint3(1, 0, 0), gridSize);
	dst[2] = sampleVolume(volume, voxel_pos + make_uint3(1, 1, 0), gridSize);
	dst[3] = sampleVolume(volume, voxel_pos + make_uint3(0, 1, 0), gridSize);

	dst[4] = sampleVolume(volume, voxel_pos + make_uint3(0, 0, 1), gridSize);
	dst[5] = sampleVolume(volume, voxel_pos + make_uint3(1, 0, 1), gridSize);
	dst[6] = sampleVolume(volume, voxel_pos + make_uint3(1, 1, 1), gridSize);
	dst[7] = sampleVolume(volume, voxel_pos + make_uint3(0, 1, 1), gridSize);
}

// compute interpolated vertex along an edge
__device__
float3 vertexInterp(float isolevel, float3 p0, float3 p1, float f0, float f1)
{
	float t = (isolevel - f0) / (f1 - f0);
	return lerp(p0, p1, isolevel);
}

__global__ void
classifyVoxel(uint16* volume, uint *voxelVerts,
	uint3 gridSize, uint numVoxels,
	float3 voxelSize, float isoValue)
{
	uint32 tx = blockIdx.x * blockDim.x + threadIdx.x;
	uint32 ty = blockIdx.y * blockDim.y + threadIdx.y;
	uint32 tz = blockIdx.z * blockDim.z + threadIdx.z;
	if (tx >= gridSize.x || ty >= gridSize.y || tz >= gridSize.z)
		return;
	uint32 idx = tx + (ty * blockDim.x * gridDim.x) + (tz * blockDim.x * gridDim.x * blockDim.y * gridDim.y);
	uint3 gridPos = make_uint3(tx, ty, tz);

	// read field values at neighbouring grid vertices
	float field[8] = { 0 };
	marching_test(field, volume, gridPos, gridSize);

	// calculate flag indicating if each vertex is inside or outside isosurface
	uint cubeindex = 0;
	cubeindex = uint(field[0] > 0);
	cubeindex += uint(field[1] > 0) * 2;
	cubeindex += uint(field[2] > 0) * 4;
	cubeindex += uint(field[3] > 0) * 8;
	cubeindex += uint(field[4] > 0) * 16;
	cubeindex += uint(field[5] > 0) * 32;
	cubeindex += uint(field[6] > 0) * 64;
	cubeindex += uint(field[7] > 0) * 128;

	// read number of vertices from texture
	uint numVerts = tex1Dfetch(numVertsTex, cubeindex);

	if (idx < numVoxels)
	{
		voxelVerts[idx] = numVerts;
	}
}

// Triangle 계산
__global__ void
genTriangles(uint16* volume, float4 *pos, float4 *norm, uint *compactedVoxelArray, uint *numVertsScanned,
	uint3 gridSize,
	float3 voxelSize, float3 voxelCenter, float isoValue, uint activeVoxels, uint maxVerts)
{
	uint32 tx = blockIdx.x * blockDim.x + threadIdx.x;
	uint32 ty = blockIdx.y * blockDim.y + threadIdx.y;
	uint32 tz = blockIdx.z * blockDim.z + threadIdx.z;
	if (tx >= gridSize.x || ty >= gridSize.y || tz >= gridSize.z)
		return;
	uint32 idx = tx + (ty * blockDim.x * gridDim.x) + (tz * blockDim.x * gridDim.x * blockDim.y * gridDim.y);
	uint3 gridPos = make_uint3(tx, ty, tz);

	if (idx > maxVerts - 1)
	{
		idx = maxVerts - 1;
	}

	float3 p = voxelCenter - make_float3(0, 0, (gridSize.z * voxelSize.z)) + make_float3(tx, ty, tz) * voxelSize;//

																												 // calculate cell vertex positions
	float3 v[8];
	v[0] = p;
	v[1] = p + make_float3(voxelSize.x, 0, 0);
	v[2] = p + make_float3(voxelSize.x, voxelSize.y, 0);
	v[3] = p + make_float3(0, voxelSize.y, 0);
	v[4] = p + make_float3(0, 0, voxelSize.z);
	v[5] = p + make_float3(voxelSize.x, 0, voxelSize.z);
	v[6] = p + make_float3(voxelSize.x, voxelSize.y, voxelSize.z);
	v[7] = p + make_float3(0, voxelSize.y, voxelSize.z);

	float field[8] = { 0 };
	marching_test(field, volume, gridPos, gridSize);

	// recalculate flag
	uint cubeindex = 0;
	cubeindex = uint(field[0] > 0);
	cubeindex += uint(field[1] > 0) * 2;
	cubeindex += uint(field[2] > 0) * 4;
	cubeindex += uint(field[3] > 0) * 8;
	cubeindex += uint(field[4] > 0) * 16;
	cubeindex += uint(field[5] > 0) * 32;
	cubeindex += uint(field[6] > 0) * 64;
	cubeindex += uint(field[7] > 0) * 128;

	// find the vertices where the surface intersects the cube
	float3 vertlist[12];

	vertlist[0] = vertexInterp(isoValue, v[0], v[1], field[0], field[1]);
	vertlist[1] = vertexInterp(isoValue, v[1], v[2], field[1], field[2]);
	vertlist[2] = vertexInterp(isoValue, v[2], v[3], field[2], field[3]);
	vertlist[3] = vertexInterp(isoValue, v[3], v[0], field[3], field[0]);

	vertlist[4] = vertexInterp(isoValue, v[4], v[5], field[4], field[5]);
	vertlist[5] = vertexInterp(isoValue, v[5], v[6], field[5], field[6]);
	vertlist[6] = vertexInterp(isoValue, v[6], v[7], field[6], field[7]);
	vertlist[7] = vertexInterp(isoValue, v[7], v[4], field[7], field[4]);

	vertlist[8] = vertexInterp(isoValue, v[0], v[4], field[0], field[4]);
	vertlist[9] = vertexInterp(isoValue, v[1], v[5], field[1], field[5]);
	vertlist[10] = vertexInterp(isoValue, v[2], v[6], field[2], field[6]);
	vertlist[11] = vertexInterp(isoValue, v[3], v[7], field[3], field[7]);

	// output triangle vertices
	uint numVerts = tex1Dfetch(numVertsTex, cubeindex);

	for (int i = 0; i<numVerts; i += 3)
	{
		uint index = numVertsScanned[idx] + i;

		float3 *v[3];
		uint edge;
		edge = tex1Dfetch(triTex, (cubeindex * 16) + i);
		v[0] = &vertlist[edge];


		edge = tex1Dfetch(triTex, (cubeindex * 16) + i + 2);
		v[1] = &vertlist[edge];

		edge = tex1Dfetch(triTex, (cubeindex * 16) + i + 1);
		v[2] = &vertlist[edge];

		// calc normal
		float3 n = calcNormal(v[0], v[1], v[2]);

		if (index < (maxVerts - 3))
		{
			pos[index] = make_float4(*v[0], 1.0f);
			norm[index] = make_float4(n, 0.0f);

			pos[index + 1] = make_float4(*v[1], 1.0f);
			norm[index + 1] = make_float4(n, 0.0f);

			pos[index + 2] = make_float4(*v[2], 1.0f);
			norm[index + 2] = make_float4(n, 0.0f);
		}
	}
}

__global__ void
inverse_depth_volume(uint16* in, uint16* out, uint3 gridSize, uint32 midpoint)
{
	uint32 tx = blockIdx.x * blockDim.x + threadIdx.x;
	uint32 ty = blockIdx.y * blockDim.y + threadIdx.y;
	uint32 tz = blockIdx.z * blockDim.z + threadIdx.z;
	if (tx >= gridSize.x || ty >= gridSize.y || tz >= midpoint)
		return;

	/* z만 뒤집어서 저장 */
	uint32 read_idx = tx + (ty * blockDim.x * gridDim.x) + (tz * blockDim.x * gridDim.x * blockDim.y * gridDim.y);
	uint32 write_idx = tx + (ty * blockDim.x * gridDim.x) + ((gridSize.z - 1 - tz) * blockDim.x * gridDim.x * blockDim.y * gridDim.y);

	uint16 swap_A = in[read_idx];
	uint16 swap_B = in[write_idx];

	out[read_idx] = swap_B;
	out[write_idx] = swap_A;
}

__global__ void
membitSum(uint16* dst, uint3 gridSize)
{
	uint32 tx = blockIdx.x * blockDim.x + threadIdx.x;
	uint32 ty = blockIdx.y * blockDim.y + threadIdx.y;
	uint32 tz = blockIdx.z * blockDim.z + threadIdx.z;
	if (tx >= gridSize.x || ty >= gridSize.y || tz >= gridSize.z)
		return;

	/*scr coord & pixel coord 축 방향 차이로 인한 inverse 저장*/
	uint32 idx = tx + ((gridSize.y - 1 - ty) * blockDim.x * gridDim.x) + (tz * blockDim.x * gridDim.x * blockDim.y * gridDim.y);

	byte rowFill = (dst[idx] >> 8) & 0xff;
	byte colFill = (dst[idx]) & 0xff;

	dst[idx] = rowFill && colFill ? 1 : 0;
}

__device__ float cross_2d(float2 v1, float2 v2, float2 p) {
	return (v1.x - p.x)*(v2.y - p.y) - (v2.x - p.x)*(v1.y - p.y);
}

//! 0 --> p, q and r are collinear
//! 1 --> Clockwise
//! 2 --> Counterclockwise
__device__ float isLeft(float2 P0, float2 P1, float2 P2)
{
	return ((P1.x - P0.x) * (P2.y - P0.y)
		- (P2.x - P0.x) * (P1.y - P0.y));
}

// Given three collinear points p, q, r, the function checks if
// point q lies on line segment 'pr'
__device__ bool onSegment(float2 p, float2 q, float2 r)
{
	if (q.x <= max(p.x, r.x) && q.x >= min(p.x, r.x) &&
		q.y <= max(p.y, r.y) && q.y >= min(p.y, r.y))
		return true;
	return false;
}

// 0 --> p, q and r are collinear
// 1 --> Clockwise
// 2 --> Counterclockwise
__device__ int orientation(float2 p, float2 q, float2 r)
{
	int val = (q.y - p.y) * (r.x - q.x) -
		(q.x - p.x) * (r.y - q.y);

	if (val == 0) return 0; // collinear
	return (val > 0) ? 1 : 2; // clock or counterclock wise
}

// The function that returns true if line segment 'p1q1'
// and 'p2q2' intersect.
__device__ bool doIntersect(float2 p1, float2 q1, float2 p2, float2 q2)
{
	// Find the four orientations needed for general and
	// special cases
	int o1 = orientation(p1, q1, p2);
	int o2 = orientation(p1, q1, q2);
	int o3 = orientation(p2, q2, p1);
	int o4 = orientation(p2, q2, q1);

	// General case
	if (o1 != o2 && o3 != o4)
		return true;

	// Special Cases
	// p1, q1 and p2 are collinear and p2 lies on segment p1q1
	if (o1 == 0 && onSegment(p1, p2, q1)) return true;

	// p1, q1 and p2 are collinear and q2 lies on segment p1q1
	if (o2 == 0 && onSegment(p1, q2, q1)) return true;

	// p2, q2 and p1 are collinear and p1 lies on segment p2q2
	if (o3 == 0 && onSegment(p2, p1, q2)) return true;

	// p2, q2 and q1 are collinear and q1 lies on segment p2q2
	if (o4 == 0 && onSegment(p2, q1, q2)) return true;

	return false; // Doesn't fall in any of the above cases
}

__global__ void
polygon_fill_2D(uint16* out, uint depth, uint3 gridSize, float2* contour, uint contour_size)
{
	// X-Y기준으로 2D 검사
	uint32 tx = blockIdx.x * blockDim.x + threadIdx.x;
	uint32 ty = blockIdx.y * blockDim.y + threadIdx.y;
	if (tx >= gridSize.x || ty >= gridSize.y || depth >= gridSize.z)
		return;

	uint32 idx = tx + (ty * blockDim.x * gridDim.x) + (depth * blockDim.x * gridDim.x * blockDim.y * gridDim.y);
	float2 q = make_float2(tx, ty);
	//q = make_float2(q.x / (float)gridSize.x * 2.0f - 1.0f, (gridSize.y - q.y) / (float)gridSize.y * 2.0f - 1.0f);

	int    wn =  0;
	float2 extreme = make_float2( 1e+5, q.y );
	for (int i = 0; i < contour_size; i++) {
		float2 v1 = contour[ i % contour_size];
		float2 v2 = contour[(i + 1) % contour_size];

		if (doIntersect(v1, v2, q, extreme))
		{
			if (orientation(v1, q, v2) == 0) {
				out[idx] = onSegment(v1, q, v2) ? 1 : 0;
				return;
			}

			wn++;
		}

		/*if (v1.y <= q.y) {
			if (v2.y > q.y)
				if (isLeft(v1, v2, q) > 0)
					++wn;
		}
		else {
			if (v2.y <= q.y)
				if (isLeft(v1, v2, q) < 0)
					--wn;
		}*/
	}

	out[idx] = (wn & 1) ? 1 : 0;
}

__global__ void
volume_metric_3D(uint16* in_volume, uint3 gridSize, uint32* out_2d)
{
	// X-Y 방향으로 전진하며 검사
	// 1. X-Y의 각 위치에서 출발하여, Z를 증가시키며, 모든 valid voxel을 카운팅
	// 쓰레드 개수 : gridSize.x * gridSize.y
	// 결과는 2D data로 저장 (X, Y resolution으로)

	uint32 tx = blockIdx.x * blockDim.x + threadIdx.x;
	uint32 ty = blockIdx.y * blockDim.y + threadIdx.y;
	if (tx >= gridSize.x || ty >= gridSize.y)
		return;

	uint32 idx = tx + (ty * blockDim.x * gridDim.x);

	uint32 validVoxel_cnt = 0;
	for (uint32 tz = 0; tz < gridSize.z; tz++) {
		uint32 search_idx = tx + (ty * blockDim.x * gridDim.x) + (tz * blockDim.x * gridDim.x * blockDim.y * gridDim.y);
		validVoxel_cnt += in_volume[search_idx] > 0 ? 1 : 0;
	}

	out_2d[idx] = validVoxel_cnt;
}

__global__ void
volume_metric_2D(uint32* in_2d, uint3 gridSize)
{
	uint32 tx = blockIdx.x * blockDim.x + threadIdx.x;

	uint32 idx = tx;

	uint32 validVoxel_cnt = 0;
	for (uint32 ty = 0; ty < gridSize.y; ty++) {
		uint32 search_idx = tx + (ty * blockDim.x * gridDim.x);
		validVoxel_cnt += in_2d[search_idx];
	}

	in_2d[tx] = validVoxel_cnt;
}