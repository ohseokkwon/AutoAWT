#include "MarchingCubes.h"
#include <thread>
#include <unordered_map>

#ifndef FAILED_FETCH
#define FAILED_FETCH 0.0f
#endif

int iDivUp(int a, int b)
{
	return (a % b != 0) ? (a / b + 1) : (a / b);
}


bool comp_pair(std::pair<float3, int> a, std::pair<float3, int> b) {
	return a.second < b.second;
}

inline void hash_combine(std::size_t & seed, const float & v)
{
	seed ^= std::hash<float>()(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

struct Hasher_float3
{
	size_t operator()(const float3& k)const
	{
		size_t h = std::hash<float>()(k.x);
		hash_combine(h, k.y);
		hash_combine(h, k.z);
		return h;

		//return std::hash<float>()(k.x) ^ std::hash<float>()(k.y) & (std::hash<float>()(k.z) << 1);
	}

	bool operator()(const float3& a, const float3& b)const
	{
		return abs(a.x - b.x) < std::numeric_limits<float>::epsilon() &&
			abs(a.y - b.y) < std::numeric_limits<float>::epsilon() &&
			abs(a.z - b.z) < std::numeric_limits<float>::epsilon();
	}
};

struct VertexInfo {
	int x = -1;
	float value = std::numeric_limits<float>::min();

	VertexInfo(int _x, float _v) : x(_x), value(_v) { }
};

float roundf_digit(float num, int d) { float t = pow(10, d - 1); return floorf(num * t) / t; }

MarchingCube::MarchingCube(std::string save_path, const void* volume, const glm::vec4 volume_size, const float* pixelSpacing, const glm::vec3 volume_center, INPUT_DATA_ATTRIB type)
{
	m_save_path = save_path;

	cudaError_t cuErr;

	m_volume_size = volume_size;
	allocFundamentalTextures_CUDA(d_edgeTable, d_triTable, d_numVertsTable);
	cuErr = cudaGetLastError();
	std::cerr << cudaGetErrorString(cuErr) << std::endl;
	uint3 gridSize = make_uint3(m_volume_size.x, m_volume_size.y, m_volume_size.z);
	/* if volume_size.w == 1 */
	uint32 size = m_volume_size.x*m_volume_size.y*m_volume_size.z * sizeof(uint16);
	if (type == from_host) {
		cuErr = cudaMalloc((void **)&m_d_volume, size);
		cuErr = cudaMemset(m_d_volume, 0, size);
		//cuErr = cudaMemcpy(m_d_volume, volume, size, cudaMemcpyHostToDevice);
		//cuErr = cudaMalloc((void **)&m_d_volume_fill, size);
		//cuErr = cudaMemcpy(m_d_volume_fill, volume, size, cudaMemcpyHostToDevice);

		//launch_inverse_depth_volume(m_gridSize, m_blockSize, (uint16*)m_d_volume_fill, m_d_volume, gridSize);
		//bindVolumeTexture((uint16*)volume, make_cudaExtent(m_volume_size.x, m_volume_size.y, m_volume_size.z));
	}
	else {
		/* LA 상하 inverse */
		cuErr = cudaMalloc((void **)&m_d_volume, size);
		cuErr = cudaMemset(m_d_volume, 0, size);
		//cuErr = cudaMemcpy(m_d_volume, volume, size, cudaMemcpyDeviceToDevice);

		//cuErr = cudaMalloc((void **)&m_d_volume_fill, size);
		//cuErr = cudaMemcpy(m_d_volume_fill, volume, size, cudaMemcpyHostToDevice);

		//launch_inverse_depth_volume(m_gridSize, m_blockSize, (uint16*)m_d_volume_fill, m_d_volume, gridSize);

		/*cuErr = cudaMemcpy(m_d_volume_fill, volume, size, cudaMemcpyDeviceToDevice);*/
		//m_d_volume = (uint16*)volume;
		/*cuErr = cudaMalloc((void **)&m_d_volume, size);
		cuErr = cudaMemcpy(m_d_volume, volume, size, cudaMemcpyDeviceToDevice);
		bindVolumeTexture((uint16*)volume, make_cudaExtent(m_volume_size.x, m_volume_size.y, m_volume_size.z));*/
	}
	cuErr = cudaGetLastError();
	std::cerr << cudaGetErrorString(cuErr) << std::endl;

	m_numVoxels = volume_size.x * volume_size.y * volume_size.z;
	m_voxelSize = make_float3(pixelSpacing[0], pixelSpacing[1], pixelSpacing[2]);
	m_voxelCenter = make_float3(volume_center.x, volume_center.y, volume_center.z);
	m_maxVertices = m_numVoxels;

	// generate triangles, writing to vertex buffers
	if (m_isUseVBO)
	{
		/*genVBO(&m_pos_vbo, m_numVoxels * sizeof(float) * 4);
		std::cerr << m_pos_vbo << std::endl;
		cuErr = cudaGraphicsGLRegisterBuffer(&m_cuda_pos_vbo_resource, m_pos_vbo, cudaGraphicsMapFlagsWriteDiscard);
		cuErr = cudaGetLastError();
		std::cerr << cudaGetErrorString(cuErr) << std::endl;

		genVBO(&m_norm_vbo, m_numVoxels * sizeof(float) * 4);
		cuErr = cudaGraphicsGLRegisterBuffer(&m_cuda_norm_vbo_resource, m_norm_vbo, cudaGraphicsMapFlagsWriteDiscard);
		cuErr = cudaGetLastError();
		std::cerr << cudaGetErrorString(cuErr) << std::endl;

		cuErr = cudaMalloc((void**)&d_pos, m_numVoxels * sizeof(float) * 4);
		cuErr = cudaMemset(d_pos, 0, m_numVoxels * sizeof(float) * 4);
		cuErr = cudaMalloc((void**)&d_normal, m_numVoxels * sizeof(float) * 4);
		cuErr = cudaMemset(d_normal, 0, m_numVoxels * sizeof(float) * 4);*/
	}
	else {
		cuErr = cudaMalloc((void**)&d_pos, m_numVoxels * sizeof(float) * 4);
		cuErr = cudaMemset(d_pos, 0, m_numVoxels * sizeof(float) * 4);
		cuErr = cudaMalloc((void**)&d_normal, m_numVoxels * sizeof(float) * 4);
		cuErr = cudaMemset(d_normal, 0, m_numVoxels * sizeof(float) * 4);
	}


	if (m_norm_vbo == 0)
		std::cerr << "error : gen normal vbo" << std::endl;

	m_blockSize = dim3(32, 16, 2);
	m_gridSize = dim3(iDivUp(volume_size.x, m_blockSize.x), iDivUp(volume_size.y, m_blockSize.y), iDivUp(1 << (int)ceil(log2((float)m_volume_size.z)), m_blockSize.z));

	// allocate device memory
	uint32 memSize = sizeof(uint32) * m_numVoxels;
	cuErr = cudaMalloc((void **)&d_voxelVerts, memSize);
	cuErr = cudaMalloc((void **)&d_voxelVertsScan, memSize);
	//cuErr = cudaMalloc((void **)&d_compVoxelArray, memSize);
	cuErr = cudaGetLastError();
	std::cerr << cudaGetErrorString(cuErr) << std::endl;
	cuErr = cudaMemset(d_voxelVerts, 0, memSize);
	cuErr = cudaMemset(d_voxelVertsScan, 0, memSize);
	//cuErr = cudaMemset(d_compVoxelArray, 0, memSize);

	//computeISOsurface(m_d_volume_fill);
	//saveMeshInfo("test");
}

MarchingCube::~MarchingCube()
{
	cudaError_t cuErr;

	cuErr = cudaFree(d_edgeTable);
	cuErr = cudaFree(d_triTable);
	cuErr = cudaFree(d_numVertsTable);

	cuErr = cudaFree(d_voxelVerts);
	cuErr = cudaFree(d_voxelVertsScan);
	//cuErr = cudaFree(d_compVoxelArray);

	cuErr = cudaFree(m_d_volume);
	//cuErr = cudaFree(m_d_volume_fill);

	// generate triangles, writing to vertex buffers
	if (m_isUseVBO)
	{
		/*if (m_pos_vbo != 0)
			glDeleteBuffers(1, &m_pos_vbo);
		if (m_norm_vbo != 0)
			glDeleteBuffers(1, &m_norm_vbo);*/
		cudaFree(d_pos);
		cudaFree(d_normal);
	}
	else {
		cudaFree(d_pos);
		cudaFree(d_normal);
	}
}

void MarchingCube::computeISOsurface(const void* volume, INPUT_DATA_ATTRIB type)
{
	cudaError_t cuErr;
	clock_t st_time = clock();

	static float isoValue = 0.5f;
	//isoValue += 0.01f;

	uint3 windowSize = make_uint3(m_volume_size.x, m_volume_size.y, m_volume_size.z);

	if (volume == nullptr) {
		// 내부에서 사용
		//computeFiltering(m_d_volume, m_d_volume, m_volume_size.x, m_volume_size.y, m_volume_size.z, Filter::MEDIAN);
		computeFiltering3D(m_gridSize, m_blockSize, windowSize, m_d_volume, m_d_volume, 3);
	}
	else {
		// 외부에서 입력
		if (type == from_host) {
			uint32 size = m_volume_size.x*m_volume_size.y*m_volume_size.z * sizeof(uint16);
			cuErr = cudaMemcpy(m_d_volume, volume, size, cudaMemcpyHostToDevice);
			launch_inverse_depth_volume(m_gridSize, m_blockSize, (uint16*)m_d_volume, m_d_volume, windowSize, (uint32)(windowSize.z*0.5f));
		}
		else {
			// 방향회전
			uint32 size = m_volume_size.x*m_volume_size.y*m_volume_size.z * sizeof(uint16);
			cuErr = cudaMemcpy(m_d_volume, volume, size, cudaMemcpyHostToDevice);
			computeFiltering3D(m_gridSize, m_blockSize, windowSize, m_d_volume, m_d_volume, 3);
			launch_inverse_depth_volume(m_gridSize, m_blockSize, (uint16*)m_d_volume, m_d_volume, windowSize, (uint32)(windowSize.z*0.5f));
		}
	}

	// calculate number of vertices need per voxel
	launch_classifyVoxel(m_gridSize, m_blockSize, (uint16*)m_d_volume,
		d_voxelVerts,
		windowSize,
		m_numVoxels, m_voxelSize, isoValue);

	// scan voxel vertex count array
	ThrustScanWrapper(d_voxelVertsScan, d_voxelVerts, m_numVoxels);

	// readback total number of vertices
	{
		uint lastElement, lastScanElement;
		cuErr = cudaMemcpy((void *)&lastElement,
			(void *)(d_voxelVerts + m_numVoxels - 1),
			sizeof(uint), cudaMemcpyDeviceToHost);
		cuErr = cudaMemcpy((void *)&lastScanElement,
			(void *)(d_voxelVertsScan + m_numVoxels - 1),
			sizeof(uint), cudaMemcpyDeviceToHost);
		m_totalVerts = lastElement + lastScanElement;

		if (m_totalVerts < 1) {
			isoValue = 0.5f;
		}
	}

	// generate triangles, writing to vertex buffers
	if (m_isUseVBO)
	{
		size_t num_bytes;
		cuErr = cudaGraphicsMapResources(1, &m_cuda_pos_vbo_resource, 0);
		cuErr = cudaGraphicsResourceGetMappedPointer((void **)&d_pos, &num_bytes, m_cuda_pos_vbo_resource);

		cuErr = cudaGraphicsMapResources(1, &m_cuda_norm_vbo_resource, 0);
		cuErr = cudaGraphicsResourceGetMappedPointer((void **)&d_normal, &num_bytes, m_cuda_norm_vbo_resource);
	}

	launch_genTriangles(m_gridSize, m_blockSize, (uint16*)m_d_volume,
		d_pos, d_normal,
		nullptr,
		d_voxelVertsScan,
		windowSize, m_voxelSize, m_voxelCenter, isoValue, m_activeVoxels,
		m_maxVertices);

	// generate triangles, writing to vertex buffers
	if (m_isUseVBO)
	{
		checkCudaErrors(cudaGraphicsUnmapResources(1, &m_cuda_norm_vbo_resource, 0));
		checkCudaErrors(cudaGraphicsUnmapResources(1, &m_cuda_pos_vbo_resource, 0));
	}

	clock_t ed_time = clock();

	float calc_time = (ed_time - st_time);

	st_time = clock();
	saveMeshInfo((m_save_path + "\\surface_mesh").c_str());
	ed_time = clock();

	//// filename <= Patient name or patient ID + date
	////saveMeshInfo("test", d_pos, d_normal, m_maxVertices);
	//calcConvexhull();

	std::cerr << "create surfaces = " << m_totalVerts << std::endl;
	std::cerr << calc_time << "ms, write file = " << (ed_time - st_time) << "ms" << std::endl;
}


void MarchingCube::computeVolume(const char* filename)
{
	dim3 vm_blockSize = dim3(32, 32, 1);
	dim3 vm_gridSize = dim3(iDivUp(m_volume_size.x, vm_blockSize.x), iDivUp(m_volume_size.y, vm_blockSize.y), 1);
	uint3 grid_size = make_uint3(m_volume_size.x, m_volume_size.y, m_volume_size.z);
	uint32* volume_metric_scan_data = nullptr;
	cudaMalloc((void**)&volume_metric_scan_data, grid_size.x*grid_size.y * sizeof(uint32));

	launch_volume_metric(vm_blockSize, vm_gridSize, m_d_volume, grid_size, volume_metric_scan_data);

	uint32* scan_data_1d = nullptr;
	scan_data_1d = new uint32[grid_size.x];
	memset(scan_data_1d, 0, grid_size.x);
	cudaMemcpy(scan_data_1d, volume_metric_scan_data, grid_size.x * sizeof(uint32), cudaMemcpyDeviceToHost);

	uint32 total_valid_voxel = 0;
	for (int i = 0; i < grid_size.x; i++) {
		total_valid_voxel += scan_data_1d[i];
	}

	float voxel_volume = m_voxelSize.x * m_voxelSize.y * m_voxelSize.z; // mm^3
	float total_volume = total_valid_voxel * voxel_volume * 0.001; // cm^3
	float max_volume = grid_size.x * grid_size.y * grid_size.z * voxel_volume * 0.001;
	if (max_volume < total_volume) {
		std::cerr << "볼륨 측정 오류\n";
		total_volume = 0;
		total_valid_voxel = 0;
	}


	cudaFree(volume_metric_scan_data);
	delete[] scan_data_1d;
	std::cerr << total_valid_voxel << "전체 유효 복셀, " << total_volume << "cm^3 전체 볼륨 \n";

	//	saveInfoReport(filename, total_volume, total_valid_voxel);
}

void MarchingCube::saveInfoReport(const char* filename, float total_volume, uint32 total_valid_voxel)
{
	using namespace std;
	ofstream fout;

	fout.open("./" + std::string(filename) + "_report.csv", ios::out);

	if (!fout.is_open()) {
		cerr << "리포팅 실패\n";
		return;
	}

	fout << "Patient name" << ", " << "Total_volume(cm^3)" << ", " << "Total_valid_voxel \n";
	fout << std::string(filename) << ", " << total_volume << ", " << total_valid_voxel << " \n";

	cerr << "report 파일 생성 성공\n";
	fout.close();
}

void MarchingCube::saveMeshInfo(const std::string filename, std::vector<float4>* vertices_list)
{
	std::thread write_stl_thread(MarchingCube::writeSTL, filename, d_pos, d_normal, m_maxVertices);
	write_stl_thread.join();

	if (vertices_list == nullptr)
		return;

	if (vertices_list->size() == 0)
		return;

	std::thread write_plt_thread(MarchingCube::writePLT, filename, d_pos, d_normal, m_maxVertices,
		vertices_list, (float*)&m_voxelSize, (float*)&glm::vec3(m_volume_size), (float*)&m_voxelCenter);
	write_plt_thread.join();
}


void MarchingCube::writePLT(const std::string filename, float4* _d_pos, float4* _d_normal, uint32 maxVertices,
	std::vector<float4>* vertices_list, float* pixelSpacing, float* windowSize, float* voxCenter)
{
	if (_d_pos == nullptr) {
		std::cerr << "vertex정보가 없음\n";
		return;
	}
	if (_d_normal == nullptr) {
		std::cerr << "normal정보가 없음\n";
		return;
	}

	float dataResolution[3] = { windowSize[0], windowSize[1], windowSize[2] };
	float adjustor = 0.0f;
	for (int i = 0; i < 3; i++) {
		adjustor += dataResolution[i] * pixelSpacing[i];
		dataResolution[i] = dataResolution[i] * pixelSpacing[i];
	}
	for (int i = 0; i < 3; i++) {
		dataResolution[i] /= adjustor;
	}

	using namespace std;


	float4* h_pos = new float4[maxVertices];
	cudaMemcpy(h_pos, _d_pos, maxVertices * sizeof(float) * 4, cudaMemcpyDeviceToHost);
	float4* h_normal = new float4[maxVertices];
	cudaMemcpy(h_normal, _d_normal, maxVertices * sizeof(float) * 4, cudaMemcpyDeviceToHost);

	// map를 통해서 동일한 vertices는 같은 정점으로 통일.
	std::unordered_map<float3, VertexInfo, Hasher_float3, Hasher_float3> vertices_map;

	int num_indices = 0;
	float adjusted_value = 1e-03f;
	for (int i = 0; i < maxVertices; i++) {
		h_pos[i] = make_float4(roundf_digit(h_pos[i].x, 3),
			roundf_digit(h_pos[i].y, 3),
			roundf_digit(h_pos[i].z, 3), h_pos[i].w);
		float4 vert = h_pos[i];
		if (vert.w == 0)
			continue;
		float3 vert_3 = make_float3(vert);

		num_indices++;
		auto find_result = vertices_map.find(vert_3);
		if (find_result != vertices_map.end())
			continue;

		vertices_map.insert(std::make_pair(vert_3, VertexInfo(vertices_map.size() + 1, 0.0f)));
	}

	ofstream fout;
	fout.open(filename + ".plt", ios::out);

	if (!fout.is_open() || maxVertices < 1) {
		cerr << "plt 파일 생성 실패\n";
		return;
	}

	fout << "VARIABLES = \"X\", \"Y\", \"Z\", \"Thickness\"\n";
	fout << "ZONE F=FEPOINT, ET=triangle, N=" << vertices_map.size() << " , E=" << int(num_indices / 3) << "\n";

	std::cout << "# of vert: " << vertices_map.size() << std::endl;
	std::vector<std::pair<float3, int>> order; order.resize(vertices_map.size());
	int idx = 0;
	for (auto it = vertices_map.begin(); it != vertices_map.end(); ++it)
	{
		auto vert = it->first;
		order[idx] = make_pair(vert, it->second.x);
		idx++;
	}
	std::sort(order.begin(), order.end(), comp_pair);
	for (auto it = order.begin(); it != order.end(); ++it)
	{
		auto elem = vertices_map.find((*it).first);
		auto vert = elem->first;

		float closestDist = std::numeric_limits<float>::max();
		float4 closest;
		for (auto it = vertices_list->begin(); it != vertices_list->end(); ++it) {
			float dist = length(make_float3(*it) - vert);
			if (dist < closestDist) {
				closestDist = dist;
				closest = *it;
			}
		}

		//! 거리 제약 조건
		float epsil = 1e-03;
		/*fout << vert.x << " " << vert.y << " " << vert.z << " " << ((closestDist > epsil) ? -1.0 : closest.w) << "\n";*/
		fout << vert.x << " " << vert.y << " " << vert.z << " " << closest.w << "\n";
	}

	int num_faces = 0;
	for (int i = 0; i < maxVertices; i += 3) {
		float4 vert1 = h_pos[i + 0];
		if (vert1.w == 0)
			continue;

		int3 index = make_int3(0);

		float4 vert2 = h_pos[i + 1];
		float4 vert3 = h_pos[i + 2];

		auto item_by_map1 = vertices_map.find(make_float3(vert1));
		auto item_by_map2 = vertices_map.find(make_float3(vert2));
		auto item_by_map3 = vertices_map.find(make_float3(vert3));

		if (item_by_map1 != vertices_map.end())
			index.x = item_by_map1->second.x;
		if (item_by_map2 != vertices_map.end())
			index.y = item_by_map2->second.x;
		if (item_by_map3 != vertices_map.end())
			index.z = item_by_map3->second.x;

		if (index.x == 0 || index.y == 0 || index.z == 0)
			continue;
		num_faces++;

		fout << index.x << " " << index.y << " " << index.z << "\n";
	}

	cerr << "plt 파일 생성 성공\n";
	
	fout.close();


	fout.open(filename + ".plt", ios::in | ios::out);
	if (!fout.is_open() || maxVertices < 1) {
		cerr << "plt 파일 생성 실패\n";
		return;
	}
	fout.seekp(0, ios::beg);
	fout << "VARIABLES = \"X\", \"Y\", \"Z\", \"Thickness\"\n";
	fout << "ZONE F=FEPOINT, ET=triangle, N=" << vertices_map.size() << " , E=" << num_faces << "\n";
	cerr << "plt 파일 생성 성공\n";
	fout.close();
	std::cout << filename + ".plt" << std::endl;

	delete[] h_pos;
	delete[] h_normal;
}

void MarchingCube::writeSTL(const std::string filename, float4* _d_pos, float4* _d_normal, uint32 maxVertices)
{
	if (_d_pos == nullptr) {
		std::cerr << "vertex정보가 없음\n";
		return;
	}
	if (_d_normal == nullptr) {
		std::cerr << "normal정보가 없음\n";
		return;
	}

	using namespace std;
	ofstream fout;
	fout.open(filename + ".stl", ios::out);

	if (!fout.is_open() || maxVertices < 1) {
		cerr << "stl 파일 생성 실패\n";
		return;
	}

	fout << "solid ascii \n";

	float4* h_pos = new float4[maxVertices];
	cudaMemcpy(h_pos, _d_pos, maxVertices * sizeof(float) * 4, cudaMemcpyDeviceToHost);
	float4* h_normal = new float4[maxVertices];
	cudaMemcpy(h_normal, _d_normal, maxVertices * sizeof(float) * 4, cudaMemcpyDeviceToHost);

	for (int i = 0; i < (maxVertices - 3); i += 3) {
		float4 pos = h_pos[i + 0];
		if (pos.w == 0)
			continue;

		float4 normal = h_normal[i + 0];
		fout << "facet normal " << normal.x << " " << normal.y << " " << normal.z << " \n";
		fout << "outer loop\n";


		fout << "vertex " << pos.x << " " << pos.y << " " << pos.z << " \n";
		pos = h_pos[i + 1];
		fout << "vertex " << pos.x << " " << pos.y << " " << pos.z << " \n";
		pos = h_pos[i + 2];
		fout << "vertex " << pos.x << " " << pos.y << " " << pos.z << " \n";

		fout << "endloop\n";
		fout << "endfacet\n";
	}
	fout << "endsolid \n";

	cerr << "stl 파일 생성 성공\n";
	fout.close();

	delete[] h_pos;
	delete[] h_normal;
}



void MarchingCube::resetShader()
{
}

int MarchingCube::allocGPUMemory()
{


	return success;
}




float4* MarchingCube::getMeshVertex()
{
	return d_pos;
}
