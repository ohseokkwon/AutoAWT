#pragma once

#include "Ext_headers.h"

class MarchingCube {
public:
	enum INPUT_DATA_ATTRIB {
		from_host,
		from_device
	};
private:
	enum ReturnState {
		success,
		error
	};
	std::string m_save_path;
	uint16* m_d_volume = nullptr;
	uint16* m_d_volume_fill = nullptr;

	uint32 m_numVoxels = 0;
	uint32 m_maxVertices = 0;
	float3 m_voxelSize = make_float3(0);
	float3 m_voxelCenter = make_float3(0);

	uint32 m_activeVoxels = 0;
	uint32 m_totalVerts = 0;

	float4 *d_pos = 0;
	float4 *d_normal = 0;

	uint32* d_edgeTable = nullptr;
	uint32* d_triTable = nullptr;
	uint32* d_numVertsTable = nullptr;


	uint* d_voxelVerts = 0;
	uint* d_voxelVertsScan = 0;
	uint* d_compVoxelArray = 0;
	//uint16* d_out = nullptr;

	glm::vec4 m_volume_size = glm::vec4(0);

	GLuint m_fbo = 0;
	GLuint m_cudaTex = 0;
	GLuint m_tex = 0;
	GLuint m_pbo = 0;

	int m_isUseVBO = 0;

	dim3 m_blockSize;
	dim3 m_gridSize;

	GLuint m_pos_vbo = 0;
	GLuint m_norm_vbo = 0;
	struct cudaGraphicsResource *m_cuda_pos_vbo_resource;
	struct cudaGraphicsResource *m_cuda_norm_vbo_resource;

public:
	~MarchingCube();


	/**
	* volume : volumetric data's ptr
	* volume_size : width, height, depth, channel
	*/
	MarchingCube(std::string save_path, const void* volume, const glm::vec4 volume_size, const float* pixelSpacing, const glm::vec3 volume_center, INPUT_DATA_ATTRIB type);
	void resetShader();
	void computeISOsurface(const void* volume = nullptr, INPUT_DATA_ATTRIB type = from_device);

	void computeVolume(const char* filename);

	void saveMeshInfo(const std::string filename, std::vector<float4>* points = nullptr);

	float4* getMeshVertex();
private:
	static void writePLT(const std::string filename, float4* _d_pos, float4* _d_normal, uint32 maxVertices,
		std::vector<float4>* vertices_list, float* pixelSpacing, float* windowSize, float* voxCenter);
	static void writeSTL(const std::string filename, float4* _d_pos, float4* _d_normal, uint32 maxVertices);


	int allocGPUMemory();
	/*GLuint genPBO_CUDA();
	GLuint genTexture(uint32 width, uint32 height);
	GLuint genPBO(uint32 size);
	void genVBO(GLuint* _vbo, uint32 size);*/

	void saveInfoReport(const char* filename, float total_volume = 0.0f, uint32 total_valid_voxel = 0);
};