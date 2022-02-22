#pragma once

#include "Ext_headers.h"

class WT {
public:
private:
	std::string m_save_path;

	uint16* m_d_WallMask = nullptr;
	uint16* m_d_ConvexMask = nullptr;
	float* m_d_vectorfields = nullptr;
	uint16* m_h_ChamberMask = nullptr;

	// xyz + w(thickness)
	float* m_d_VFInit = nullptr;

	float3* m_d_normal = nullptr;
	float* m_d_thick = nullptr;

	float4 *d_pos = 0;
	float4 *d_normal = 0;

	glm::vec3 m_volume_size = glm::vec3(0);
	glm::vec3 m_voxel_spacing = glm::vec3(0);
	glm::vec3 m_volume_position = glm::vec3(0);


	dim3 m_blockSize;
	dim3 m_gridSize;
public:
	~WT();

	std::vector<float4> m_endo_vertices_list;

	/**
	* volume : volumetric data's ptr
	* volume_size : width, height, depth, channel
	*/
	WT(std::string save_path, const glm::vec3 volume_size, const glm::vec3 voxel_spacing, const glm::vec3 volume_position, const void* m_WallMask, const void* m_ConvexMask);

	void initialize(Uint16* d_HU);

	void detectEpiEndo(void* mainVolume);
	void segmentEndo_Epi();
	void evalWT();
	void savePLT(std::string fname, float4* vertices, int elemCnt, std::vector<float4>* vertices_list = nullptr);

	uint16* getChamberMask() {
		return m_h_ChamberMask;
	}


	//! 임시코드 테스트 후 삭제
	void WT::find_boundary(void* mainVolume);
private:

	static void exportBMP(const char* filename, float* d_HU, uint3 windowSize, glm::vec3 voxelSize, dim3 gridSize, dim3 blockSize, char* fName);
};