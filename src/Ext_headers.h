#pragma once
#include <windows.h>

#include <glm\glm.hpp>
#include <glm\gtx\transform.hpp>
#include <iostream>
#include <string.h>
#include <vector>
#include <ShObjIdl.h> // 파일오픈다이얼로그 관련
#include <Shlwapi.h>
#include <algorithm>
#include <limits>

#include"dcmtk\ofstd\oftypes.h"
#include "dcmtk\dcmdata\dctk.h"
#include "dcmtk\dcmdata\dcddirif.h"
#include "dcmtk\dcmdata\dcpxitem.h"

#include <driver_functions.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>
#include <helper_cuda.h>
#include <helper_math.h>
#include <helper_cuda_gl.h>
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>

typedef unsigned int uint32;
typedef unsigned char uchar;
typedef unsigned short uint16;

extern "C" void bindVolumeTexture(uint16 *d_volume, cudaExtent volumeSize);
extern "C" void allocFundamentalTextures_CUDA(uint32 *d_edgeTable, uint32 *d_triTable, uint32 *d_numVertsTable);

extern "C" void
launch_classifyVoxel(dim3 grid, dim3 threads,
	uint16* volume, uint *voxelVerts,
	uint3 gridSize, uint numVoxels, float3 voxelSize, float isoValue);

extern "C" void
launch_genTriangles(dim3 grid, dim3 threads, uint16* volume,
	float4 *pos, float4 *norm, uint *compactedVoxelArray, uint *numVertsScanned,
	uint3 gridSize,
	float3 voxelSize, float3 voxelCenter, float isoValue, uint activeVoxels, uint maxVerts);
extern "C" void ThrustScanWrapper(unsigned int *output, unsigned int *input, unsigned int numElements);

extern "C" void
launch_inverse_depth_volume(dim3 grid, dim3 threads, uint16* in, uint16* out, uint3 gridSize, uint32 midpoint);
extern "C" void
launch_polygon_fill_2D(dim3 grid, dim3 threads, uint16* out, uint depth, uint3 gridSize, float2* contour, uint contour_size);
extern "C" void
launch_volume_metric(dim3 grid, dim3 threads, uint16* in_volume, uint3 gridSize, uint32* out_2d);

extern "C" void
launch_membitSum(dim3 grid, dim3 threads, uint16* d_filled, uint3 gridSize);

extern "C" int iDivUp(int a, int b);

extern enum Filter
{
	GAUSSIAN = 1,
	MEDIAN = 2,
	MEDIAN3D
};
extern enum MORPHOLOGY
{
	DILATION = 1,
	ERODE,
	OPENING,
	CLOSING
};

enum class CTview {
	axial = 0, coronal, sagittal
};

extern "C"
void computeHist(uint16* d_HU, int* d_hist, uint3 WHD);
extern "C"
void computeFiltering3D(dim3 grid, dim3 threads, uint3 windowSize, uint16* in_Vox, uint16* out_Vox, const uint kernelSize);
extern "C"
void fetch_in_Voxel(uint3 windowSize, uint16* input, uint16* output, int depth, int axis);
extern "C"
void normalizeCUDA(dim3 grid, dim3 threads, uint3 windowSize, uint16* input, uint16* output, int* d_thresholds, int threshold);


extern "C"
void fillupVolumebyMask(dim3 grid, dim3 threads, uint3 windowSize, uint16* in_Mask, float* out_Mask, float setValue, uint16 baseValue = 0);
extern "C"
void copy_uint16_to_float(dim3 grid, dim3 threads, uint3 windowSize, float* out_buf, uint16* in_buf);
extern "C"
void inverseMask3DU(dim3 grid, dim3 threads, uint3 windowSize, uint16* in_Mask);
extern "C"
void inverseMask3DF(dim3 grid, dim3 threads, uint3 windowSize, float* in_Mask);
extern "C"
void cutoffVolume(dim3 grid, dim3 threads, uint3 windowSize, float* in_vf3D, float cutoff, float setValue);
extern "C"
void normalize_floatbuf(dim3 grid, dim3 threads, uint3 windowSize, float* input, uint16* output);
extern "C"
void binalized(dim3 grid, dim3 threads, uint3 windowSize, float* in_vf3D, float cutoff);
extern "C"
void connectivityFiltering(dim3 grid, dim3 threads, uint3 windowSize, uint16* in_wall, float* in_cond, float* out_cond, float setValue);
extern "C"
void subtract_by_bool(dim3 grid, dim3 threads, uint3 windowSize, float* io_bufA, uint16* in_boolBuf);

extern "C"
void cufloat_memset_by_value(dim3 grid, dim3 threads, uint3 windowSize, float* in_buffer, float setValue);
extern "C"
void cufloat4_memset_by_value(dim3 grid, dim3 threads, uint3 windowSize, float4* in_buffer, int setOffset, float setValue);
extern "C"
void computeLaplaceEquation(dim3 grid, dim3 threads, uint iterCnt, uint3 windowSize, float* in_vf3D, float* out_vf3D, uint16* limitCond);
extern "C"
void computeLaplaceEquation_with_Vector(dim3 grid, dim3 threads, uint iterCnt, uint3 windowSize, float* in_vectorfields3D, float* out, float4* G, uint16* mask);
extern "C"
void compute_thickness(dim3 grid, dim3 threads, uint3 windowSize, uint16* in_mask, float4* in_vectorfields, uint mode, float4* in_Vertices, float3* out_normal, int totalSize, float3 voxel_size);
extern "C"
void morphological(dim3 grid, dim3 threads, uint3 windowSize, uint16* d_HUvox, uint16* d_FilteredHU, uint kernelSize, MORPHOLOGY filter);

// CCL
extern "C"
void cuda_ccl(dim3 grid, dim3 threads, uint3 windowSize, float* d_buffer, int degree_of_connectivity);


extern glm::ivec2 g_screenSize;
extern char* argv_file_path;


#define NTHREADS 64
#define USE_SHARED 0

#define DIM_TEST 1

#define NO_ISOVALUE 0