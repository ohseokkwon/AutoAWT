#pragma once
#ifndef _MARCHING_CUBES_KERNEL_CU_
#define _MARCHING_CUBES_KERNEL_CU_

#include "../Ext_headers.h"
#include "marchingCube_kernel.cuh"
#include "mcFundamentalTables.h"

cudaArray* d_volumeArray;

extern "C"
void bindVolumeTexture(uint16 *d_volume, cudaExtent volumeSize)
{
	// bind to linear texture
	//checkCudaErrors(cudaBindTexture(0, volumeTex, d_volume, cudaCreateChannelDesc(16, 0, 0, 0, cudaChannelFormatKindUnsigned)));

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uint16>();
	cudaMalloc3DArray(&d_volumeArray, &channelDesc, volumeSize);

	// copy data to 3D array
	cudaMemcpy3DParms copyParams = { 0 };
	copyParams.srcPtr = make_cudaPitchedPtr(d_volume, volumeSize.width * sizeof(uint16), volumeSize.width, volumeSize.height);
	copyParams.dstArray = d_volumeArray;
	copyParams.extent = volumeSize;
	copyParams.kind = cudaMemcpyHostToDevice;
	cudaMemcpy3D(&copyParams);

	// set texture parameters
	volumeTex.normalized = true;                      // access with normalized texture coordinates
	volumeTex.filterMode = cudaFilterModeLinear;      // linear interpolation
	volumeTex.addressMode[0] = cudaAddressModeWrap;  // clamp texture coordinates
	volumeTex.addressMode[1] = cudaAddressModeWrap;
}

extern "C"
void allocFundamentalTextures_CUDA(uint32 *d_edgeTable, uint32 *d_triTable, uint32 *d_numVertsTable)
{
	checkCudaErrors(cudaMalloc((void **)&d_edgeTable, 256 * sizeof(uint)));
	checkCudaErrors(cudaMemcpy(d_edgeTable, (void *)edgeTable, 256 * sizeof(uint), cudaMemcpyHostToDevice));
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned);
	checkCudaErrors(cudaBindTexture(0, edgeTex, d_edgeTable, channelDesc));

	checkCudaErrors(cudaMalloc((void **)&d_triTable, 256 * 16 * sizeof(uint)));
	checkCudaErrors(cudaMemcpy(d_triTable, (void *)triTable, 256 * 16 * sizeof(uint), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaBindTexture(0, triTex, d_triTable, channelDesc));

	checkCudaErrors(cudaMalloc((void **)&d_numVertsTable, 256 * sizeof(uint)));
	checkCudaErrors(cudaMemcpy(d_numVertsTable, (void *)numVertsTable, 256 * sizeof(uint), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaBindTexture(0, numVertsTex, d_numVertsTable, channelDesc));
}

extern "C" void
launch_classifyVoxel(dim3 grid, dim3 threads,
	uint16* volume, uint *voxelVerts,
	uint3 gridSize, uint numVoxels, float3 voxelSize, float isoValue)
{
	// calculate number of vertices need per voxel
	classifyVoxel << <grid, threads >> >(volume, voxelVerts,
		gridSize,
		numVoxels, voxelSize, isoValue);
	getLastCudaError("classifyVoxel failed");
}

extern "C" void
launch_genTriangles(dim3 grid, dim3 threads, uint16* volume,
	float4 *pos, float4 *norm, uint *compactedVoxelArray, uint *numVertsScanned,
	uint3 gridSize,
	float3 voxelSize, float3 voxelCenter, float isoValue, uint activeVoxels, uint maxVerts)
{
	genTriangles << <grid, threads >> >(volume, pos, norm,
		compactedVoxelArray,
		numVertsScanned,
		gridSize,
		voxelSize, voxelCenter, isoValue, activeVoxels,
		maxVerts);
	getLastCudaError("genTriangles failed");
}

extern "C" void
launch_inverse_depth_volume(dim3 grid, dim3 threads, uint16* in, uint16* out, uint3 gridSize, uint32 midpoint)
{
	inverse_depth_volume << <grid, threads >> > (in, out, gridSize, midpoint);

	getLastCudaError("inv 3d model failed");
}

extern "C" void
launch_polygon_fill_2D(dim3 grid, dim3 threads, uint16* out, uint depth, uint3 gridSize, float2* contour, uint contour_size)
{
	polygon_fill_2D << <grid, threads >> > (out, depth, gridSize, contour, contour_size);

	getLastCudaError("polygon_fill_2D failed");
}

extern "C" void
launch_volume_metric(dim3 grid, dim3 threads, uint16* in_volume, uint3 gridSize, uint32* out_2d)
{
	volume_metric_3D << <grid, threads >> > (in_volume, gridSize, out_2d);
	getLastCudaError("volume_metric_3D failed");

	dim3 blockSize_1d = dim3(32, 1, 1);
	dim3 gridSize_1d = dim3(iDivUp(gridSize.x, blockSize_1d.x), 1, 1);

	volume_metric_2D << <gridSize_1d, blockSize_1d >> > (out_2d, gridSize);
	getLastCudaError("volume_metric_2D failed");
}


extern "C" void
launch_membitSum(dim3 grid, dim3 threads, uint16* d_filled, uint3 gridSize)
{
	membitSum << <grid, threads >> > (d_filled, gridSize);

	getLastCudaError("membitSum failed");
}

extern "C" void ThrustScanWrapper(unsigned int *output, unsigned int *input, unsigned int numElements)
{
	thrust::exclusive_scan(thrust::device_ptr<unsigned int>(input),
		thrust::device_ptr<unsigned int>(input + numElements),
		thrust::device_ptr<unsigned int>(output));
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif _MARCHING_CUBES_KERNEL_CU_