#pragma once
#ifndef _util_3D_KERNEL_CU_
#define _util_3D_KERNEL_CU_

#include "../Ext_headers.h"
#include "util_3D.cuh"
#include <npp.h>

extern "C"
void computeHist(uint16* d_HU, int* d_hist, uint3 WHD)
{
	for (int d = 0; d < WHD.z; d++) {
		calculate2DHistogram <<<WHD.x, WHD.y >> > (d_HU, d_hist, WHD.x, WHD.y, WHD.z, d);
	}
}

extern "C"
void computeFiltering3D(dim3 grid, dim3 threads, uint3 windowSize, uint16* in_Vox, uint16* out_Vox, const uint kernelSize)
{
	median_3Dvoxels << <grid, threads >> > (in_Vox, out_Vox, windowSize, kernelSize);
}

extern "C"
void fetch_in_Voxel(uint3 windowSize, uint16* input, uint16* output, int depth, int axis)
{
	fetch_in_voxel_kernel << <windowSize.x, windowSize.y >> > (windowSize, input, output, depth, axis);
}
extern "C"
void normalizeCUDA(dim3 grid, dim3 threads, uint3 windowSize, uint16* input, uint16* output, int* d_thresholds, int threshold)
{
	normalize_kernel << <grid, threads >> > (windowSize, input, output, d_thresholds, threshold);
}

#endif _util_3D_KERNEL_CU_