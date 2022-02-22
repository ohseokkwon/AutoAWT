#pragma once
#ifndef _LAWT_KERNEL_CU_
#define _LAWT_KERNEL_CU_

#include "../Ext_headers.h"
#include "WT_kernel.cuh"
#include <npp.h>

extern "C"
void fillupVolumebyMask(dim3 grid, dim3 threads, uint3 windowSize, uint16* in_Mask, float* out_Mask, float setValue, uint16 baseValue)
{
	fillupVolumebyMask_kernel << <grid, threads >> > (windowSize, in_Mask, out_Mask, setValue, baseValue);
}

extern "C"
void inverseMask3DU(dim3 grid, dim3 threads, uint3 windowSize, uint16* in_Mask)
{
	inverseMask3D_kernel << <grid, threads >> > (windowSize, in_Mask);
}

extern "C"
void inverseMask3DF(dim3 grid, dim3 threads, uint3 windowSize, float* in_Mask)
{
	inverseMask3D_kernel << <grid, threads >> > (windowSize, in_Mask);
}

extern "C"
void copy_uint16_to_float(dim3 grid, dim3 threads, uint3 windowSize, float* out_buf, uint16* in_buf)
{
	copy_uint16_to_float_kernel << <grid, threads >> > (windowSize, out_buf, in_buf);
}

extern "C"
void cutoffVolume(dim3 grid, dim3 threads, uint3 windowSize, float* in_vf3D, float cutoff, float setValue)
{
	cutoffVolume_kernel << <grid, threads >> > (windowSize, in_vf3D, cutoff, setValue);
}

extern "C"
void binalized(dim3 grid, dim3 threads, uint3 windowSize, float* in_vf3D, float cutoff)
{
	binalized_kernel << <grid, threads >> > (windowSize, in_vf3D, cutoff);
}

extern "C"
void connectivityFiltering(dim3 grid, dim3 threads, uint3 windowSize, uint16* in_wall, float* in_cond, float* out_cond, float setValue)
{
	connectivityFiltering_kernel << <grid, threads >> > (windowSize, in_wall, in_cond, out_cond, setValue);
}

extern "C"
void subtract_by_bool(dim3 grid, dim3 threads, uint3 windowSize, float* io_bufA, uint16* in_boolBuf)
{
	subtract_by_bool_kernel << <grid, threads >> > (windowSize, io_bufA, in_boolBuf);
}

extern "C"
void morphological(dim3 grid, dim3 threads, uint3 windowSize, uint16* d_HUvox, uint16* d_FilteredHU, uint kernelSize, MORPHOLOGY filter)
{
	switch (filter)
	{
	case DILATION:
		dilation_kernel << <grid, threads >> > (windowSize, d_HUvox, d_FilteredHU, kernelSize);
		break;
	case ERODE:
		erode_kernel << <grid, threads >> > (windowSize, d_HUvox, d_FilteredHU, kernelSize);
		break;
	case OPENING:
		erode_kernel << <grid, threads >> > (windowSize, d_HUvox, d_FilteredHU, kernelSize);
		dilation_kernel << <grid, threads >> > (windowSize, d_HUvox, d_FilteredHU, kernelSize);
		break;
	case CLOSING:
		dilation_kernel << <grid, threads >> > (windowSize, d_HUvox, d_FilteredHU, kernelSize);
		erode_kernel << <grid, threads >> > (windowSize, d_HUvox, d_FilteredHU, kernelSize);
		break;
	}
}

extern "C"
void normalize_floatbuf(dim3 grid, dim3 threads, uint3 windowSize, float* input, uint16* output)
{
	float* d_min = nullptr, *d_max = nullptr;
	cudaMalloc((void**)&d_min, sizeof(float) * 1);
	cudaMemset(d_min, 0, sizeof(float) * 1);
	cudaMalloc((void**)&d_max, sizeof(float) * 1);
	cudaMemset(d_max, 0, sizeof(float) * 1);
	reduce_fMinMax << <grid, threads >> > (windowSize, input, d_min, d_max);
	float h_min = 0, h_max = 0;
	cudaMemcpy(&h_min, d_min, sizeof(float) * 1, cudaMemcpyDeviceToHost);
	cudaMemcpy(&h_max, d_max, sizeof(float) * 1, cudaMemcpyDeviceToHost);
	std::cout << "min/max:" << h_min << ", " << h_max << std::endl;

	normalize_floatbuf_kernel << <grid, threads >> > (windowSize, input, output, d_min, d_max);

	cudaFree(d_min);
	cudaFree(d_max);
}

extern "C"
void cufloat_memset_by_value(dim3 grid, dim3 threads, uint3 windowSize, float* in_buffer, float setValue)
{
	cufloat_memset_by_value_kernel << <grid, threads >> > (windowSize, in_buffer, setValue);
}

extern "C"
void cufloat4_memset_by_value(dim3 grid, dim3 threads, uint3 windowSize, float4* in_buffer, int setOffset, float setValue)
{
	cufloat4_memset_by_value_kernel << <grid, threads >> > (windowSize, in_buffer, setOffset, setValue);
}

extern "C"
void computeLaplaceEquation(dim3 grid, dim3 threads, uint iterCnt, uint3 windowSize, float* in_vf3D, float* out_vf3D, uint16* limitCond)
{
	float* in_ptr = in_vf3D;
	float* out_ptr = out_vf3D;
	float* d_Epsilon = nullptr;
	cudaMalloc((void**)&d_Epsilon, sizeof(float) * 1);
	cudaMemset(d_Epsilon, 0, sizeof(float) * 1);
	float h_Epsilon = 0.0f;

	for (int iter = 0; iter < iterCnt; iter++) {
		float p_ei = h_Epsilon;
		cudaMemset(d_Epsilon, 0, sizeof(float) * 1);
		computeLaplaceEquation_kernel << <grid, threads >> > (windowSize, in_ptr, out_ptr, limitCond, d_Epsilon);
		float* tmp = in_ptr;
		in_ptr = out_ptr;
		out_ptr = tmp;

		cudaMemcpy(&h_Epsilon, d_Epsilon, sizeof(float) * 1, cudaMemcpyDeviceToHost);

		float p_ei_1 = h_Epsilon;

		if (p_ei > 0) {
			double err = abs((p_ei - p_ei_1) / p_ei);
			if (err < 1e-5 || iter > iterCnt) {
				std::cerr << "e=" << err << std::endl;
				break;
			}
			std::cerr << "iter = " << iter << ", E=" << p_ei << ", next E=" << p_ei_1 << ", ERR= " << err << std::endl;
		}
	}

	cudaFree(d_Epsilon);
}

extern "C"
void computeLaplaceEquation_with_Vector(dim3 grid, dim3 threads, uint iterCnt, uint3 windowSize, float* in_vectorfields3D, float* out, float4* G, uint16* mask)
{
	float* in_ptr = in_vectorfields3D;
	float* out_ptr = out;
	float* d_Epsilon = nullptr;
	cudaMalloc((void**)&d_Epsilon, sizeof(float) * 1);
	cudaMemset(d_Epsilon, 0, sizeof(float) * 1);
	float h_Epsilon = 0.0f;

	for (int iter = 0; iter < iterCnt; iter++) {
		float p_ei = h_Epsilon;
		cudaMemset(d_Epsilon, 0, sizeof(float) * 1);
		computeLaplaceEquation_with_Vector_kernel << <grid, threads >> > (windowSize, in_ptr, out_ptr, G, d_Epsilon, mask);
		float* tmp = in_ptr;
		in_ptr = out_ptr;
		out_ptr = tmp;

		cudaMemcpy(&h_Epsilon, d_Epsilon, sizeof(float) * 1, cudaMemcpyDeviceToHost);

		float p_ei_1 = h_Epsilon;

		if (p_ei > 0) {
			double err = abs((p_ei - p_ei_1) / p_ei);
			if (err < 1e-5 || iter > 400) {
				std::cerr << "e=" << err << std::endl;
				break;
			}
			std::cerr << "iter = " << iter << ", E=" << p_ei << ", next E=" << p_ei_1 << ", ERR= " << err << std::endl;
		}
	}

	cudaFree(d_Epsilon);
}

extern "C"
void compute_thickness(dim3 grid, dim3 threads, uint3 windowSize, uint16* in_mask, float4* in_vectorfields, uint mode, float4* in_Vertices, float3* out_normal, int totalSize, float3 voxel_size)
{
	compute_thickness_kernel << <grid, threads >> > (windowSize, in_mask, in_vectorfields, mode, in_Vertices, out_normal, totalSize, voxel_size);
}


// CCL
const int BLOCK = 256;
extern "C"
void cuda_ccl(dim3 grid, dim3 threads, uint3 windowSize, float* d_buffer, int degree_of_connectivity)
{
	int N = windowSize.x * windowSize.y * windowSize.z;
	int iter = max(windowSize.x, windowSize.y);
	iter = max(iter, windowSize.z);

	uint32* Ld = nullptr;
	uint32* Rd = nullptr;
	
	cudaMalloc((void**)&Ld, sizeof(uint32) * N);
	cudaMalloc((void**)&Rd, sizeof(uint32) * N);

	init_CCL << <grid, threads >> > (windowSize, Ld, Rd);
	
	uint32* Ld_ptr = Ld;
	uint32* Rd_ptr = Rd;
	for (int i = 0; i < iter - 1; i++) {
		scanning << <grid, threads >> > (windowSize, d_buffer, Ld, Rd);
		cudaMemcpy(Ld, Rd, sizeof(float) * N, cudaMemcpyDeviceToDevice);
	}
	Ld_ptr = Ld;
	cudaFree(Rd);
	
	//memcpy_uint32_to_float << <grid, threads >> > (windowSize, d_buffer, Ld);
	//cudaMemcpy(d_buffer, Ld_ptr, sizeof(float) * N, cudaMemcpyDeviceToDevice);

	uint32* hist = nullptr;
	cudaMalloc((void**)&hist, sizeof(uint32) * N);
	cudaMemset(hist, 0, sizeof(uint32) * N);
	analysis_ccl << <grid, threads >> > (windowSize, Ld, hist);

	uint32 h_min = 0, h_max = 0;
	uint32* d_min = nullptr, *d_max = nullptr;
	cudaMalloc((void**)&d_min, sizeof(uint32) * 1);
	cudaMemset(d_min, 0, sizeof(uint32) * 1);
	cudaMalloc((void**)&d_max, sizeof(uint32) * 1);
	cudaMemset(d_max, 0, sizeof(uint32) * 1);
	
	reduce_MinMax << <grid, threads >> > (windowSize, hist, d_min, d_max, true);
	cudaMemcpy(&h_min, d_min, sizeof(uint32) * 1, cudaMemcpyDeviceToHost);
	cudaMemcpy(&h_max, d_max, sizeof(uint32) * 1, cudaMemcpyDeviceToHost);
	std::cout << "CCL1 min/max:" << h_min << ", " << h_max << std::endl;

	cudaMemset(d_min, 0, sizeof(uint32) * 1);
	uint32* argmax_ptr = d_min;
	find_argmax << <grid, threads >> > (windowSize, hist, d_max, d_min);
	
	h_min = h_max = 0;
	cudaMemcpy(&h_min, d_min, sizeof(uint32) * 1, cudaMemcpyDeviceToHost);
	cudaMemcpy(&h_max, d_max, sizeof(uint32) * 1, cudaMemcpyDeviceToHost);
	std::cout << "CCL argmax:" << h_min << std::endl;
	
	remain_largest_CCL << <grid, threads >> > (windowSize, Ld, d_min);
	cudaFree(d_min);
	cudaFree(d_max);
	cudaFree(hist);
	
	// 내막의 내부가 1인 마스크입니다.
	memcpy_uint32_to_float << <grid, threads >> > (windowSize, d_buffer, Ld, false);
	//cudaMemcpy(d_buffer, Ld_ptr, sizeof(float) * N, cudaMemcpyDeviceToDevice);
	cudaFree(Ld);
}


#endif _LAWT_KERNEL_CU_