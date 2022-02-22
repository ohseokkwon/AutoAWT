#include "cuHistogram.h"

cuHist::cuHist(uint16* _d_HU, uint2 HU_minmax)
{
	cudaError_t cuErr;
	m_hist_memSize = HU_minmax.x * HU_minmax.y;
	cuErr = cudaMalloc((void**)&d_hist, sizeof(int) * m_hist_memSize);
	cuErr = cudaMemset(d_hist, 0, sizeof(int) * m_hist_memSize);

	h_hist = new int[HU_minmax.x * HU_minmax.y];
	memset(h_hist, 0, sizeof(int) * m_hist_memSize);
}

cuHist::~cuHist()
{
	cudaFree(d_hist);
	delete[] h_hist;
}

void cuHist::computeHist(uint16* _d_HU, int* _d_hist, glm::ivec3 _WHD)
{
	computeHist(_d_HU, _d_hist, _WHD);
	cudaMemcpy(h_hist, d_hist, sizeof(int) * m_hist_memSize, cudaMemcpyDeviceToHost);
}

int cuHist::getMin()
{
	return 0;
}

int cuHist::getMax()
{
	return 0;
}