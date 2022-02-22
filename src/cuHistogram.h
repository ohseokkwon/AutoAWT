#pragma once
#include "Ext_headers.h"

class cuHist {
public:

	int* d_hist = nullptr;
	int* h_hist = nullptr;
	int m_hist_memSize = 0;

	cuHist(uint16* _d_HU, uint2 HU_minmax);
	~cuHist();

	void computeHist(uint16* _d_HU, int* _d_hist, glm::ivec3 _WHD);
	int getMin();
	int getMax();
};