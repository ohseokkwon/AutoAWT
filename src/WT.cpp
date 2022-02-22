#include "WT.h"
#include <thread>
#include <iostream>
#include <fstream>

WT::WT(std::string save_path, const glm::vec3 volume_size, const glm::vec3 voxel_spacing, const glm::vec3 volume_position, const void* wallMask, const void* convexMask)
{
	m_save_path = save_path;

	cudaError_t cuErr;

	m_volume_size = volume_size;
	m_voxel_spacing = voxel_spacing;
	m_volume_position = volume_position;

	uint3 gridSize = make_uint3(m_volume_size.x, m_volume_size.y, m_volume_size.z);
	/* if volume_size.w == 1 */
	uint32 vol1DSize = m_volume_size.x*m_volume_size.y*m_volume_size.z;
	
	m_blockSize = dim3(32, 16, 2);
	m_gridSize = dim3(iDivUp(volume_size.x, m_blockSize.x), iDivUp(volume_size.y, m_blockSize.y), iDivUp(1 << (int)ceil(log2((float)m_volume_size.z)), m_blockSize.z));

	// allocate device memory
	cuErr = cudaMalloc((void **)&m_d_WallMask, vol1DSize * sizeof(uint16));
	cuErr = cudaMemset(m_d_WallMask, 0, vol1DSize * sizeof(uint16));
	cudaMemcpy(m_d_WallMask, wallMask, vol1DSize * sizeof(uint16), cudaMemcpyHostToDevice);

	cuErr = cudaMalloc((void **)&m_d_ConvexMask, vol1DSize * sizeof(uint16));
	cuErr = cudaMemset(m_d_ConvexMask, 0, vol1DSize * sizeof(uint16));

	m_h_ChamberMask = new uint16[vol1DSize];
	memset(m_h_ChamberMask, 0, vol1DSize * sizeof(uint16));
	

	//float calc_time = 0.0f;
	//clock_t st_time = clock();

	//uint16* ptr_wall = (uint16*)wallMask;
	//// Z axis
	//for (int z = 0; z < volume_size.z; z++) {
	//	std::vector<Point_2> points, result;

	//	for (int y = 0; y < volume_size.y; y++) {
	//		for (int x = 0; x < volume_size.x; x++) {
	//			if (*(ptr_wall + int(volume_size.x*volume_size.y*z + volume_size.x*y + x)) > 0)
	//				points.push_back(Point_2(x, y));
	//		}
	//	}
	//	CGAL::convex_hull_2(points.begin(), points.end(), std::back_inserter(result));

	//	std::cout << result.size() << std::endl;
	//}
	//// Y axis
	//for (int y = 0; y < volume_size.y; y++) {
	//	std::vector<Point_2> points, result;
	//	for (int z = 0; z < volume_size.z; z++) {
	//		for (int x = 0; x < volume_size.x; x++) {
	//			if (*(ptr_wall + int(volume_size.x*volume_size.y*z + volume_size.x*y + x)) > 0)
	//				points.push_back(Point_2(x, z));
	//		}
	//	}
	//	CGAL::convex_hull_2(points.begin(), points.end(), std::back_inserter(result));

	//	std::cout << result.size() << std::endl;
	//}
	//// X axis
	//for (int x = 0; x < volume_size.x; x++) {
	//	std::vector<Point_2> points, result;
	//	for (int z = 0; z < volume_size.z; z++) {
	//		for (int y = 0; y < volume_size.y; y++) {
	//			if (*(ptr_wall + int(volume_size.x*volume_size.y*z + volume_size.x*y + x)) > 0)
	//				points.push_back(Point_2(z, y));
	//		}
	//	}
	//	CGAL::convex_hull_2(points.begin(), points.end(), std::back_inserter(result));

	//	std::cout << result.size() << std::endl;
	//}
	//clock_t ed_time = clock();
	//calc_time = (ed_time - st_time);
	//std::cerr << "Finished convex-hull calculation : " << calc_time / 1e3f << " s \n";

	cudaMemcpy(m_d_ConvexMask, convexMask, vol1DSize * sizeof(uint16), cudaMemcpyHostToDevice);
	// 초기세팅
}

WT::~WT()
{
	cudaError_t cuErr;

	cuErr = cudaFree(m_d_WallMask);
	cuErr = cudaFree(m_d_ConvexMask);

	if (m_d_vectorfields != nullptr)
		cudaFree(m_d_vectorfields);

	if (m_d_normal != nullptr) {
		cudaFree(m_d_normal);
		m_d_normal = nullptr;
	}
}

void WT::initialize(Uint16* d_HU) {

}

//! 임시코드 테스트 후 삭제
void WT::find_boundary(void* mainVolume)
{
	uint3 uVoxSize = make_uint3(m_volume_size.x, m_volume_size.y, m_volume_size.z);
	uint32 memSize = uVoxSize.x*uVoxSize.y*uVoxSize.z;

	float* d_vf3D_frontbuf = nullptr, *d_vf3D_backbuf = nullptr;
	cudaMalloc((void**)&d_vf3D_frontbuf, memSize * sizeof(float));
	cudaMemset(d_vf3D_frontbuf, 0, memSize * sizeof(float));
	cudaMalloc((void**)&d_vf3D_backbuf, memSize * sizeof(float));
	cudaMemset(d_vf3D_backbuf, 0, memSize * sizeof(float));

	// 마스크 영역 플로팅 버퍼 복사
	copy_uint16_to_float(m_gridSize, m_blockSize, uVoxSize, d_vf3D_backbuf, m_d_WallMask);
	// 플로팅 버퍼 반전사
	inverseMask3DF(m_gridSize, m_blockSize, uVoxSize, d_vf3D_backbuf);

	connectivityFiltering(m_gridSize, m_blockSize, uVoxSize, m_d_WallMask, d_vf3D_backbuf, d_vf3D_frontbuf, 3.0);

	exportBMP(m_save_path.c_str(), d_vf3D_frontbuf, uVoxSize, m_voxel_spacing, m_gridSize, m_blockSize, "Boundary_voxels");
}

void WT::detectEpiEndo(void* mainVolume) 
{
	uint3 uVoxSize = make_uint3(m_volume_size.x, m_volume_size.y, m_volume_size.z);
	uint32 memSize = uVoxSize.x*uVoxSize.y*uVoxSize.z;

	float* d_vf3D_frontbuf = nullptr, *d_vf3D_backbuf = nullptr;
	cudaMalloc((void**)&d_vf3D_frontbuf, memSize * sizeof(float));
	cudaMemset(d_vf3D_frontbuf, 0, memSize * sizeof(float));
	cudaMalloc((void**)&d_vf3D_backbuf, memSize * sizeof(float));
	cudaMemset(d_vf3D_backbuf, 0, memSize * sizeof(float));

	// 반전 Mask = 0, outer = 1
	inverseMask3DU(m_gridSize, m_blockSize, uVoxSize, m_d_ConvexMask);
	inverseMask3DU(m_gridSize, m_blockSize, uVoxSize, m_d_WallMask);

	// 초기값 할당
	fillupVolumebyMask(m_gridSize, m_blockSize, uVoxSize,
		m_d_ConvexMask, d_vf3D_frontbuf, 10.0f, 1);
	
	// Laplace equation
	computeLaplaceEquation(m_gridSize, m_blockSize, 100, uVoxSize,
		d_vf3D_frontbuf,
		d_vf3D_backbuf,
		m_d_WallMask);

#ifdef PRINT_BMP
	exportBMP(m_save_path.c_str(), d_vf3D_backbuf, uVoxSize, m_voxel_spacing, m_gridSize, m_blockSize, "Boundary_PDE");
#endif

	// epsilon=1e-05 으로 boundary 추출
	cutoffVolume(m_gridSize, m_blockSize, uVoxSize, d_vf3D_backbuf, 1e-01, -2.0f);
	// 이진화 (외부가 1인 마스크)
	binalized(m_gridSize, m_blockSize, uVoxSize, d_vf3D_backbuf, -2.0f);
	
#ifdef PRINT_BMP
	exportBMP(m_save_path.c_str(), d_vf3D_backbuf, uVoxSize, m_voxel_spacing, m_gridSize, m_blockSize, "pre-CCL");
#endif

	// 마스크 반전 (내부가 1인 마스크)
	inverseMask3DF(m_gridSize, m_blockSize, uVoxSize, d_vf3D_backbuf); // 외부->내부 계산시 (LV용)
	// 마스크 반전 (마스크 원래대로)
	inverseMask3DU(m_gridSize, m_blockSize, uVoxSize, m_d_WallMask);

	// 심근영역 제거
	subtract_by_bool(m_gridSize, m_blockSize, uVoxSize, d_vf3D_backbuf, m_d_WallMask);
	cudaFree(m_d_ConvexMask);

	// CCL check and 가장큰 CCL만 놔두고 마스크화
	cuda_ccl(m_gridSize, m_blockSize, uVoxSize, d_vf3D_backbuf, 4);
	//// 심근영역 제거
	subtract_by_bool(m_gridSize, m_blockSize, uVoxSize, d_vf3D_backbuf, m_d_WallMask);

	//exportBMP(m_save_path.c_str(), d_vf3D_backbuf, uVoxSize, m_voxel_spacing, m_gridSize, m_blockSize, "CCL");
	uint16* normBuf = nullptr;
	cudaMalloc(&normBuf, sizeof(uint16)*memSize);
	normalize_floatbuf(m_gridSize, m_blockSize, uVoxSize, d_vf3D_backbuf, normBuf);
	cudaMemcpy(m_h_ChamberMask, normBuf, sizeof(uint16)*memSize, cudaMemcpyDeviceToHost);
	cudaFree(normBuf);
#ifdef PRINT_BMP
	exportBMP(m_save_path.c_str(), d_vf3D_backbuf, uVoxSize, m_voxel_spacing, m_gridSize, m_blockSize, "CCL");
#endif

	//// 내부-벽-외부 정보 초기화
	cudaMalloc((void**)&m_d_VFInit, memSize * sizeof(float));
	cudaMemcpy(m_d_VFInit, d_vf3D_backbuf, memSize * sizeof(float), cudaMemcpyDeviceToDevice);

	
	// 심외막 추출

	//// wall dilate  ***after adopted***
	//morphological(m_gridSize, m_blockSize, uVoxSize, m_d_WallMask, m_d_ConvexMask, 3U, MORPHOLOGY::DILATION);
	//cudaMemcpy(m_d_WallMask, m_d_ConvexMask, memSize * sizeof(uint16), cudaMemcpyDeviceToDevice);

	// 내부/벽/외부 적용 시 (아니면 주석처리)
	fillupVolumebyMask(m_gridSize, m_blockSize, uVoxSize,
		m_d_WallMask, m_d_VFInit, 0.5f, 1);

#ifdef PRINT_BMP
	exportBMP(m_save_path.c_str(), m_d_VFInit, uVoxSize, m_voxel_spacing, m_gridSize, m_blockSize, "Boundary");
#endif

	// 심근벽은 2으로 초기화
	cudaMemset(d_vf3D_frontbuf, 0, memSize * sizeof(float));
	fillupVolumebyMask(m_gridSize, m_blockSize, uVoxSize,
		m_d_WallMask, d_vf3D_frontbuf, 2.0f, 1);

	
	// 심외막은 3로 초기화
	connectivityFiltering(m_gridSize, m_blockSize, uVoxSize, m_d_WallMask, d_vf3D_backbuf, d_vf3D_frontbuf, 3.0);

	// 심내막은 1로 초기화
	inverseMask3DF(m_gridSize, m_blockSize, uVoxSize, d_vf3D_backbuf);
	subtract_by_bool(m_gridSize, m_blockSize, uVoxSize, d_vf3D_backbuf, m_d_WallMask);
	connectivityFiltering(m_gridSize, m_blockSize, uVoxSize, m_d_WallMask, d_vf3D_backbuf, d_vf3D_frontbuf, 1.0);

	//// 내부 내벽-벽-외벽 외부 정보 초기화
	//cudaMalloc((void**)&m_d_VFInit, memSize * sizeof(float));
	//cudaMemcpy(m_d_VFInit, d_vf3D_frontbuf, memSize * sizeof(float), cudaMemcpyDeviceToDevice);


	//copy_uint16_to_float(m_gridSize, m_blockSize, uVoxSize, d_vf3D_frontbuf, m_d_WallMask);
	cudaMalloc((void**)&m_d_vectorfields, memSize * sizeof(float));
	cudaMemset(m_d_vectorfields, 0, memSize * sizeof(float));
	cudaMemcpy(m_d_vectorfields, d_vf3D_frontbuf, memSize * sizeof(float), cudaMemcpyDeviceToDevice);

#ifdef PRINT_BMP
	// 내부/내벽/벽/외벽/외부 BMP 그림 출력시
#endif
	exportBMP(m_save_path.c_str(), d_vf3D_frontbuf, uVoxSize, m_voxel_spacing, m_gridSize, m_blockSize, "Epi-Endo");

	cudaFree(d_vf3D_backbuf);
	cudaFree(d_vf3D_frontbuf);
}

void WT::exportBMP(const char* filename, float* d_HU, uint3 windowSize, glm::vec3 voxelSize, dim3 gridSize, dim3 blockSize, char* fName)
{
	// 정규화
	uint16* normBuf = nullptr;
	cudaMalloc(&normBuf, sizeof(uint16)*windowSize.x*windowSize.y*windowSize.z);
	normalize_floatbuf(gridSize, blockSize, windowSize, d_HU, normBuf);

	char savePath[128] = { 0 };
	if (fName == nullptr)
		sprintf_s(savePath, "%s\\BMP", filename, fName);
	else
		sprintf_s(savePath, "%s\\%s", filename, fName);

	CreateDirectoryA(savePath, nullptr);

	int memSize = windowSize.x * windowSize.y * sizeof(uint16);
	int channel = 1;

	uint16* d_slice = nullptr;
	cudaMalloc(&d_slice, sizeof(uint16) * windowSize.x * windowSize.y);
	uint16* h_slice = new uint16[windowSize.x * windowSize.y];

	byte* h_slice_mask = new byte[windowSize.x * windowSize.y];

	for (int z = 0; z < windowSize.z; z++) {
		//std::cout << "DCM row = " << z + 1 << std::endl;

		try {
			cudaMemset(d_slice, 0x00, sizeof(uint16) * windowSize.x * windowSize.y);

			fetch_in_Voxel(windowSize, normBuf, d_slice, z, 0);
			cudaMemcpy(h_slice, d_slice, sizeof(uint16) * windowSize.x * windowSize.y, cudaMemcpyDeviceToHost);

			memset(h_slice_mask, 0x00, sizeof(byte)*windowSize.x * windowSize.y);
			for (int h = 0; h < windowSize.y; h++) {
				for (int w = 0; w < windowSize.x; w++) {
					if (h_slice[(h)*windowSize.x+w] > 0)
						h_slice_mask[(windowSize.y - 1 - h)*windowSize.x+w] = h_slice[(h)*windowSize.x+w];
				}
			}

			// 저장폴더 지정
			char saveFileName[128] = { 0 };
			if (fName == nullptr) {
				sprintf_s(saveFileName, "%s\\BMP\\%d.bmp", filename, z + 1);
			}
			else {
				sprintf_s(saveFileName, "%s\\%s\\%d.bmp", filename, fName, z + 1);
			}
			
			std::cout << saveFileName << std::endl;

			FILE *fp;
			fopen_s(&fp, saveFileName, "wb");

			/*std::ofstream fout;
			fout.open(saveFileName, std::ios::binary | std::ios::out);
			if (!fout.is_open()) {
			std::cerr << "BMP file open error!" << std::endl;
			return;
			}*/

			// BMP header
			BITMAPINFOHEADER bi;
			BITMAPFILEHEADER bf;

			bf.bfType = 0x4D42;
			bf.bfOffBits = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER) + sizeof(RGBQUAD) * 256;
			bf.bfSize = windowSize.x * windowSize.y * channel + bf.bfOffBits;
			bf.bfReserved1 = bf.bfReserved2 = 0;

			bi.biSize = sizeof(BITMAPINFOHEADER);
			bi.biWidth = windowSize.x;
			bi.biHeight = windowSize.y;
			bi.biPlanes = 1;
			bi.biBitCount = 8 * channel;
			bi.biCompression = BI_RGB;
			bi.biSizeImage = windowSize.x * windowSize.y * channel;
			bi.biClrUsed = 0;
			bi.biClrImportant = 0;
			bi.biXPelsPerMeter = bi.biYPelsPerMeter = 0;

			RGBQUAD palette[256];
			for (int i = 0; i < 256; ++i)
			{
				palette[i].rgbBlue = (byte)i;
				palette[i].rgbGreen = (byte)i;
				palette[i].rgbRed = (byte)i;
			}

			fwrite(&bf, sizeof(BITMAPFILEHEADER), 1, fp);
			fwrite(&bi, sizeof(BITMAPINFOHEADER), 1, fp);
			fwrite(&palette[0], sizeof(RGBQUAD) * 256, 1, fp);
			fwrite(h_slice_mask, 1, windowSize.x * windowSize.y * sizeof(byte), fp);
			fclose(fp);

			/*fout.write((const char*)&bf, sizeof(BITMAPFILEHEADER));
			fout.write((const char*)&bi, sizeof(BITMAPINFOHEADER));
			fout.write((const char*)h_slice, windowSize.x * windowSize.y * sizeof(byte));

			fout.close();*/
		}
		catch (std::exception& e) {
			std::cout << "exeption : " << e.what();
		}
		// MEM release
	}
	delete[] h_slice_mask;

	delete[] h_slice;
	cudaFree(d_slice);
	cudaFree(normBuf);
}

void WT::segmentEndo_Epi()
{

}

void WT::evalWT()
{
	uint3 uVoxSize = make_uint3(m_volume_size.x, m_volume_size.y, m_volume_size.z);
	uint32 memSize = uVoxSize.x*uVoxSize.y*uVoxSize.z;
	float* voltage3D = nullptr, *voltage3D_backup = nullptr;
	float4* vectorFields = nullptr;

	// VF의 초기 state를 결정합니다. (내외부에 따른 기초값 설정)
	cudaMalloc((void**)&voltage3D, memSize * sizeof(float));
	cudaMemset(voltage3D, 0, memSize * sizeof(float));
	cudaMemcpy(voltage3D, m_d_VFInit, memSize * sizeof(float), cudaMemcpyDeviceToDevice);
	// backup버퍼는 0으로 초기화합니다.
	cudaMalloc((void**)&voltage3D_backup, memSize * sizeof(float));
	cudaMemset(voltage3D_backup, 0, memSize * sizeof(float));
	// vector fiedls의 w 성분은 모두 1로 초기화 합니다.
	cudaMalloc((void**)&vectorFields, memSize * sizeof(float4));
	cudaMemset(vectorFields, 0, memSize * sizeof(float4));
	cufloat4_memset_by_value(m_gridSize, m_blockSize, uVoxSize, vectorFields, 3, 1);

	// Laplace 적용전 출력 Debug
	/*exportBMP(voltage3D, uVoxSize, m_voxel_size, m_gridSize, m_blockSize, nullptr);
	copy_uint16_to_float(m_gridSize, m_blockSize, uVoxSize, voltage3D_backup, m_d_WallMask);
	exportBMP(voltage3D_backup, uVoxSize, m_voxel_size, m_gridSize, m_blockSize, "Mask");
	cudaMemset(voltage3D_backup, 0, memSize * sizeof(float));*/

	// Laplace equation
	computeLaplaceEquation_with_Vector(m_gridSize, m_blockSize, 400, uVoxSize,
		voltage3D,
		voltage3D_backup,
		vectorFields,
		m_d_WallMask);

#ifdef PRINT_BMP
	exportBMP(m_save_path.c_str(), voltage3D_backup, uVoxSize, m_voxel_spacing, m_gridSize, m_blockSize, "VF");
#endif

	/*float4* hhh1 = new float4[memSize];
	cudaMemcpy(hhh1, vectorFields, memSize * sizeof(float4), cudaMemcpyDeviceToHost);*/

	cudaFree(voltage3D_backup);
	cudaFree(voltage3D);

	/* Euler's methods로 출발할 정점위치 표시 (출발정점은 (endo 또는 epi 입니다.)
	출발정점 coordinate는 voxel의 좌표계 입니다. */
	float* h_vectorfields = new float[memSize];
	cudaMemcpy(h_vectorfields, m_d_vectorfields, memSize * sizeof(float), cudaMemcpyDeviceToHost);
	float4* endoVertices = new float4[memSize], *epiVertices = new float4[memSize];
	int endoVCnt = 0, epiVCnt = 0;
	for (int i = 0; i < memSize; i++) {
		int x = i % uVoxSize.x;
		int y = (i / uVoxSize.x) % uVoxSize.y;
		int z = i / (uVoxSize.x*uVoxSize.y);
		if (h_vectorfields[i] == 1.0) {
			endoVertices[endoVCnt] = make_float4(x, y, z, 0);
			endoVCnt++;
		}
		if (h_vectorfields[i] == 3.0) {
			epiVertices[epiVCnt] = make_float4(x, y, z, 0);
			epiVCnt++;
		}
		/*if (h_vectorfields[i] > 0)
			std::cout << i << ", " << h_vectorfields[i] << std::endl;*/
	}
	//delete[] hhh1;

	std::cerr << "total_endoVCnt_size = " << endoVCnt << ", " << epiVCnt << std::endl;
	delete[] h_vectorfields;
	realloc(endoVertices, endoVCnt * sizeof(float4));
	realloc(epiVertices, epiVCnt * sizeof(float4));

	// compute thickness
	if (m_d_normal != nullptr) {
		cudaFree(m_d_normal);
		m_d_normal = nullptr;
	}
	cudaMalloc((void**)&m_d_normal, endoVCnt * sizeof(float3));
	cudaMemset(m_d_normal, 0, endoVCnt * sizeof(float3));

	dim3 blockSize = dim3(256, 1, 1);
	dim3 gridSize = dim3(iDivUp(endoVCnt, blockSize.x), 1, 1);

	//m_voxel_spacing = glm::vec3(0.476, 0.476, 0.5);
	//// 5 patient
	//m_voxel_spacing = glm::vec3(0.412109, 0.412109, 0.5);
	//m_voxel_spacing = glm::vec3(0.463, 0.463, 0.5);
	//m_voxel_size = glm::vec3(0.359375, 0.359375, 0.625006);
	//m_voxel_size = glm::vec3(0.349609, 0.349609, 0.699951);
	//m_voxel_size = glm::vec3(0.46, 0.46, 0.5);
	//m_voxel_size = glm::vec3(0.462891, 0.462891, 0.499994);
	//m_voxel_size = glm::vec3(0.567, 0.567, 0.5);
	//m_voxel_size = glm::vec3(0.361328, 0.361328, 0.625002);
	
	float4* d_endoVertices = nullptr, *d_epiVertices = nullptr;
	cudaMalloc((void**)&d_endoVertices, endoVCnt * sizeof(float4));
	cudaMemcpy(d_endoVertices, endoVertices, endoVCnt * sizeof(float4), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_epiVertices, epiVCnt * sizeof(float4));
	cudaMemcpy(d_epiVertices, epiVertices, epiVCnt * sizeof(float4), cudaMemcpyHostToDevice);

	compute_thickness(gridSize, blockSize, uVoxSize, m_d_WallMask, vectorFields, 0, d_endoVertices, m_d_normal, endoVCnt, make_float3(m_voxel_spacing.x, m_voxel_spacing.y, m_voxel_spacing.z));
	compute_thickness(gridSize, blockSize, uVoxSize, m_d_WallMask, vectorFields, 0, d_epiVertices, m_d_normal, epiVCnt, make_float3(m_voxel_spacing.x, m_voxel_spacing.y, m_voxel_spacing.z));
	
	cudaMemcpy(endoVertices, d_endoVertices, endoVCnt * sizeof(float4), cudaMemcpyDeviceToHost);
	cudaMemcpy(epiVertices, d_epiVertices, epiVCnt * sizeof(float4), cudaMemcpyDeviceToHost);
	// Memcpy device thick to host thick
	float3* m_h_normal = new float3[endoVCnt];
	cudaMemcpy(m_h_normal, m_d_normal, endoVCnt * sizeof(float3), cudaMemcpyDeviceToHost);
	
	savePLT(m_save_path + "\\endo", endoVertices, endoVCnt, &m_endo_vertices_list);
	savePLT(m_save_path + "\\epi", epiVertices, epiVCnt);

	cudaFree(d_endoVertices);
	cudaFree(d_epiVertices);
	
	cudaFree(vectorFields);
	free(endoVertices);
	free(epiVertices);
}

void WT::savePLT(std::string fname, float4* vertices, int elemCnt, std::vector<float4>* vertices_list)
{
	using namespace std;
	/*string fileName = "Thickness_";
	fileName.append(patientName).append("-").append(patientID.c_str()).append(".plt");*/
	ofstream thicknessFile;
	thicknessFile.open(fname.append(".plt"), ios::trunc);
	stringstream buffer;

	std::cout << "m_volume_size = " << m_volume_size.x << ", " << m_volume_size.y << ", " << m_volume_size.z << std::endl;
	std::cout << "m_voxel_spacing = " << m_voxel_spacing.x << ", " << m_voxel_spacing.y << ", " << m_voxel_spacing.z << std::endl;
	std::cout << "m_volume_position = " << m_volume_position.x << ", " << m_volume_position.y << ", " << m_volume_position.z << std::endl;

	buffer << "VARIABLES = \"X\", \"Y\", \"Z\", \"Thickness(mm)\"\n";
	buffer << "ZONE I=" << elemCnt << " , DATAPACKING=POINT\n";

	float averageWT = 0.0f;
	for (int i = 0; i < elemCnt; i++) {
		float4 vertex = vertices[i];
		
		// 정규화 (0 ~ 1)
		vertex = make_float4(make_float3(vertex) / make_float3(m_volume_size.x, m_volume_size.y, m_volume_size.z), vertex.w);

		vertex = make_float4(vertex.x * (m_voxel_spacing.x * m_volume_size.x),
			(vertex.y) * (m_voxel_spacing.y * m_volume_size.y),
			(1.0f - vertex.z) * (m_voxel_spacing.z * m_volume_size.z),
			vertex.w);
		vertex = make_float4(make_float3(m_volume_position.x, m_volume_position.y, m_volume_position.z) - 
			make_float3(0, 0, (m_volume_size.z * m_voxel_spacing.z))
			+ make_float3(vertex), vertex.w);

		//vertex = make_float4(make_float3(m_volume_position.x, m_volume_position.y, m_volume_position.z)
		//	- make_float3(0, 0, (m_volume_size.z * m_voxel_spacing.z)) + make_float3(vertex), vertex.w);

		buffer << vertex.x << " " << vertex.y << " " << vertex.z << " " << vertex.w << "\n";
		averageWT += vertex.w;

		if (vertices_list != nullptr)
			vertices_list->push_back(vertex);
	}

	std::cout << fname << "-average WT = " << (float)(averageWT/elemCnt) << std::endl;

	thicknessFile << buffer.rdbuf();
	thicknessFile.close();
}