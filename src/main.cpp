#include "Ext_headers.h"
#include "MarchingCubes.h"
#include "util.h"
#include "WT.h"
#include "QHull.h"

uint16* g_WallMask = nullptr;
uint16* g_ConvexMask = nullptr;

glm::u16vec4* g_display_data = nullptr;
glm::vec3 g_pixel_spacing = glm::vec3(1.0);

glm::ivec4 volume_size;
glm::vec3 volume_center = glm::vec3(0.0);
glm::ivec2 g_screenSize;

MarchingCube* surfaces = nullptr;

char* argv_file_path = nullptr;

void read_binary(char* file_path = nullptr) {
	//byte* binary_volume = new byte[512 * 512 * 381];
	using namespace std;
	ifstream fin;
	if (file_path == nullptr || strlen(file_path) < 1)
		fin.open("D:/Kwon/Lab/Develop/Source/Python/Computer Vision/LA segmentaion/contourVol.bin", ios::binary);
	else {
		fin.open(file_path, ios::binary);
		cerr << file_path << endl;
	}

	if (!fin.is_open())
		cerr << "파일 오픈 실패\n";

	uint32 w, h, d;
	fin.read((char*)&w, sizeof(uint32));
	fin.read((char*)&h, sizeof(uint32));
	fin.read((char*)&d, sizeof(uint32));

	fin.read((char*)&g_pixel_spacing.x, sizeof(float));
	fin.read((char*)&g_pixel_spacing.y, sizeof(float));
	fin.read((char*)&g_pixel_spacing.z, sizeof(float));

	fin.read((char*)&volume_center.x, sizeof(float));
	fin.read((char*)&volume_center.y, sizeof(float));
	fin.read((char*)&volume_center.z, sizeof(float));

	volume_size = glm::ivec4(w, h, d, 1);
	g_pixel_spacing.z = 0.5f;
	cerr << volume_size.x << ", " << volume_size.y << ", " << volume_size.z << endl;
	cerr << g_pixel_spacing.x << ", " << g_pixel_spacing.y << ", " << g_pixel_spacing.z << endl;
	cerr << volume_center.x << ", " << volume_center.y << ", " << volume_center.z << endl;

	byte* binary_volume = new byte[volume_size.x*2 * volume_size.y * volume_size.z];
	
	fin.read((char*)binary_volume, volume_size.x*2 * volume_size.y * volume_size.z);

	fin.close();

	if (g_WallMask == nullptr)
		g_WallMask = new uint16[volume_size.x * volume_size.y * volume_size.z * volume_size.w];
	else {
		delete[] g_WallMask;
		g_WallMask = new uint16[volume_size.x * volume_size.y * volume_size.z * volume_size.w];
	}
	memset(g_WallMask, 0x00, volume_size.x * volume_size.y * volume_size.z * sizeof(uint16));

	for (int z = 0; z < volume_size.z; z++) {
		for (int y = 0; y < volume_size.y; y++) {
			for (int x = 0; x < volume_size.x; x++) {
				int idx		  = z * volume_size.x*2 * volume_size.y + y * volume_size.x*2 + x*2;
				int idx_write = (z) * volume_size.x * volume_size.y + y * volume_size.x + x;
				//g_volume_data[idx_write] = binary_volume[idx] > 0 ? 1 : 0;
				/*if (binary_volume[idx] > 0)
					g_volume_data[idx_write] = 1;
				else
					g_volume_data[idx_write] = 0;*/

				byte item1 = binary_volume[idx + 0];
				byte item2 = binary_volume[idx + 1];

				uint16 item = item1 % 0x10 + item2;
				g_WallMask[idx_write] = item > 0 ? 1 : 0;
			}
		}
	}
	delete[] binary_volume;

	cerr << "read binary files \n";
}


int main()
{
	//초기화
	/* 파일 오픈 */
	std::vector<std::string> file_names;
	std::string fpath = InitializerPath(file_names, L"Wall mask 파일 선택 (BMP)", L".bmp");
	volume_size = readBMPFiles(file_names, &g_WallMask);

#define GET_DCM_RESOLUTION
#ifdef GET_DCM_RESOLUTION
	glm::vec3 volume_position; std::string patientID;
	InitializerPath(file_names, L"DICOM 파일 선택 pixel spacing (dcm)", L".dcm");
	readDCMFiles_for_pixelSpacing(file_names, &g_pixel_spacing, &volume_position, nullptr, &patientID);
	std::cout << "voxel spacing = " << g_pixel_spacing.x << ", " << g_pixel_spacing.y << ", " << g_pixel_spacing.z << std::endl;
	std::cout << "volume_position = " << volume_position.x << ", " << volume_position.y << ", " << volume_position.z << std::endl;
#else
#endif

	clock_t st_time, ed_time;
	float calc_time = 0.0f;

	auto ret = true;
	if (ret == TRUE) {
		uint16* merged_hull = new uint16[volume_size.x * volume_size.y * volume_size.z];
		memset(merged_hull, 0, sizeof(uint16)*volume_size.x * volume_size.y * volume_size.z);

		uint16* hull_mask = new uint16[volume_size.x * volume_size.y * volume_size.z];
		memset(hull_mask, 0, sizeof(uint16)*volume_size.x * volume_size.y * volume_size.z);


		//! Generate and obtain Points
		double sum_time = 0;
		//! axial
		for (int i = 0; i < volume_size.z; i++) {
			auto st = clock();
			QHull quickHull;
			quickHull.setPointCloud(g_WallMask, volume_size, i, CTview::axial);
			if (quickHull.getPointCloud().size() < 1)
				continue;

			//! Generate and obtain boundary points
			quickHull.initialize();
			vector<glm::vec2> convex_points = quickHull.getDrawablePoints();

			auto ed = clock();
			double dt = (ed - st);
			sum_time += dt;
#ifdef _DEBUG
			std::cout << "idx: " << i << ", convex-point: " << convex_points.size() << ", dt: " << dt << "ms" << std::endl;
			for (int j = 0; j < convex_points.size(); j++) {
				glm::vec2 pt1 = convex_points[j];
				merged_hull[i*volume_size.x*volume_size.y + (int)pt1.y*volume_size.x + (int)pt1.x] = j;
			}
			exportBMP(merged_hull, volume_size, "axial2");
#endif _DEBUG
			computeFillSpace(merged_hull, convex_points, i, volume_size, CTview::axial);
		}
		std::cout << "acc_time: " << sum_time << std::endl;
		exportBMP(merged_hull, volume_size, "axial");

		//! coronal
		sum_time = 0;
		for (int i = 0; i < volume_size.x; i++) {
			auto st = clock();
			QHull quickHull;
			quickHull.setPointCloud(g_WallMask, volume_size, i, CTview::sagittal);
			if (quickHull.getPointCloud().size() < 1)
				continue;

			//! Generate and obtain boundary points
			quickHull.initialize();
			vector<glm::vec2> convex_points = quickHull.getDrawablePoints();

			auto ed = clock();
			double dt = (ed - st);
			sum_time += dt;
#ifdef _DEBUG
			std::cout << "idx: " << i << ", convex-point: " << convex_points.size() << ", dt: " << dt << "ms" << std::endl;
			for (int j = 0; j < convex_points.size(); j++) {
				glm::vec2 pt = convex_points[j];
				merged_hull[(int)pt.x*volume_size.x*volume_size.y + (int)pt.y*volume_size.x + i] = 1;
			}
			exportBMP(merged_hull, volume_size, "sagittal");
#endif _DEBUG
			computeFillSpace(hull_mask, convex_points, i, volume_size, CTview::sagittal);
		}
		arr_logical_and(merged_hull, hull_mask, volume_size);
		std::cout << "acc_time: " << sum_time << std::endl;

		//! coronal
		sum_time = 0;
		memset(hull_mask, 0, sizeof(uint16)*volume_size.x * volume_size.y * volume_size.z);
		for (int i = 0; i < volume_size.y; i++) {
			auto st = clock();
			QHull quickHull;
			quickHull.setPointCloud(g_WallMask, volume_size, i, CTview::coronal);
			if (quickHull.getPointCloud().size() < 1)
				continue;

			//! Generate and obtain boundary points
			quickHull.initialize();
			vector<glm::vec2> convex_points = quickHull.getDrawablePoints();

			auto ed = clock();
			double dt = (ed - st);
			sum_time += dt;
#ifdef _DEBUG
			std::cout << "idx: " << i << ", convex-point: " << convex_points.size() << ", dt: " << dt << "ms" << std::endl;
			for (int j = 0; j < convex_points.size(); j++) {
				glm::vec2 pt = convex_points[j];
				merged_hull[(int)pt.y*volume_size.x*volume_size.y + i*volume_size.x + (int)pt.x] = 1;
			}
			exportBMP(merged_hull, volume_size, "coronal");
#endif
			computeFillSpace(hull_mask, convex_points, i, volume_size, CTview::coronal);
		}
		arr_logical_and(merged_hull, hull_mask, volume_size);
		std::cout << "acc_time: " << sum_time << std::endl;

		exportBMP(merged_hull, volume_size, "Merged-hull");
		delete[] hull_mask;
		//delete[] merged_hull;

		calc_time = 0.0f;
		st_time = clock();

		//! WT 계산
		CreateDirectoryA((fpath + "\\Results").c_str(), nullptr);
		WT* wt_algorithms = new WT(fpath + "\\Results", volume_size, g_pixel_spacing, volume_position, g_WallMask, merged_hull);

//#define GET_MAIN_VOLUME
#ifdef GET_MAIN_VOLUME
		// 임시 MainVolume 계산 (Seg3D로 가져옴)
		InitializerPath(file_names, L".bmp");
		readBMPFiles(file_names, &g_ConvexMask);
		wt_algorithms->detectEpiEndo(g_ConvexMask);
		// 임시 MainVolume 계산 (Seg3D로 가져옴)
#else
		wt_algorithms->detectEpiEndo(nullptr);
#endif
		ed_time = clock();
		calc_time = (ed_time - st_time);
		std::cerr << "Finished epi-endo calculation : " << calc_time << "ms (" << calc_time/1e3 << " s) \n";

		calc_time = 0.0f;
		st_time = clock();
		wt_algorithms->evalWT();
		ed_time = clock();
		calc_time = (ed_time - st_time);
		std::cerr << "Finished WT : " << calc_time << "ms (" << calc_time / 1e3 << " s) \n";

		memcpy(g_WallMask, wt_algorithms->getChamberMask(), sizeof(uint16)*volume_size.x*volume_size.y*volume_size.z);
		std::vector<float4> endo_vertices_list;
		endo_vertices_list.resize(wt_algorithms->m_endo_vertices_list.size());
		std::copy(wt_algorithms->m_endo_vertices_list.begin(), wt_algorithms->m_endo_vertices_list.end(), endo_vertices_list.begin());
		delete wt_algorithms;

		

		surfaces = new MarchingCube(fpath + "\\Results", g_WallMask, volume_size, (float*)&g_pixel_spacing, volume_position, MarchingCube::from_host);

		surfaces->computeISOsurface(g_WallMask, MarchingCube::from_host);
		surfaces->saveMeshInfo(fpath + "\\Results\\" + "WT(projected)-" + patientID, &endo_vertices_list);
		delete[] merged_hull;
	}

	return 0;
}
