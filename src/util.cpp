#include "util.h"


bool sortFiles(std::wstring str1, std::wstring str2)
{
	return (StrCmpLogicalW(str1.c_str(), str2.c_str()) == -1 ? true : false);
}

std::string InitializerPath(std::vector<std::string>& lists, wchar_t* title, wchar_t* ext)
{
	lists.clear();

	CoInitialize(0);
	IFileOpenDialog* dlg;
	CoCreateInstance(CLSID_FileOpenDialog, nullptr, CLSCTX_ALL, IID_IFileOpenDialog, reinterpret_cast<void**>(&dlg));
	DWORD dwOpt;
	dlg->SetTitle(title);
	if (SUCCEEDED(dlg->GetOptions(&dwOpt))) dlg->SetOptions(dwOpt | FOS_PICKFOLDERS);
	dlg->Show(0);

	IShellItem* item;// = make_unique<unique_ptr>();

	wchar_t *f_path = nullptr;

	if (dlg->GetResult(&item) == S_OK)
	{
		if (!SUCCEEDED(item->GetDisplayName(SIGDN_DESKTOPABSOLUTEPARSING, &f_path)))
		{
			perror("FAILED FOLDER SELECTION\n");
		}
		else
		{
			std::wstring pattern(f_path);
			pattern.append(L"\\*");
			WIN32_FIND_DATAW data;
			HANDLE hFind;
			int i = 0;
			if ((hFind = FindFirstFileW(pattern.c_str(), &data)) != INVALID_HANDLE_VALUE) {
				do {
					PWCHAR it = data.cFileName;
					if (wcsstr(it, ext) != nullptr)
					{
						std::wstring te(f_path);


						std::wstring temp = _wcsdup(std::wstring(te + L'\\' + data.cFileName).c_str());

						std::string str(temp.begin(), temp.end());
						//wstring temp = data.cFileName;
						lists.push_back(str);
						//dxlIdx.push_back(i);		
					}
				} while (FindNextFileW(hFind, &data) != 0);
				FindClose(hFind);
			}
		}
	}

	// 파일 선택 정렬 (selection sort)
	std::vector<std::wstring> tmp_wlist;
	for (auto i = 0; i < lists.size(); i++) {
		std::wstring unsorted_fname;
		unsorted_fname.assign(lists.at(i).begin(), lists.at(i).end());
		tmp_wlist.push_back(unsorted_fname);
	}
	std::sort(tmp_wlist.begin(), tmp_wlist.end(), sortFiles);
	lists.clear();
	for (auto i = 0; i < tmp_wlist.size(); i++) {
		std::string sorted_fname;
		sorted_fname.assign(tmp_wlist.at(i).begin(), tmp_wlist.at(i).end());
		lists.push_back(sorted_fname);
	}
	tmp_wlist.clear();

	CoUninitialize();
	std::cout << "Found " << lists.size() << " files\n";

	std::wstring ws(f_path);
	return std::string(ws.begin(), ws.end());
}

void readDCMFiles_for_pixelSpacing(std::vector<std::string>& lists, glm::vec3* pixel_spacing, glm::vec3* volume_position, glm::vec4* volume_size, std::string* patientID)
{
	DcmTagKey tagkeys[128]{
		DCM_Rows,
		DCM_Columns,
		DCM_SamplesPerPixel,
		DCM_PatientID,
	};

	if (lists.size() < 1) {
		std::cerr << "no have to DCM files" << std::endl;
		return;
	}

	// [ w, h, d, c ]
	if (volume_size != nullptr)
		*volume_size = glm::ivec4(0, 0, lists.size(), 0);

	Sint16 r_pvmin = 10000;
	Sint16 r_pvmax = -10000;
	Sint16 g_PVMin = 10000;
	Sint16 g_PVMax = -10000;

	bool once_volume_position = true;
	for (int i = 0; i < lists.size(); i++) {
		DcmFileFormat dcm_format;
		DcmElement* dcm_elem;

		OFCondition result = EC_Normal;
		result = dcm_format.loadFile(lists.at(i).c_str());
		if (result.bad()) {
			std::cerr << "dcm load error" << std::endl;
			continue;
		}

		DcmDataset *data = dcm_format.getDataset();
		if (data == nullptr) {
			std::cerr << "dcm data not exists" << std::endl;
			continue;
		}

		DcmTagKey key = DCM_Rows;
		bool haveTags = data->tagExists(key);
		Uint16 uint16_elem;
		if (haveTags) {

			result = data->findAndGetUint16(key, uint16_elem);
			if (result.bad())
				continue;

			if (volume_size != nullptr)
				(*volume_size).x = uint16_elem;
		}

		key = DCM_Columns;
		haveTags = data->tagExists(key);
		if (haveTags) {
			result = data->findAndGetUint16(key, uint16_elem);
			if (result.bad())
				continue;

			if (volume_size != nullptr)
				(*volume_size).y = uint16_elem;
		}

		key = DCM_SamplesPerPixel;
		haveTags = data->tagExists(key);
		if (haveTags) {
			result = data->findAndGetUint16(key, uint16_elem);
			if (result.bad())
				continue;

			if (volume_size != nullptr)
				(*volume_size).w = 1;
		}
		if (i == 0) {
			key = DCM_PatientID;
			haveTags = data->tagExists(key);
			if (haveTags) {
				OFString ofs_patientID;
				result = data->findAndGetOFString(key, ofs_patientID);
				if (result.bad())
					continue;

				*patientID = std::string(ofs_patientID.c_str());
				std::cout << "patientID: " << *patientID << std::endl;
			}
		}

		int bitCounts = 0;
		key = DCM_BitsAllocated;
		haveTags = data->tagExists(key);
		if (haveTags) {
			result = data->findAndGetUint16(key, uint16_elem);
			if (result.bad())
				continue;

			bitCounts = uint16_elem / 8;
		}

		key = DCM_PixelRepresentation;
		haveTags = data->tagExists(key);
		if (haveTags)
		{
			result = data->findAndGetElement(key, dcm_elem);
			Uint16 data = 0;
			result = dcm_elem->getUint16(data);

			//std::cout << "DCM_PixelRepresentation = " << data << std::endl;
		}

		// 가장 작은 픽셀값  (정규화 용)
		key = DCM_SmallestImagePixelValue;
		haveTags = data->tagExists(key);
		if (haveTags) {
			result = data->findAndGetElement(key, dcm_elem);
			Uint16 data = 0;
			result = dcm_elem->getUint16(data);

			//std::cout << "DCM_SmallestImagePixelValue = " << data << std::endl;
		}
		else {
		}

		// 가장 큰 픽셀값 (정규화 용)
		key = DCM_LargestImagePixelValue;
		haveTags = data->tagExists(key);
		if (haveTags) {
			result = data->findAndGetElement(key, dcm_elem);
			Uint16 data = 0;
			result = dcm_elem->getUint16(data);

			//std::cout << "DCM_LargestImagePixelValue = " << data << std::endl;
		}
		else {
		}
		// 
		key = DCM_RescaleSlope;
		haveTags = data->tagExists(key);
		if (haveTags) {
			result = data->findAndGetElement(key, dcm_elem);
			Uint16 data = 0;
			result = dcm_elem->getUint16(data);

			//std::cout << "DCM_RescaleSlope = " << data << std::endl;
		}
		key = DCM_PixelSpacing;
		haveTags = data->tagExists(key);
		if (haveTags) {
			const char* PP;
			result = data->findAndGetString(key, PP);
			if (result.bad())
				break;
			else
			{
				char* pppp = strdup(PP);
				char* ppppp = strtok(pppp, "\\\\");
				if (pixel_spacing != nullptr)
					(*pixel_spacing).x = atof(ppppp);
				ppppp = strtok(nullptr, "\\\\");
				if (pixel_spacing != nullptr)
					(*pixel_spacing).y = atof(ppppp);
				free(pppp);

				if (pixel_spacing != nullptr)
					(*pixel_spacing).z = 0.5f;

				/*std::cout << "DCM_PixelSpacing = " << g_pixel_spacing.x << ", " <<
				g_pixel_spacing.y << ", " << g_pixel_spacing.z << std::endl;*/
			}
		}

		// 좌표는 첫번째 DICOM 이미지의 DCM_ImagePositionPatient로 합니다.
		if (volume_position != nullptr && once_volume_position) {
			const char *chr;
			char *pc = nullptr, *pc2 = nullptr;
			result = data->findAndGetString(DCM_ImagePositionPatient, chr);
			if (result.bad())
				return;
			float3 z;
			char* du = strdup(chr);
			z.x = atof(strtok_s(du, "\\", &pc));
			z.y = atof(strtok_s(nullptr, "\\", &pc));
			z.z = atof(strtok_s(nullptr, "\\", &pc));


			(*volume_position).x = z.x;
			(*volume_position).y = z.y;
			(*volume_position).z = z.z;
			once_volume_position = false;
		}
	}

	if (pixel_spacing != nullptr) {
		OFCondition result;
		(*pixel_spacing).z = 0;
		float voxel_depth_size = (float)lists.size();
		for (int i = 0; i < lists.size(); i++) {
			DcmFileFormat ff2;
			DcmElement* ele2;
			ff2.loadFile(lists.at(i).c_str());
			DcmDataset *dat = ff2.getDataset();
			if (dat->tagExists(DCM_SliceLocation))
			{
				const char* dcm_data;
				result = dat->findAndGetString(DCM_SliceLocation, dcm_data);
				if (result.bad()) {
					std::cout << "Not find DCM_SliceLocation\n";
					return;
				}
				else
				{
					char* ptr = strdup(dcm_data);
					char* splitted_str = strtok(ptr, "\\\\");
					if (i == 0) {
						(*pixel_spacing).z = atof(splitted_str);
					}
					else {
						if (voxel_depth_size > atof(splitted_str) - (*pixel_spacing).z)
							voxel_depth_size = atof(splitted_str) - (*pixel_spacing).z;
						(*pixel_spacing).z = atof(splitted_str);
					}
					free(ptr);
				}
			}
		}
		(*pixel_spacing).z = abs(voxel_depth_size);
	}

	std::cout << "PVMin = " << g_PVMin << std::endl;
	std::cout << "PVMax = " << g_PVMax << std::endl;

	std::cout << "volume_position=" << (*volume_position).x << ", " << (*volume_position).y << ", " << (*volume_position).z << std::endl;
	std::cout << "pixel_spacing=" << (*pixel_spacing).x << ", " << (*pixel_spacing).y << ", " << (*pixel_spacing).z << std::endl;

	return;
}


void GetFileList_by_path(std::string f_path, std::vector<std::string>& lists, wchar_t* ext)
{
	lists.clear();

	std::wstring pattern(f_path.begin(), f_path.end());
	pattern.append(L"\\*");
	WIN32_FIND_DATAW data;
	HANDLE hFind;
	int i = 0;
	if ((hFind = FindFirstFileW(pattern.c_str(), &data)) != INVALID_HANDLE_VALUE) {
		do {
			PWCHAR it = data.cFileName;
			if (wcsstr(it, ext) != nullptr)
			{
				std::wstring te(f_path.begin(), f_path.end());


				std::wstring temp = _wcsdup(std::wstring(te + L'\\' + data.cFileName).c_str());

				std::string str(temp.begin(), temp.end());
				//wstring temp = data.cFileName;
				lists.push_back(str);
				//dxlIdx.push_back(i);		
			}
		} while (FindNextFileW(hFind, &data) != 0);
		FindClose(hFind);
	}

	// 파일 선택 정렬 (selection sort)
	std::vector<std::wstring> tmp_wlist;
	for (auto i = 0; i < lists.size(); i++) {
		std::wstring unsorted_fname;
		unsorted_fname.assign(lists.at(i).begin(), lists.at(i).end());
		tmp_wlist.push_back(unsorted_fname);
	}
	std::sort(tmp_wlist.begin(), tmp_wlist.end(), sortFiles);
	lists.clear();
	for (auto i = 0; i < tmp_wlist.size(); i++) {
		std::string sorted_fname;
		sorted_fname.assign(tmp_wlist.at(i).begin(), tmp_wlist.at(i).end());
		lists.push_back(sorted_fname);
	}
	tmp_wlist.clear();

	std::cout << "Found " << lists.size() << " files\n";
}

glm::ivec4 readBMPFiles(std::vector<std::string>& lists, uint16** volume, uint direction)
{
	if (lists.size() < 1) {
		std::cerr << "no have to BMP files" << std::endl;
		return glm::ivec4(0);
	}

	// [ w, h, d, c ]
	glm::ivec4 volume_size = glm::ivec4(0, 0, lists.size(), 1);
	bool once = true;
	int pitch = 0;
	int channel = 0;

	byte* raw_data = nullptr;
	uint16* data_ptr = nullptr;

	uint16* volume_internal = nullptr;
	uint16* volume_ptr = nullptr;
	for (int i = 0; i < lists.size(); i++) {
		// BMP header
		BITMAPINFOHEADER bi;
		BITMAPFILEHEADER bf;
		RGBQUAD palette[256];

		FILE *fp;
		fopen_s(&fp, lists.at(i).c_str(), "rb");

		fread(&bf, sizeof(BITMAPFILEHEADER), 1, fp);
		fread(&bi, sizeof(BITMAPINFOHEADER), 1, fp);
		//fread(&palette, sizeof(RGBQUAD) * 256, 1, fp);
		fseek(fp, bf.bfOffBits, 0);

		if (once) {
			volume_size.x = bi.biWidth;
			volume_size.y = bi.biHeight;
			channel = bi.biBitCount / 8;

			once = false;

			volume_internal = new uint16[volume_size.x * volume_size.y * lists.size()];
			volume_ptr = volume_internal;

			raw_data = new byte[volume_size.x * volume_size.y * channel];
			data_ptr = new uint16[volume_size.x * volume_size.y];
		}


		fread(raw_data, volume_size.x * volume_size.y * channel, 1, fp);
		fclose(fp);

		// Y 축만 변환
		if (channel == 3) {
			for (int h = 0; h < volume_size.y; h++)
			{
				for (int w = 0; w < volume_size.x; w++) {
					int idx_in = h*volume_size.x + w;
					int idx_out = (volume_size.y - 1 - h)*volume_size.x + w;

					byte r = raw_data[idx_in * 3 + 0];
					byte g = raw_data[idx_in * 3 + 1];
					byte b = raw_data[idx_in * 3 + 2];

					if (r == g && g == b && r == b)
						data_ptr[idx_out] = 0;
					else
						data_ptr[idx_out] = 255;
				}
			}
		}
		else if (channel == 1) {
			for (int h = 0; h < volume_size.y; h++)
			{
				for (int w = 0; w < volume_size.x; w++) {
					int idx_in = h*volume_size.x + w;
					int idx_out = (volume_size.y - 1 - h)*volume_size.x + w;
					byte intensity = raw_data[idx_in];

					if (intensity > 0)
						data_ptr[idx_out] = 255;
					else
						data_ptr[idx_out] = 0;
				}
			}
		}

		memcpy(volume_ptr + pitch, data_ptr, sizeof(uint16) * volume_size.x * volume_size.y);
		pitch += volume_size.x * volume_size.y;
	}

	delete[] raw_data;
	delete[] data_ptr;

	if (*volume == nullptr) {
		*volume = new uint16[volume_size.x * volume_size.y * volume_size.z];
		memset(*volume, 0x00, volume_size.x * volume_size.y * volume_size.z * sizeof(uint16));
		memcpy(*volume, volume_internal, sizeof(uint16) * volume_size.x * volume_size.y * volume_size.z);
	}
	else {
		memcpy(*volume, volume_internal, sizeof(uint16) * volume_size.x * volume_size.y * volume_size.z);
	}


	delete[] volume_internal;

	return volume_size;
}

void exportBMP(uint16* d_HU, glm::ivec4 windowSize, std::string fName)
{
	auto save_path = (std::string("../BMP-") + fName);
	CreateDirectoryA(save_path.c_str(), nullptr);
	int memSize = windowSize.x * windowSize.y * sizeof(uint16);
	int channel = 1;


	uint16* h_slice = new uint16[windowSize.x * windowSize.y];
	byte* h_slice_mask = new byte[windowSize.x * windowSize.y];

	for (int z = 0; z < windowSize.z; z++) {
		std::cout << "DCM row = " << z + 1 << std::endl;

		try {
			memcpy(h_slice, d_HU + z * windowSize.x * windowSize.y, sizeof(uint16)*windowSize.x * windowSize.y);
			memset(h_slice_mask, 0x00, sizeof(byte)*windowSize.x * windowSize.y);
			for (int i = 0; i <windowSize.x * windowSize.y; i++) {
				h_slice_mask[i] = h_slice[i];
			}

			// 저장폴더 지정
			char saveFileName[128] = { 0 };
			sprintf_s(saveFileName, "%s/%d.bmp", save_path.c_str(), z + 1);

			FILE *fp;
			fopen_s(&fp, saveFileName, "wb");

			// BMP header
			BITMAPINFOHEADER bi;
			BITMAPFILEHEADER bf;

			bf.bfType = 0x4D42;
			bf.bfOffBits = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER) + sizeof(RGBQUAD) * 256;
			bf.bfSize = windowSize.x * windowSize.y * channel + bf.bfOffBits;
			bf.bfReserved1 = bf.bfReserved2 = 0;

			bi.biSize = sizeof(BITMAPINFOHEADER);
			bi.biWidth = windowSize.x;
			bi.biHeight = -windowSize.y;
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
		}
		catch (std::exception& e) {
			std::cout << "exeption : " << e.what();
		}
		// MEM release
	}
	delete[] h_slice_mask;
	delete[] h_slice;
}


void computeFillSpace(uint16* h_res_buffer, std::vector<glm::vec2> lineBuffer, uint idx, glm::vec3 volume_size)
{
	// 2D Polygon Fill
	dim3 fill_blockSize = dim3(32, 16, 1);
	dim3 fill_gridSize = dim3(iDivUp(volume_size.x, fill_blockSize.x), iDivUp(volume_size.y, fill_blockSize.y), 1);
	uint3 grid_size = make_uint3(volume_size.x, volume_size.y, volume_size.z);
	float2* d_contour = nullptr;
	uint16* d_res_buffer = nullptr;
	cudaMalloc((void**)&d_res_buffer, grid_size.x*grid_size.y*grid_size.z * sizeof(uint16));
	cudaMemcpy(d_res_buffer, h_res_buffer, grid_size.x*grid_size.y*grid_size.z * sizeof(uint16), cudaMemcpyHostToDevice);


	uint32 line_size = lineBuffer.size();
	if (0 < line_size)
	{
		float2* h_contour = new float2[line_size];
		for (int i = 0; i < line_size; i++) {
			auto pt = lineBuffer[i];
			h_contour[i] = make_float2(pt.x, pt.y);
		}
		checkCudaErrors(cudaMalloc((void**)&d_contour, line_size * sizeof(float2)));
		checkCudaErrors(cudaMemcpy(d_contour, h_contour, line_size * sizeof(float2), cudaMemcpyHostToDevice));

		launch_polygon_fill_2D(fill_gridSize, fill_blockSize, d_res_buffer, idx, grid_size, d_contour, line_size);

		delete[] h_contour;
		cudaFree(d_contour);
	}
	
	fill_blockSize = dim3(32, 16, 2);
	fill_gridSize = dim3(iDivUp(volume_size.x, fill_blockSize.x), iDivUp(volume_size.y, fill_blockSize.y), iDivUp(1 << (int)ceil(log2((float)volume_size.z*0.5f)), fill_blockSize.z));

	/*std::cerr << fill_gridSize.x << ", " << fill_gridSize.y << ", " << fill_gridSize.z << std::endl;
	launch_inverse_depth_volume(fill_gridSize, fill_blockSize, (uint16*)res_buffer, res_buffer, grid_size, (uint32)(grid_size.z*0.5f));*/

	cudaMemcpy(h_res_buffer, d_res_buffer, grid_size.x*grid_size.y*grid_size.z * sizeof(uint16), cudaMemcpyDeviceToHost);
	cudaFree(d_res_buffer);
}