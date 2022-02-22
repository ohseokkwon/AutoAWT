#pragma once
#include "Ext_headers.h"

bool sortFiles(std::wstring str1, std::wstring str2);
std::string InitializerPath(std::vector<std::string>& lists, wchar_t* title = L"Selection DICOM Path", wchar_t* ext = L"dcm");
void readDCMFiles_for_pixelSpacing(std::vector<std::string>& lists, glm::vec3* pixel_spacing, glm::vec3* volume_position, glm::vec4* volume_size, std::string* patientID);
void GetFileList_by_path(std::string f_path, std::vector<std::string>& lists, wchar_t* ext = L"dcm");

glm::ivec4 readBMPFiles(std::vector<std::string>& lists, uint16** volume, uint direction = 0);
void exportBMP(uint16* d_HU, glm::ivec4 windowSize, std::string fName);
void computeFillSpace(uint16* h_res_buffer, std::vector<glm::vec2> lineBuffer, uint idx, glm::vec3 volume_size = glm::vec3(0));