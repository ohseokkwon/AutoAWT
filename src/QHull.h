#include "Ext_headers.h"

using namespace std;
class QHull
{
private:
	typedef unsigned short uint16;
	typedef unsigned int uint;

	//Random points
	vector<glm::vec2> pointCloud;
	//Final boundary Point
	vector<glm::vec2> hull;

public:
	void initialize();
	vector<glm::vec2> getPointCloud();
	vector<glm::vec2> getDrawablePoints();
	void generatePointCloud(int limit);
	void QHull::setPointCloud(uint16* volume_2d, glm::ivec4 volume_size, int idx, CTview view);
	int findSide(glm::vec2 p1, glm::vec2 p2, glm::vec2 p3);
	int lineDist(glm::vec2 p1, glm::vec2 p2, glm::vec2 p);
	void quickHull(vector<glm::vec2> a, int n, glm::vec2 p1, glm::vec2 p2, int side);
	void generateHull();

	void jarvisHull();
};
