#include "QHull.h"

void QHull::initialize()
{
    hull.clear();

	//generateHull();

	jarvisHull();
}

vector<glm::vec2> QHull::getPointCloud()
{
	return pointCloud;
}


vector<glm::vec2> QHull::getDrawablePoints()
{
    return hull;
}

void QHull::setPointCloud(uint16* volume_2d, glm::ivec4 volume_size, int idx, CTview view)
{
	pointCloud.clear();
	if (view == CTview::axial) {
		for (int y = 0; y < volume_size.y; y++) {
			for (int x = 0; x < volume_size.x; x++) {
				if (0 < volume_2d[idx*volume_size.x*volume_size.y + y * volume_size.x + x])
					pointCloud.push_back(glm::vec2(x, y));
			}
		}
	}
	else if (view == CTview::sagittal) {
		for (int y = 0; y < volume_size.y; y++) {
			for (int z = 0; z < volume_size.z; z++) {
				if (0 < volume_2d[z*volume_size.x*volume_size.y + y * volume_size.x + idx])
					pointCloud.push_back(glm::vec2(z, y));
			}
		}
	}
	else if (view == CTview::coronal) {
		for (int z = 0; z < volume_size.z; z++) {
			for (int x = 0; x < volume_size.x; x++) {
				if (0 < volume_2d[z*volume_size.x*volume_size.y + idx * volume_size.x + x])
					pointCloud.push_back(glm::vec2(x, z));
			}
		}
	}

#ifdef _DEBUG
	std::cout << "size: " << pointCloud.size() << std::endl;
#endif
}

void QHull::generatePointCloud(int limit)
{
    pointCloud.clear();

    for(int i=0;i<limit;i++)
    {
            srand (i*time(NULL));
            float randX = (rand()%350);
            float randY = (rand()%350);

            pointCloud.push_back(glm::vec2(randX,randY));
    }
}


int QHull::findSide(glm::vec2 p1, glm::vec2 p2, glm::vec2 p3)
{
	int val = (p3.y - p1.y) * (p2.x - p1.x) -
		(p2.y - p1.y) * (p3.x - p1.x);

	if (val > 0)
		return 1;
	if (val < 0)
		return -1;
	return 0;
}

int QHull::lineDist(glm::vec2 p1, glm::vec2 p2, glm::vec2 p)
{
	return abs((p.y - p1.y) * (p2.x - p1.x) -
		(p2.y - p1.y) * (p.x - p1.x));
}

void QHull::quickHull(vector<glm::vec2> a, int n, glm::vec2 p1, glm::vec2 p2, int side)
{
	int ind = -1;
	int max_dist = 0;

	// finding the point with maximum distance 
	// from L and also on the specified side of L. 
	for (int i = 0; i < n; i++)
	{
		int temp = lineDist(p1, p2, a[i]);
		if (findSide(p1, p2, a[i]) == side && temp > max_dist)
		{
			ind = i;
			max_dist = temp;
		}
	}

	// If no point is found, add the end points 
	// of L to the convex hull. 
	if (ind == -1)
	{
		hull.push_back(p1);
		hull.push_back(p2);
		return;
	}

	// Recur for the two parts divided by a[ind] 
	quickHull(a, n, a[ind], p1, -findSide(a[ind], p1, p2));
	quickHull(a, n, a[ind], p2, -findSide(a[ind], p2, p1));
}

void QHull::generateHull()
{
	int n = pointCloud.size();
	auto a = pointCloud;
	// a[i].second -> y-coordinate of the ith point 
	if (n < 3)
	{
		cout << "Convex hull not possible\n";
		return;
	}

	// Finding the point with minimum and 
	// maximum x-coordinate 
	int min_x = 0, max_x = 0;
	for (int i = 1; i < n; i++)
	{
		if (a[i].x < a[min_x].x)
			min_x = i;
		if (a[i].x > a[max_x].x)
			max_x = i;
	}

	// Recursively find convex hull points on 
	// one side of line joining a[min_x] and 
	// a[max_x] 
	quickHull(a, n, a[min_x], a[max_x], 1);

	// Recursively find convex hull points on 
	// other side of line joining a[min_x] and 
	// a[max_x] 
	quickHull(a, n, a[min_x], a[max_x], -1);

#ifdef _DEBUG
	cout << "The points in Convex Hull are:\n";
#endif
}

bool ccw(glm::vec2 O, glm::vec2 A, glm::vec2 B) {
	// OA X OB < 0 => clockwise
	if ((A.x - O.x) * (B.y - O.y) >= (A.y - O.y)* (B.x - O.x)) return true;
	else return false;
}

int orientation(glm::vec2 p, glm::vec2 q, glm::vec2 r)
{
	int val = (q.y - p.y) * (r.x - q.x) -
		(q.x - p.x) * (r.y - q.y);

	if (val == 0) return 0;  // collinear
	return (val > 0) ? 1 : 2; // clock or counterclock wise
}

void QHull::jarvisHull()
{
	int n = pointCloud.size();
	auto a = pointCloud;
	if (n < 3)
	{
		cout << "Convex hull not possible\n";
		return;
	}

	// Find the leftmost point
	int l = 0;
	for (int i = 1; i < n; i++)
		if (a[i].x < a[l].x)
			l = i;

	// Start from leftmost point, keep moving counterclockwise
	// until reach the start point again.  This loop runs O(h)
	// times where h is number of points in result or output.
	int p = l, q;
	do
	{
		// Add current point to result
		hull.push_back(a[p]);

		// Search for a point 'q' such that orientation(p, q,
		// x) is counterclockwise for all points 'x'. The idea
		// is to keep track of last visited most counterclock-
		// wise point in q. If any point 'i' is more counterclock-
		// wise than q, then update q.
		q = (p + 1) % n;
		for (int i = 0; i < n; i++)
		{
			// If i is more counterclockwise than current q, then
			// update q
			if (orientation(a[p], a[i], a[q]) == 2)
				q = i;
		}

		// Now q is the most counterclockwise with respect to p
		// Set p as q for next iteration, so that q is added to
		// result 'hull'
		p = q;

	} while (p != l);  // While we don't come to first point
}