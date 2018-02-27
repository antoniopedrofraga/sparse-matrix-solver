#ifndef IO_MANAGER_H
#define IO_MANAGER_H

#include "../matrix/matrix.h"
#include <string>

using namespace std;

class IOmanager {
	string filename;
public:
	IOmanager();
	Matrix * readFile(string filename);
};

#endif
