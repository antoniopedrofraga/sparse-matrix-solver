#ifndef IO_MANAGER_H
#define IO_MANAGER_H

#include "../matrix/csr.h"
#include "../matrix/ellpack.h"
#include <string>

using namespace std;

class IOmanager {
	string filename;
public:
	IOmanager();
	std::string parseArguments(int argc, char ** argv);
	std::pair<CSR*, Ellpack*> readFile(string filename);
};

#endif
