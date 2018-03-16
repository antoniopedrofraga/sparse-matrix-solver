#ifndef IO_MANAGER_H
#define IO_MANAGER_H

#include "../matrix/csr.h"
#include "../matrix/ellpack.h"
#include "../utils/utils.h"
#include <string>
#include <fstream>

using namespace std;

class IOmanager {
	string filename;

	std::string extractName(std::string path);
public:
	IOmanager();
	std::string parseArguments(int argc, char ** argv);
	std::pair<CSR*, Ellpack*> readFile(string filename);
	void exportResults(std::string output_file, std::string path, CSR * csr, Ellpack * ellpack);
};

#endif
