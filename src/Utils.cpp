#include <algorithm> // std::equal
#include <assert.h>
#include <vector>
#include <dirent.h>
#include <errno.h>
#include <fstream>
#include <iterator>  // std::rbegin, std::rend
#include <string>
#include <iostream>
#include <stdlib.h>
#include <sys/types.h>
#include "Utils.hpp"

using namespace std;

Utils::Utils()
{
}

Utils::~Utils()
{
}

void Utils::outputLog(const char *file, const char *func, int line, std::string message)
{
  printf("%s (%d) : %s() : %s\n", file, line, func, message.c_str());
}
void Utils::error(const char *s)
{
    perror(s);
    assert(0);
    exit(-1);
}

std::string Utils::stringReplace(std::string org_string, std::string from_string, std::string to_string)
{
    std::string::size_type  pos( org_string.find( from_string ) );
    while( pos != std::string::npos ){
        org_string.replace( pos, from_string.length(), to_string );
        pos = org_string.find( from_string, pos + to_string.length() );
    }
    return org_string;
}

bool Utils::stringEndsWith(const std::string &mainStr, const std::string &toMatch)
{
	if(mainStr.size() >= toMatch.size() &&
			mainStr.compare(mainStr.size() - toMatch.size(), toMatch.size(), toMatch) == 0){
    return true;
  }else{
    return false;
  }
}

bool Utils::fileExists(const std::string& str)
{
    std::ifstream ifs(str);
    return ifs.is_open();
}

int Utils::listDir (string dir, vector<string> &files, string ext)
{
    DIR *dp;
    struct dirent *dirp;
    if((dp  = opendir(dir.c_str())) == NULL) {
        cout << "Error(" << errno << ") opening " << dir << endl;
        return errno;
    }
    while ((dirp = readdir(dp)) != NULL) {
      if (stringEndsWith(string(dirp->d_name), ext)){
        files.push_back(string(dirp->d_name));
      }
    }
    closedir(dp);
    return 0;
}
