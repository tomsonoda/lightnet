#include <stdio.h>
#include <string>

using namespace std;

class Utils
{
private:
public:
  Utils();
  ~Utils();
  void outputLog(const char *file, const char *func, int line, std::string message);
  void error(const char *s);
  bool stringEndsWith(const std::string &mainStr, const std::string &toMatch);
  std::string stringReplace( std::string org_string, std::string from_string, std::string to_string);
  bool fileExists(const std::string& str);
  int listDir (string dir, vector<string> &files, string ext);
};
