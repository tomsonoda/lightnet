#include <iostream>

using namespace std;
class ArgumentProcessor
{
private:
public:
  ArgumentProcessor();
  ~ArgumentProcessor();
  float get_float_parameter(int argc, char **argv, string name, float default_value);
  string get_chars_parameter(int argc, char **argv, string name, string default_value);
};
