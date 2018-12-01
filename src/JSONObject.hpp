#include <stddef.h>
#include <vector>
#include <map>
#include <variant>

using namespace std;

class JSONObject
{
public:
  typedef enum {
    JSON_TYPE_NONE = 0,
    JSON_TYPE_DICTIONARY,
    JSON_TYPE_ARRAY,
    JSON_TYPE_STRING,
    JSON_TYPE_BASIC
  } json_type_t;

  typedef struct {
    int index;
    int parent_token;
    json_type_t type;
    int start;
    int end;
    int size;
  } json_token_t;

  JSONObject();
  ~JSONObject();

  std::string org_string;
  std::vector <json_token_t*> tokens;
  std::vector<json_token_t*> load(const std::string file_path);
  std::vector <json_token_t*> parse(const std::string json_string);
  std::vector<json_token_t*> getChildrenForToken(json_token_t* parent);
  json_token_t* getChildForToken(json_token_t* parent, std::string key);
  std::string getChildValueForToken(json_token_t* parent, std::string key);
  std::string getValueForToken(json_token_t* parent);
  std::vector<json_token_t *>getArrayForToken(json_token_t* parent, std::string key);
  void showTokens(std::vector<json_token_t *> tokens);

private:
  enum json_errors {
  	JSON_ERROR_NOT_GENERATED = -100,
  	JSON_ERROR_INVALID_SYNTACS = -110,
  	JSON_ERROR_INVALID_UNICODE_SYNTACS = -111,
  	JSON_ERROR_CORRUPTED_DATA = -200
  };

  typedef struct {
    int parent_token;
  	unsigned int pos;
  	unsigned int next_token;
  	unsigned int max_tokens;
  } json_parser;

  void clean();
  char *readFileToChars(string filename);
  std::string readFileToString(std::string filename);
  void assignToken(json_parser *parser, json_token_t* token);
  int parseNumbersBools(json_parser *parser, std::string json_string);
  int parseString(json_parser *parser, std::string json_string);
  std::string getTokenTypeLabel(json_type_t type);
};
