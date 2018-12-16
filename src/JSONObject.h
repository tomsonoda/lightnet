#pragma once
#include <stddef.h>
#include <vector>
#include <map>
#include <variant>

using namespace std;

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

struct JSONObject
{
public:
  std::string org_string;
  std::vector <json_token_t*> tokens;

  JSONObject()
  {
    clean();
  }

  ~JSONObject()
  {
    clean();
  }

  std::vector<json_token_t*> load(const std::string file_path)
  {
    std::string json_string = readFileToString(file_path);
    std::vector<json_token_t*> tokens = parse(json_string);
    getChildrenForToken(tokens[0]);
    getChildForToken(tokens[0], "net");
    return tokens;
  }

  std::vector<json_token_t*> parse(const std::string json_string)
  {
    this->org_string = json_string;
    json_parser parser;
  	parser.pos = 0;
  	parser.next_token = 0;
  	parser.parent_token = -1;

  	int token_count = parser.next_token;

  	for (; parser.pos < json_string.length() && json_string[parser.pos]!='\0'; parser.pos++) {
  		char ch = json_string[parser.pos];
  		switch (ch) {
  			case ' ': case '\t' : case '\r' : case '\n' :
  				break;
  			case '{': case '[':
        {
          token_count++;
          json_token_t *token = new json_token_t;
          assignToken(&parser, token);
          token->index = this->tokens.size();
          this->tokens.push_back(token);
  				if (parser.parent_token != -1) {
  					this->tokens[parser.parent_token]->size++;
  					token->parent_token = parser.parent_token;
  				}

  				if (ch == '{'){
  					token->type = JSON_TYPE_DICTIONARY;
  				}else{
  					token->type = JSON_TYPE_ARRAY;
  				}
  				token->start = parser.pos;
  				parser.parent_token = parser.next_token - 1;
        }
        break;

  			case '}': case ']':
        {
          json_type_t type;
  				if(ch=='}'){
  					type = JSON_TYPE_DICTIONARY;
  				}else{
  					type = JSON_TYPE_ARRAY;
  				}
  				if (parser.next_token < 1) {
            return this->tokens;
  				}
  				json_token_t *token = this->tokens[parser.next_token-1];

  				while (1) {
  					if (token->start != -1 && token->end == -1) {
  						if (token->type != type) {
                return this->tokens;
  						}
  						token->end = parser.pos + 1;
  						parser.parent_token = token->parent_token;
  						break;
  					}
  					if(token->parent_token==-1) {
  						if(token->type != type || parser.parent_token == -1) {
                return this->tokens;
  						}
  						break;
  					}
  					token = this->tokens[token->parent_token];
  				}
        }
  			break;

  			case '\"':
        {
          int ret = parseString(&parser, json_string);
  				if (ret < 0){
            return this->tokens;
  				}
  				token_count++;
  				if (parser.parent_token != -1){
  					this->tokens[parser.parent_token]->size++;
  				}
        }
  			break;

  			case ':':
  				parser.parent_token = parser.next_token - 1;
  				break;

  			case ',':
  				if (parser.parent_token != -1 &&
  						this->tokens[parser.parent_token]->type != JSON_TYPE_ARRAY &&
  						this->tokens[parser.parent_token]->type != JSON_TYPE_DICTIONARY) {
  					parser.parent_token = this->tokens[parser.parent_token]->parent_token;
  				}
  				break;

  			case 't': case 'f': case 'n' :
  			case '0': case '1' : case '2': case '3' : case '4': case '5': case '6': case '7' : case '8': case '9': case '-':
        {
          if (parser.parent_token != -1) {
  					json_token_t *t = this->tokens[parser.parent_token];
  					if ((t->type == JSON_TYPE_DICTIONARY) ||
  							(t->type == JSON_TYPE_STRING && t->size != 0)) {
                  return this->tokens;
  					}
  				}
  				int ret = parseNumbersBools(&parser, json_string);
  				if(ret<0){
            return this->tokens;
  				}
  				token_count++;
  				if (parser.parent_token != -1){
  					this->tokens[parser.parent_token]->size++;
  				}
        }
        break;
  			default:
          return this->tokens;
  			}
  	}

    for(int i=parser.next_token-1; i>=0; i--) {
      if (this->tokens[i]->start != -1 && this->tokens[i]->end == -1) {
        return this->tokens;
      }
    }
    return this->tokens;
  }

  std::vector<json_token_t*> getChildrenForToken(json_token_t* parent)
  {
    std::vector <json_token_t*> children;
    std::vector <json_token_t*> tokens = this->tokens;
    for(int i=0; i<tokens.size(); i++){
      if(tokens[i]->parent_token == parent->index){
        children.push_back(tokens[i]);
      }
    }
    return children;
  }

  json_token_t* getChildForToken(json_token_t* parent, std::string key) // returns the first child for key.
  {
    json_token_t* child = new json_token_t();
    child->parent_token = -1;
  	child->start = -1;
  	child->end = -1;
  	child->size = 0;

    std::vector <json_token_t*> tokens = this->tokens;
    bool is_hit = false;
    for(int i=0; i<tokens.size(); i++){
      if(tokens[i]->parent_token == parent->index){
        if (tokens[i]->type == JSON_TYPE_STRING){
          string str = this->org_string.substr(tokens[i]->start, (tokens[i]->end-tokens[i]->start));
          if (str==key){
            for(int j=i; j<tokens.size(); j++){
              if(tokens[i]->index == tokens[j]->parent_token){
                child = tokens[j];
                is_hit = true;
                break;
              }
            }
            if(is_hit){
              break;
            }
          }
        }
      }
    }
    return child;
  }

  std::string getValueForToken(json_token_t* parent)
  {
    if (parent->start >= 0){
      std::string value = this->org_string.substr(parent->start, (parent->end-parent->start));
      return value;
    }else{
      return "";
    }
  }

  std::string getChildValueForToken(json_token_t* parent, std::string key)
  {
    json_token_t* child_token = getChildForToken(parent, key);
    if (child_token->start >= 0){
      std::string value = this->org_string.substr(child_token->start, (child_token->end-child_token->start));
      return value;
    }else{
      return "";
    }
  }

  std::vector<json_token_t*> getArrayForToken(json_token_t* parent, std::string key) // returns the first child for key.
  {
    std::vector <json_token_t*> tokens = this->tokens;
    std::vector <json_token_t*> result;
    bool is_hit = false;
    for(int i=0; i<tokens.size(); i++){
      if(tokens[i]->parent_token == parent->index){
        if (tokens[i]->type == JSON_TYPE_STRING){
          string str = this->org_string.substr(tokens[i]->start, (tokens[i]->end-tokens[i]->start));
          if (str==key){
            for(int j=i; j<tokens.size(); j++){
              if( (tokens[i]->index == tokens[j]->parent_token) && (tokens[j]->type==JSON_TYPE_ARRAY)){
                is_hit = true;
                for(int k=j; k<tokens.size(); k++){
                  if(tokens[j]->index == tokens[k]->parent_token){
                    result.push_back(tokens[k]);
                  }
                }
                break;
              }
            }
            if (is_hit){
              break;
            }
          }
        }
      }
    }
    return result;
  }

  void showTokens(std::vector <json_token_t*> tokens)
  {
    for(int i=0; i<tokens.size(); i++){
      string type = "";
      string str = "";
      if (tokens[i]->type == JSON_TYPE_STRING){
        str = this->org_string.substr(tokens[i]->start, (tokens[i]->end-tokens[i]->start));
      }
      printf("%d %s start=%d, end=%d, size=%d, parent=%d %s \n",tokens[i]->index, getTokenTypeLabel(tokens[i]->type).c_str(), tokens[i]->start, tokens[i]->end, tokens[i]->size, tokens[i]->parent_token, str.c_str());
    }
  }

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

  void clean()
  {
    this->tokens.clear();
    this->tokens.shrink_to_fit();
  }

  std::string readFileToString(std::string filename)
  {
    char* ret = readFileToChars(filename);
    return string(ret);
  }

  char* readFileToChars(std::string filename)
  {
    char *buffer = NULL;
    size_t size = 0;
    FILE *fp = fopen(filename.c_str(), "r");
    if (fp==NULL){
      fprintf(stderr, "Not found: %s\n", filename.c_str());
      exit(0);
    }
    fseek(fp, 0, SEEK_END);
    size = ftell(fp);
    rewind(fp);
    buffer = new char[size + 1];
    fread(buffer, size, 1, fp);
    buffer[size] = '\0';
    return buffer;
  }

  void assignToken(json_parser *parser, json_token_t *token)
  {
    parser->next_token++;
    token->parent_token = -1;
  	token->start = -1;
  	token->end = -1;
  	token->size = 0;
  }

  int parseNumbersBools(json_parser *parser, std::string json_string)
  {
    json_token_t *token = new json_token_t;
  	int cutout = 0;
  	int start = parser->pos;
  	for (; parser->pos < json_string.length() && json_string[parser->pos] != '\0'; parser->pos++) {
  		switch (json_string[parser->pos]) {
  			case ':':
  			case '\t' :
  			case '\r' :
  			case '\n' :
  			case ' ' :
  			case ','  :
  			case ']'  :
  			case '}' :
  				cutout = 1;
  		}
  		if(cutout == 1){
  			break;
  		}
  		if((json_string[parser->pos]<' ')||(json_string[parser->pos]>=0x7f)) { // 0x7f = DEL
  			parser->pos = start;
  			return JSON_ERROR_INVALID_SYNTACS;
  		}
  	}
  	assignToken(parser, token);
  	token->type = JSON_TYPE_BASIC;
  	token->start = start;
  	token->end = parser->pos;
  	token->size = 0;
  	token->parent_token = parser->parent_token;
    token->index = this->tokens.size();
    this->tokens.push_back(token);
  	parser->pos--;
  	return 0;
  }

  int parseString(json_parser *parser, std::string json_string)
  {
    json_token_t *token = new json_token_t;
  	int start = parser->pos;
  	parser->pos++;

  	for(; (parser->pos<json_string.length()) && (json_string[parser->pos]!='\0'); parser->pos++){
  		char ch = json_string[parser->pos];
  		if (ch == '\"') {
  			assignToken(parser, token);
  			token->type = JSON_TYPE_STRING;
  			token->start = start+1;
  			token->end = parser->pos;
  			token->size = 0;
  			token->parent_token = parser->parent_token;
        token->index = this->tokens.size();
        this->tokens.push_back(token);
  			return 0;
  		}

  		if((ch=='\\') && (parser->pos+1 < json_string.length())){
  			parser->pos++;
  			switch (json_string[parser->pos]) {
  				case '\"': case '/' : case '\\' : case 'b' : case 'f' : case 'r' : case 'n'  : case 't' :
  					break;
  				case 'u':
  					parser->pos++;
  					for(int i=0; i<4 && parser->pos < json_string.length() && json_string[parser->pos] != '\0'; i++) {
  						if(!(json_string[parser->pos] >= '0' && json_string[parser->pos] <= '9') &&
  							 !(json_string[parser->pos] >= 'A' && json_string[parser->pos] <= 'F') &&
  							 !(json_string[parser->pos] >= 'a' && json_string[parser->pos] <= 'f') ){
  							parser->pos = start;
  							return JSON_ERROR_INVALID_UNICODE_SYNTACS;
  						}
  						parser->pos++;
  					}
  					parser->pos--;
  					break;
  				default:
  					parser->pos = start;
  					return JSON_ERROR_INVALID_SYNTACS;
  			}
  		}
  	}
  	parser->pos = start;
  	return JSON_ERROR_CORRUPTED_DATA;
  }

  std::string getTokenTypeLabel(json_type_t type)
  {
    int t = type;
    switch(t){
      case JSON_TYPE_NONE:
        return "NONE";
      case JSON_TYPE_DICTIONARY:
        return "DICTIONARY";
      case JSON_TYPE_ARRAY:
        return "ARRAY";
      case JSON_TYPE_STRING:
        return "STRING";
      case JSON_TYPE_BASIC:
        return "BASIC";
    }
    return "LABEL_NOT_FOUND";
  }

};
