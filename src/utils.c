#include "utils.h"
#include <assert.h>
#include <stdlib.h>

void error(const char *s)
{
    perror(s);
    assert(0);
    exit(-1);
}

char *read_file_to_chars(char *filename)
{
  char *buffer = NULL;
  size_t size = 0;
  FILE *fp = fopen(filename, "r");
  if (fp==NULL){
    printf("%s\n", filename);
    error("file not found");
  }
  fseek(fp, 0, SEEK_END);
  size = ftell(fp);
  rewind(fp);
  buffer = malloc(size + 1);
  fread(buffer, size, 1, fp);
  buffer[size] = '\0';
  return buffer;
}
