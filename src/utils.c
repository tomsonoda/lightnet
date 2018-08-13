#include "utils.h"
#include <assert.h>
#include <stdlib.h>

float array_mean(float *array, int elements)
{
    return array_sum(array,elements)/elements;
}

float array_sum(float *array, int elements)
{
    int i;
    float sum = 0;
    for(i = 0; i < elements; ++i){
      sum += array[i];
    }
    return sum;
}

void error(const char *s)
{
    perror(s);
    assert(0);
    exit(-1);
}
