#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
struct Data{
  int x;
  int y;
};

float cost(float weight, struct Data arr[8])
{
    float error = 0;
    float pavg = 0;
    for(int j = 0; j < 5; j++)
    {
      float pred = arr[j].x * weight;

      pavg += pred;
       float temp = arr[j].y - pred;
      temp *= temp;
      error += temp;
    }
    return error/5.0f;
}

int main(void)
{
  struct Data arr[8] = {
   {2, 6},     {3, 9},     {4, 12},     {5, 15},     {6, 18},     {7, 21},     {8, 24},     {9, 27}
  };

  float weight = 1e-3;
  float learning_rate = 1e-3;
  // y = x * w + b
  float h = 1e-5;
  for(int i = 0; i < 500; i++)
  {
    float gradient = (cost(weight + h, arr) - cost(weight, arr))/h;
    weight = weight - learning_rate *  gradient; 
    printf("epoch: %d ", i);
     printf("weight: %f ", weight);
    float result = cost(weight, arr);
    printf("cost: %f\n", result);
  }
  
   return 0;
}
