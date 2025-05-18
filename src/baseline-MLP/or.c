#include <stdio.h>
#include <math.h>

struct Data {
  int x1;
  int x2;
  int y;
};

struct Params {
  float weight1;
  float bias1;

  float weight2;
  float bias2;

  float weight3;
  float bias3;
};

float sigmoid(float x)
{
  return  1/(1 + exp(-x));
}


float mse(float weight1, float weight2, float weight3, float bias1,  struct Data arr[4])
{
  float temp = 0;
  for(int i = 0; i < 4; i++)
  {
    float pred = (weight1 * arr[i].x1) + (weight2 * arr[i].x2) + bias1;

    pred = sigmoid(pred);
    pred = weight3 * pred;
    
    float tempError = arr[i].y - pred;

    tempError *= tempError;
    
    temp += tempError; 
  }

  return temp/4.0f;
}

int main(void)
{
 struct Data arr[4] = {
    {0, 0, 0},
    {1, 0, 1},
    {0, 1, 1},
    {1, 1, 1}
  };

  struct Params params;

  params.weight1 = 1e-3;

  params.weight2 = 1e-3;

  params.bias1 = 1e-3;
  float h = 1e-5;
  float learning_rate = 1e-3;
  for(int i = 0; i < 1000000 + 1000; i++)
  {
    float gradient1 = (mse(params.weight1 + h, params.weight2, params.weight3, params.bias1,  arr) - mse(params.weight1, params.weight2, params.weight3, params.bias1,  arr))/h;
    
    float gradient2 = (mse(params.weight1, params.weight2 + h, params.weight3, params.bias1,  arr) - mse(params.weight1, params.weight2, params.weight3, params.bias1,  arr))/h;

    float gradient4 = (mse(params.weight1, params.weight2, params.weight3, params.bias1 + h,  arr) - mse(params.weight1, params.weight2, params.weight3, params.bias1,  arr))/h;

   float gradient3 = (mse(params.weight1, params.weight2, params.weight3 + h, params.bias1,  arr) - mse(params.weight1, params.weight2, params.weight3, params.bias1,  arr))/h;

    params.weight1 = params.weight1 - learning_rate * gradient1;
    params.weight2 = params.weight2 - learning_rate * gradient2;
    params.weight3 = params.weight3 - learning_rate * gradient3;

    params.bias1 = params.bias1 - learning_rate * gradient4;
    printf("cost: %f \n", mse(params.weight1, params.weight2, params.weight3, params.bias1,  arr));
  }

  
  return 0;
}
