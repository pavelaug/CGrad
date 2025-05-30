#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#define N 1000 * 10

struct Data {
  int x1;
  int x2;
  int y;
};

struct Gradients
{
  float dw1, dw2, db1;
  float dw3, dw4, db2;
  float dw5, dw6, db3;
};

struct Params {
  float weight1;
  float weight2;
  float bias1;

  float weight3;
  float weight4;
  float bias2;

  float weight5;
  float weight6;
  float bias3;
};

float randw() { return (((float)rand()/(float)RAND_MAX)); }

float sigmoid_f(float x)
{
  return (1/(1 + exp(-x)));
}

float forward(struct Params params, struct Gradients *grads, float input1, float input2, float output)
{
  
   float neuron_output_1  = (params.weight1 * input1) + (params.weight2 * input2) + params.bias1;

   float neuron_output_2 = (params.weight3 * input1) + (params.weight4 * input2) + params.bias2;

   float a1  = sigmoid_f(neuron_output_1);

   float a2  = sigmoid_f(neuron_output_2);

   float pred = params.weight5 * a1  + params.weight6 * a2 + params.bias3;

   pred = sigmoid_f(pred);
   

  float dL_dmse = -(output - pred);

  float dL_dsig1 = pred * (1 - pred);

  float dL_dw5 = a1;
  float dL_dw6 = a2;

  grads->dw5 += dL_dmse * dL_dsig1 * dL_dw5;
  grads->dw6 += dL_dmse * dL_dsig1 * dL_dw6;
  grads->db3 += dL_dmse * dL_dsig1;

  float dL_da1 = params.weight5;
  float dL_da2 = params.weight6;
  
  float dL_dsig2 = a1 * (1 - a1);    
  float dL_dsig3 = a2 * (1 - a2);

  float dl_dw1 = input1;
  float dL_dw2 = input2;

  grads->dw4 += dL_dmse * dL_dsig1 * dL_da2 * dL_dsig3 * dL_dw2;
  grads->dw3 += dL_dmse * dL_dsig1 * dL_da2 * dL_dsig3 * dl_dw1;
  grads->db2 += dL_dmse * dL_dsig1 * dL_da2 * dL_dsig3;

  grads->dw2 += dL_dmse * dL_dsig1 * dL_da1 * dL_dsig2 * dL_dw2;
  grads->dw1 += dL_dmse * dL_dsig1 * dL_da1 * dL_dsig2 * dl_dw1;
  grads->db1 += dL_dmse * dL_dsig1 * dL_da1 * dL_dsig2;
   return pred;
}

float mse(struct Params params, struct Gradients * grads, struct Data arr[4])
{
  float temp = 0;
  for(int i = 0; i < 4; i++)
  {
     float pred = forward(params, grads,  arr[i].x1, arr[i].x2, arr[i].y);
     float tempError = arr[i].y - pred;

    tempError *= tempError;
    
    temp += tempError; 
  }

  return temp/4.0f;
}

void backward(struct Params *params,  struct Data arr[4])
{
  
  float h = 1e-3;
  float learning_rate = 1e-1;
  for(int i = 0; i < N; i++)
  {     struct Gradients grads;
        grads.dw1 = 0.0f; grads.dw2 = 0.0f; grads.db1 = 0.0f;
        grads.dw3 = 0.0f; grads.dw4 = 0.0f; grads.db2 = 0.0f;
        grads.dw5 = 0.0f; grads.dw6 = 0.0f; grads.db3 = 0.0f;
        for(int i = 0; i < 4; i++)
        {
           float pred = forward(*params, &grads,  arr[i].x1, arr[i].x2, arr[i].y);
        }

    grads.dw1 /= 4.0f; grads.dw2 /= 4.0f; grads.db1 /= 4.0f;
    grads.dw3 /= 4.0f; grads.dw4 /= 4.0f; grads.db2 /= 4.0f;
    grads.dw5 /= 4.0f; grads.dw6 /= 4.0f; grads.db3 /= 4.0f;
        // --- Update Parameters using Gradient Descent ---
        params->weight1 -= learning_rate * grads.dw1;
        params->weight2 -= learning_rate * grads.dw2;
        params->bias1 -= learning_rate * grads.db1;

        params->weight3 -= learning_rate * grads.dw3;
        params->weight4 -= learning_rate * grads.dw4;
        params->bias2 -= learning_rate * grads.db2;

        params->weight5 -= learning_rate * grads.dw5;
        params->weight6 -= learning_rate * grads.dw6;
        params->bias3 -= learning_rate * grads.db3;

        printf("cost: %f \n", mse(*params, &grads,  arr));
  }

}

int main(void)
{
  srand((unsigned)time(NULL));
  // for you ignorant fucks xor logic gate is just binary addition...
  // 0 + 0, 1 + 0 is all intuitive but 1 + 1 = 0 cus integer overflow, this xor gates truth table is non linear is not modellable by a single neuron or perceptron :)
  struct Data arr[] = {
  {0, 0, 0},
  {1, 0, 1},
  {0, 1, 1},
  {1, 1, 0}
   };
  struct Params params;

  params.weight1 = randw();

  params.weight2 = randw();

  params.weight3 = randw();

  params.weight4 = randw();
  
  params.weight5 = randw();

  params.weight6 = randw();

  params.bias1 = randw();

  params.bias2 = randw();

  params.bias3 = randw();

  struct Gradients grads;

  backward(&params,  arr);  

  for(int i = 0; i < 4; i++)
  {
     float pred = forward(params, &grads,  arr[i].x1, arr[i].x2, arr[i].y);
     printf("%d + %d = %f\n", arr[i].x1, arr[i].x2, pred); 
  }

  float  c_a, c_b, c_c;

  printf("Give your custom params for the model pred, input1, input2:\n");
  scanf("%f %f %f", &c_a, &c_b, &c_c);

  printf("Result: %f\n", forward(params, &grads, c_a, c_b, c_c ));
  return 0;
}
