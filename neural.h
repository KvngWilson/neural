#ifndef NEURAL_H
#define NEURAL_H

#define INPUT_SIZE 3
#define HIDDEN_SIZE 4
#define OUTPUT_SIZE 1
#define LEARNING_RATE 0.0001
#define EPOCHS 1000

typedef struct
{
    double weights_ih[HIDDEN_SIZE][INPUT_SIZE];
    double bias_h[HIDDEN_SIZE];
    double weights_ho[OUTPUT_SIZE][HIDDEN_SIZE];
    double bias_o[OUTPUT_SIZE];
    double hidden[HIDDEN_SIZE];
    double output[OUTPUT_SIZE];
    double z_hidden[HIDDEN_SIZE];
    double z_output[OUTPUT_SIZE];
} NeuralNetwork;

// Activation functions
double relu(double x);
double relu_derivative(double x);

// Neural network functions
void init_network(NeuralNetwork *nn);
void forward(NeuralNetwork *nn, double input[INPUT_SIZE]);
void backward(NeuralNetwork *nn, double input[INPUT_SIZE], double target[OUTPUT_SIZE]);
void train(NeuralNetwork *nn, double inputs[][INPUT_SIZE], double targets[][OUTPUT_SIZE], int num_samples);

// Utility functions
void normalize(double data[][INPUT_SIZE], int num_samples, double min[INPUT_SIZE], double max[INPUT_SIZE]);
void normalize_targets(double targets[][OUTPUT_SIZE], int num_samples, double *min, double *max);

#endif
