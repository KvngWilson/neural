#include "neural.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

double relu(double x)
{
    return (x > 0) ? x : 0;
}

double relu_derivative(double x)
{
    return (x > 0) ? 1 : 0;
}

void init_network(NeuralNetwork *nn)
{
    srand(time(NULL));
    for (int i = 0; i < HIDDEN_SIZE; i++)
    {
        for (int j = 0; j < INPUT_SIZE; j++)
        {
            nn->weights_ih[i][j] = ((double)rand() / RAND_MAX) - 0.5;
        }
        nn->bias_h[i] = ((double)rand() / RAND_MAX) - 0.5;
    }

    for (int i = 0; i < OUTPUT_SIZE; i++)
    {
        for (int j = 0; j < HIDDEN_SIZE; j++)
        {
            nn->weights_ho[i][j] = ((double)rand() / RAND_MAX) - 0.5;
        }
        nn->bias_o[i] = ((double)rand() / RAND_MAX) - 0.5;
    }
}

void forward(NeuralNetwork *nn, double input[INPUT_SIZE])
{
    for (int i = 0; i < HIDDEN_SIZE; i++)
    {
        nn->z_hidden[i] = nn->bias_h[i];
        for (int j = 0; j < INPUT_SIZE; j++)
        {
            nn->z_hidden[i] += nn->weights_ih[i][j] * input[j];
        }
        nn->hidden[i] = relu(nn->z_hidden[i]);
    }

    for (int i = 0; i < OUTPUT_SIZE; i++)
    {
        nn->z_output[i] = nn->bias_o[i];
        for (int j = 0; j < HIDDEN_SIZE; j++)
        {
            nn->z_output[i] += nn->weights_ho[i][j] * nn->hidden[j];
        }
        nn->output[i] = relu(nn->z_output[i]);
    }
}

void backward(NeuralNetwork *nn, double input[INPUT_SIZE], double target[OUTPUT_SIZE])
{
    double output_error[OUTPUT_SIZE];
    for (int i = 0; i < OUTPUT_SIZE; i++)
    {
        output_error[i] = nn->output[i] - target[i];
    }

    double hidden_error[HIDDEN_SIZE] = {0};
    for (int i = 0; i < HIDDEN_SIZE; i++)
    {
        for (int j = 0; j < OUTPUT_SIZE; j++)
        {
            hidden_error[i] += output_error[j] * nn->weights_ho[j][i];
        }
        hidden_error[i] *= relu_derivative(nn->z_hidden[i]);
    }

    for (int i = 0; i < OUTPUT_SIZE; i++)
    {
        for (int j = 0; j < HIDDEN_SIZE; j++)
        {
            nn->weights_ho[i][j] -= LEARNING_RATE * output_error[i] * nn->hidden[j];
        }
        nn->bias_o[i] -= LEARNING_RATE * output_error[i];
    }

    for (int i = 0; i < HIDDEN_SIZE; i++)
    {
        for (int j = 0; j < INPUT_SIZE; j++)
        {
            nn->weights_ih[i][j] -= LEARNING_RATE * hidden_error[i] * input[j];
        }
        nn->bias_h[i] -= LEARNING_RATE * hidden_error[i];
    }
}

void normalize(double data[][INPUT_SIZE], int num_samples, double min[INPUT_SIZE], double max[INPUT_SIZE])
{
    // Find min and max for each feature
    for (int j = 0; j < INPUT_SIZE; j++)
    {
        min[j] = data[0][j];
        max[j] = data[0][j];
        for (int i = 1; i < num_samples; i++)
        {
            if (data[i][j] < min[j])
                min[j] = data[i][j];
            if (data[i][j] > max[j])
                max[j] = data[i][j];
        }
    }

    // Normalize with zero-variance guard
    for (int i = 0; i < num_samples; i++)
    {
        for (int j = 0; j < INPUT_SIZE; j++)
        {
            if (max[j] - min[j] > 1e-10)
            {
                data[i][j] = (data[i][j] - min[j]) / (max[j] - min[j]);
            }
            else
            {
                data[i][j] = 0.5; // Default to middle value for constant features
            }
        }
    }
}

void normalize_targets(double targets[][OUTPUT_SIZE], int num_samples, double *min, double *max)
{
    // Find min and max
    *min = targets[0][0];
    *max = targets[0][0];
    for (int i = 1; i < num_samples; i++)
    {
        if (targets[i][0] < *min)
            *min = targets[i][0];
        if (targets[i][0] > *max)
            *max = targets[i][0];
    }

    // Normalize with zero-variance guard
    if (*max - *min > 1e-10)
    {
        for (int i = 0; i < num_samples; i++)
        {
            targets[i][0] = (targets[i][0] - *min) / (*max - *min);
        }
    }
    else
    {
        // If all targets are the same, set to 0.5
        for (int i = 0; i < num_samples; i++)
        {
            targets[i][0] = 0.5;
        }
    }
}

void train(NeuralNetwork *nn, double inputs[][INPUT_SIZE], double targets[][OUTPUT_SIZE], int num_samples)
{
    for (int epoch = 0; epoch < EPOCHS; epoch++)
    {
        double total_loss = 0.0;
        for (int i = 0; i < num_samples; i++)
        {
            forward(nn, inputs[i]);
            for (int j = 0; j < OUTPUT_SIZE; j++)
            {
                double error = nn->output[j] - targets[i][j];
                total_loss += error * error;
            }
            backward(nn, inputs[i], targets[i]);
        }

        total_loss /= (num_samples * OUTPUT_SIZE);
        if (epoch % 100 == 0)
        {
            printf("Epoch %d, Loss: %.4f\n", epoch, total_loss);
        }
    }
}
