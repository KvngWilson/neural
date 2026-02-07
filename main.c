#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define INPUT_SIZE 3
#define HIDDEN_SIZE 4
#define OUTPUT_SIZE 2
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

    for (int i = 0; i < num_samples; i++)
    {
        for (int j = 0; j < INPUT_SIZE; j++)
        {
            if (max[j] - min[j] != 0)
            {
                data[i][j] = (data[i][j] - min[j]) / (max[j] - min[j]);
            }
            else
            {
                data[i][j] = 0;
            }
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

        total_loss /= num_samples;
        if (epoch % 100 == 0)
        {
            printf("Epoch %d, Loss: %.4f\n", epoch, total_loss);
        }
    }
}

int main()
{
    double training_inputs[][INPUT_SIZE] = {
        {3, 1500, 2.0}, {4, 2000, 3.0}, {2, 1200, 1.5}, {5, 800, 4.0}, {3, 1800, 2.5}, {4, 2200, 3.5}, {2, 900, 1.8}, {5, 2700, 4.5}, {3, 3000, 2.2}, {6, 2100, 3.2}};

    double training_targets[][OUTPUT_SIZE] = {
        {300}, {400}, {200}, {500}, {350}, {450}, {250}, {180}, {600}, {480}};

    int num_samples = 8;

    double min[INPUT_SIZE], max[INPUT_SIZE];
    normalize(training_inputs, num_samples, min, max);

    double target_min = training_targets[0][0];
    double target_max = training_targets[0][0];

    for (int i = 1; i < num_samples; i++)
    {
        if (training_targets[i][0] < target_min)
            target_min = training_targets[i][0];
        if (training_targets[i][0] > target_max)
            target_max = training_targets[i][0];
    }

    for (int i = 0; i < num_samples; i++)
    {
        training_targets[i][0] = (training_targets[i][0] - target_min) / (target_max - target_min);
    }

    NeuralNetwork nn;
    init_network(&nn);

    printf("Starting training...\n");
    train(&nn, training_inputs, training_targets, num_samples);

    printf("\n=== Testing ===\n");
    for (int i = 0; i < num_samples; i++)
    {
        forward(&nn, training_inputs[i]);
        double predicted = nn.output[0] * (target_max - target_min) + target_min;
        double actual = training_targets[i][0] * (target_max - target_min) + target_min;
        printf("Input: [%.2f, %.2f, %.2f], Predicted Price: %.2f, Actual Price: %.2f\n", training_inputs[i][0], training_inputs[i][1], training_inputs[i][2], predicted, actual);
    }

    printf("\n=== New Predictions ===\n");
    double new_inputs[INPUT_SIZE] = {4, 2500, 3.0};
    for (int j = 0; j < INPUT_SIZE; j++)
    {
        new_inputs[j] = (new_inputs[j] - min[j]) / (max[j] - min[j]);
    }
    forward(&nn, new_inputs);
    double prediction = nn.output[0] * (target_max - target_min) + target_min;
    printf("Input: [%.2f, %.2f, %.2f]", new_inputs[0], new_inputs[1], new_inputs[2]);
    printf("Predicted Price: %.2f\n", prediction);

    return 0;
}