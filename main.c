#include <stdio.h>
#include "neural.h"

int main()
{
    double training_inputs[][INPUT_SIZE] = {
        {3, 1500, 2.0}, {4, 2000, 3.0}, {2, 1200, 1.5}, {5, 800, 4.0}, {3, 1800, 2.5}, {4, 2200, 3.5}, {2, 900, 1.8}, {5, 2700, 4.5}, {3, 3000, 2.2}, {6, 2100, 3.2}};

    double training_targets[][OUTPUT_SIZE] = {
        {300}, {400}, {200}, {500}, {350}, {450}, {250}, {180}, {600}, {480}};

    int num_samples = 10;

    double min[INPUT_SIZE], max[INPUT_SIZE];
    normalize(training_inputs, num_samples, min, max);

    double target_min, target_max;
    normalize_targets(training_targets, num_samples, &target_min, &target_max);

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
        if (max[j] - min[j] > 1e-10)
        {
            new_inputs[j] = (new_inputs[j] - min[j]) / (max[j] - min[j]);
        }
        else
        {
            new_inputs[j] = 0.5;
        }
    }
    forward(&nn, new_inputs);
    double prediction = nn.output[0] * (target_max - target_min) + target_min;
    printf("Input: [4.00, 2500.00, 3.00], Predicted Price: %.2f\n", prediction);

    return 0;
}