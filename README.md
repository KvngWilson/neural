# Neural Network in C

A simple feedforward neural network implementation in C for price prediction.

## Project Structure

```
neural/
├── neural.h        # Header file with structures and function declarations
├── neural.c        # Implementation of neural network functions
├── main.c          # Main program with training data and execution
├── Makefile        # Build automation
└── README.md       # This file
```

## Architecture

- **Input Layer**: 3 neurons (features)
- **Hidden Layer**: 4 neurons with ReLU activation
- **Output Layer**: 1 neuron with ReLU activation
- **Learning Rate**: 0.0001
- **Training Epochs**: 1000

## Features

- Min-max normalization with zero-variance guards
- ReLU activation function
- Backpropagation with gradient descent
- Mean squared error loss function
- Modular, maintainable codebase

## Building

### Using Makefile (recommended)
```bash
make              # Build the project
make run          # Build and run
make clean        # Remove build artifacts
```

### Manual compilation
```bash
gcc -o neural main.c neural.c -lm
./neural
```

## Usage

The program trains on sample data and makes predictions. Modify the training data in `main.c` to fit your use case.

## Recent Improvements

- **Modular structure**: Separated into header, implementation, and main files
- **Bug fixes**: 
  - Fixed OUTPUT_SIZE mismatch (changed from 2 to 1)
  - Fixed num_samples (now correctly uses all 10 samples)
  - Added zero-variance guards in normalization
  - Fixed divide-by-zero in target normalization
- **Code quality**: Extracted normalize_targets function for better organization
