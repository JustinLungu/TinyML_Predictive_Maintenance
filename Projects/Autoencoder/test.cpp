#include <iostream>
#include <cmath>
using namespace std;

// Function to calculate Mean Squared Error (MSE) between two matrices
float calculateMSE(float data1[][24], float data2[][24], int rows, int cols) {
    float sumSquaredDiff = 0.0;

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            float diff = data1[i][j] - data2[i][j];
            sumSquaredDiff += diff * diff;
        }
    }

    // Calculate the mean squared error
    float mse = sumSquaredDiff / (rows * cols);
    return mse;
}

int main() {
    const int rows = 3;
    const int cols = 24;

    float data[rows][cols] = { { /* Initialize your data here */ } };
    float decoded[rows][cols] = { { /* Initialize your decoded data here */ } };

    // Calculate Mean Squared Error (MSE)
    float mse = calculateMSE(data, decoded, rows, cols);

    cout << "Mean Squared Error (MSE): " << mse << endl;

    return 0;
}
