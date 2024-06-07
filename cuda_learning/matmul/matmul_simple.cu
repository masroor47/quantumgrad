#include<stdio.h>
#include<cuda.h>
#include<cuda_runtime.h>


__global__ void matmul_simple(float *left, float *right, float *result, int lrows, int lcols, int rrows, int rcols) {
    // current row of left, current col of right
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float temp_sum = 0.0;
    if ((row < lrows) && (col < rcols)) {
        // dot product
        for (int i = 0; i < lcols; i++) {
            temp_sum += left[row * lcols + i] * right[i * rcols + col];
        }
        result[row * rcols + col] = temp_sum;
    }

}

void printMatrixPretty(float *matrix, int rows, int cols) {
    if (rows < 4 || cols < 4) {
        printf("Matrix small to use this nice matrix printer.\n");
    }

    printf("\n");

    for (int r = 0; r < 2; r++) {
        for (int c = 0; c < 2; c++) {
            printf("%f, ", matrix[r * cols + c]);
        }
        printf("... ");
        for (int c = cols - 2; c < cols - 1; c++) {
            printf("%f, ", matrix[r * cols + c]);
        }
        printf("%f\n", matrix[r * cols + cols - 1]);
    }

    for (int i = 0; i < 3; i++) {
        printf(".\n");
    }

    for (int r = rows - 2; r < rows; r++) {
        for (int c = 0; c < 2; c++) {
            printf("%f, ", matrix[r * cols + c]);
        }
        printf("... ");
        for (int c = cols - 2; c < cols - 1; c++) {
            printf("%f, ", matrix[r * cols + c]);
        }
        printf("%f\n", matrix[r * cols + cols - 1]);
    } 
    printf("(%i, %i)\n\n", rows, cols);
}


int main() {
    int leftRows = 2048;
    int leftCols = 1024;
    int rightRows = 1024;
    int rightCols = 4096;

    int leftSize = leftRows * leftCols;
    int rightSize = rightRows * rightCols;
    int resultSize = leftRows * rightCols;

    if (leftCols != rightRows) {
        printf("Shapes don't match for matmul\n");
        return -1;
    }
    

    float *left, *right, *result;
    
    cudaMallocHost(&left, leftSize * sizeof(float));
    cudaMallocHost(&right, rightSize * sizeof(float));
    cudaMallocHost(&result, resultSize * sizeof(float));
    printf("Allocated memory on host\n");

    for (int r = 0; r < leftRows; r++) {
        for (int c = 0; c < leftCols; c++) {
            left[r * leftCols + c] = 1;
            right[c * rightCols + r] = 1;
        }
    }
    printf("Initialized matrices on host\n");
    printMatrixPretty(left, leftRows, leftCols);
    printMatrixPretty(right, rightRows, rightCols);
 
    float *d_left, *d_right, *d_result;
    cudaMalloc(&d_left, leftSize * sizeof(float));
    cudaMalloc(&d_right, rightSize * sizeof(float));
    cudaMalloc(&d_result, resultSize * sizeof(float));

    printf("Allocated memory on device\n");

    cudaMemcpy(d_left, left, leftSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_right, right, rightSize * sizeof(float), cudaMemcpyHostToDevice);

    printf("Copied matrices ot device\n");

    int BLOCK_SIZE = 16;
    int GRID_SIZE_ROWS = (int)ceil((float)leftRows / BLOCK_SIZE);
    int GRID_SIZE_COLS = (int)ceil((float)rightCols / BLOCK_SIZE);
    
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize(GRID_SIZE_COLS, GRID_SIZE_ROWS);

    printf("Grid : {%d, %d, %d} blocks. Blocks : {%d, %d, %d} threads.\n",
    gridSize.x, gridSize.y, gridSize.z, blockSize.x, blockSize.y, blockSize.z);

    printf("Launching kernel...\n");

    matmul_simple<<<gridSize, blockSize>>>(d_left, d_right, d_result, leftRows, leftCols, rightRows, rightCols);

    cudaError_t kernelErr = cudaGetLastError();
    if (kernelErr != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(kernelErr));
    } else {
        printf("Completed successfully\n");
    }

    cudaMemcpy(result, d_result, resultSize * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_left);
    cudaFree(d_right);
    cudaFree(d_result);
    
    printf("\n");
    printMatrixPretty(result, leftRows, rightCols);

}
