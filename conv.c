#include    <wb.h>


#define wbCheck(stmt) do {                                                    \
  cudaError_t err = stmt;                                               \
  if (err != cudaSuccess) {                                             \
    wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
    wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
    return -1;                                                        \
  }                                                                     \
} while(0)

#define Mask_width  5
#define Mask_radius Mask_width/2
#define O_TILE_WIDTH 12
#define BLOCK_WIDTH (O_TILE_WIDTH + (Mask_width-1))

//@@ INSERT CODE HERE
__global__ void convolution_image_kernel(float *out, float *in, int height, int width, int channels,
    const float* __restrict__ M){

  __shared__ float Ns[BLOCK_WIDTH][BLOCK_WIDTH];
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int row_o = blockIdx.y*O_TILE_WIDTH + ty;
  int col_o = blockIdx.x*O_TILE_WIDTH + tx;

  int row_i = row_o - Mask_radius;
  int col_i = col_o - Mask_radius;

  float sum;
  float pixel;
  float maskVal;


  //load image data for mask (and also ghost elements if necessary)
  //into shared variable. e.g. loading a 16x16x3 subimage
  for (int c = 0; c < channels; c++) {
    if ( (row_i >= 0) && (row_i < height) &&
        (col_i >= 0) && (col_i < width) ) {
      Ns[ty][tx] = in[(row_i*width + col_i)*channels + c]; //interleaved image; width should be pitch if pitch!=width for DRAM bursting considerations.
    }
    else {
      Ns[ty][tx] = 0.0f;
    }
    __syncthreads();
    sum = 0.0;
    if (ty < O_TILE_WIDTH && tx < O_TILE_WIDTH) {
      for (int y = 0; y < Mask_width; y++){
        for (int x = 0; x < Mask_width; x++){
          pixel = Ns[ty + y][tx + x];
          maskVal = M[y*Mask_width + x];
          sum += pixel*maskVal;
        }
      }


      if (row_o < height && col_o < width) {
        out[ (row_o * width + col_o) * channels + c] = min(max(0.0f,sum),1.0f);
      }
    }

    __syncthreads();
  } 

}


int main(int argc, char* argv[]) {
  wbArg_t args;
  int maskRows;
  int maskColumns;
  int imageChannels;
  int imageWidth;
  int imageHeight;
  char * inputImageFile;
  char * inputMaskFile;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float * hostInputImageData;
  float * hostOutputImageData;
  float * hostMaskData;
  float * deviceInputImageData;
  float * deviceOutputImageData;
  float * deviceMaskData;

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);
  inputMaskFile = wbArg_getInputFile(args, 1);

  inputImage = wbImport(inputImageFile);
  hostMaskData = (float *) wbImport(inputMaskFile, &maskRows, &maskColumns);

  assert(maskRows == 5); /* mask height is fixed to 5 in this mp */
  assert(maskColumns == 5); /* mask width is fixed to 5 in this mp */

  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);

  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  cudaMalloc((void **) &deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
  cudaMalloc((void **) &deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
  cudaMalloc((void **) &deviceMaskData, maskRows * maskColumns * sizeof(float));
  wbTime_stop(GPU, "Doing GPU memory allocation");


  wbTime_start(Copy, "Copying data to the GPU");
  cudaMemcpy(deviceInputImageData,
      hostInputImageData,
      imageWidth * imageHeight * imageChannels * sizeof(float),
      cudaMemcpyHostToDevice);
  cudaMemcpy(deviceMaskData,
      hostMaskData,
      maskRows * maskColumns * sizeof(float),
      cudaMemcpyHostToDevice);
  wbTime_stop(Copy, "Copying data to the GPU");


  wbTime_start(Compute, "Doing the computation on the GPU");

  //@@ INSERT CODE HERE
  dim3 DimGrid( (imageWidth-1)/O_TILE_WIDTH + 1, (imageHeight-1)/O_TILE_WIDTH + 1, 1 ); //write out
  dim3 DimBlock( BLOCK_WIDTH, BLOCK_WIDTH, 1 ); //read in

  convolution_image_kernel<<<DimGrid,DimBlock>>>(deviceOutputImageData, deviceInputImageData,
      imageHeight,imageWidth,imageChannels,
      deviceMaskData);


  wbTime_stop(Compute, "Doing the computation on the GPU");


  wbTime_start(Copy, "Copying data from the GPU");
  cudaMemcpy(hostOutputImageData,
      deviceOutputImageData,
      imageWidth * imageHeight * imageChannels * sizeof(float),
      cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  wbSolution(args, outputImage);

  cudaFree(deviceInputImageData);
  cudaFree(deviceOutputImageData);
  cudaFree(deviceMaskData);

  free(hostMaskData);
  wbImage_delete(outputImage);
  wbImage_delete(inputImage);

  return 0;
}

