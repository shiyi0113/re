#include <iostream>
#include "kernels/vectorAdd_cute.cuh"
#include "kernels/hgemm_cute_naive.cuh"
#include "kernels/hgemm_cute_multiStage.cuh"

int main(int argc, char **argv){
    //vector_add_kernel(102400);
    //hgemm_cute_naive();
    hgemm_cute_multiStage();
}