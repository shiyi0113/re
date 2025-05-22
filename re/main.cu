#include <iostream>
#include "kernels/vectorAdd_cute.cuh"
#include "kernels/hgemm_cute_naive.cuh"

int main(int argc, char **argv){
    vector_add_kernel(102400);
    hgemm_cute_naive_kernel();
}