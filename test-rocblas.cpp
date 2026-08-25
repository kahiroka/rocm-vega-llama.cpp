#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <cstdio>

int main() {
    int count = 0;
    hipError_t hip_status = hipGetDeviceCount(&count);
    if (hip_status != hipSuccess) {
        std::fprintf(stderr, "hipGetDeviceCount failed: %s\n", hipGetErrorString(hip_status));
        return 1;
    }

    std::printf("HIP devices: %d\n", count);
    for (int i = 0; i < count; ++i) {
        hipDeviceProp_t prop{};
        hipGetDeviceProperties(&prop, i);
        std::printf("device %d: %s  arch=%s\n", i, prop.name, prop.gcnArchName);
    }

    if (count == 0) return 1;

    // Device 0 is intentionally used to validate gfx900 on the tested mixed rig.
    hipSetDevice(0);

    constexpr int N = 2;
    float hA[N * N] = {1, 2, 3, 4};
    float hB[N * N] = {5, 6, 7, 8};
    float hC[N * N] = {0, 0, 0, 0};
    float *dA = nullptr, *dB = nullptr, *dC = nullptr;

    if (hipMalloc(&dA, sizeof(hA)) != hipSuccess ||
        hipMalloc(&dB, sizeof(hB)) != hipSuccess ||
        hipMalloc(&dC, sizeof(hC)) != hipSuccess) {
        std::fprintf(stderr, "hipMalloc failed\n");
        return 1;
    }

    hipMemcpy(dA, hA, sizeof(hA), hipMemcpyHostToDevice);
    hipMemcpy(dB, hB, sizeof(hB), hipMemcpyHostToDevice);
    hipMemset(dC, 0, sizeof(hC));

    rocblas_handle handle{};
    rocblas_status status = rocblas_create_handle(&handle);
    if (status != rocblas_status_success) {
        std::fprintf(stderr, "rocblas_create_handle failed: %d\n", (int)status);
        return 1;
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;

    std::printf("Calling rocblas_sgemm on device 0...\n");
    status = rocblas_sgemm(handle,
                           rocblas_operation_none,
                           rocblas_operation_none,
                           N, N, N,
                           &alpha,
                           dA, N,
                           dB, N,
                           &beta,
                           dC, N);

    std::printf("rocblas_sgemm returned: %d\n", (int)status);
    hip_status = hipDeviceSynchronize();
    std::printf("hipDeviceSynchronize: %s\n", hipGetErrorString(hip_status));

    if (status == rocblas_status_success && hip_status == hipSuccess) {
        hipMemcpy(hC, dC, sizeof(hC), hipMemcpyDeviceToHost);
        std::printf("C = [%g %g; %g %g]\n", hC[0], hC[1], hC[2], hC[3]);
    }

    rocblas_destroy_handle(handle);
    hipFree(dA);
    hipFree(dB);
    hipFree(dC);

    return (status == rocblas_status_success && hip_status == hipSuccess) ? 0 : 1;
}
