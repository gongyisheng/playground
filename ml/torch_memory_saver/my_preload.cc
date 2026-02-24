#include <sys/types.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <iostream>
#include <stdio.h>
#include <dlfcn.h>
#include <map>

// https://stackoverflow.com/questions/6083337/overriding-malloc-using-the-ld-preload-mechanism
static cudaError_t (*real_cudaMalloc)(void**, size_t) = NULL;
static cudaError_t (*real_cudaFree)(void*) = NULL;

struct MyInfo {
    CUmemGenericAllocationHandle allocHandle;
    size_t size;
};

int currentDev = 0; // HACK

std::map<void*, MyInfo> info_of_ptr_map;

static void my_init(void) {
    real_cudaMalloc = (cudaError_t (*)(void**, size_t)) dlsym(RTLD_NEXT, "cudaMalloc");
    if (NULL == real_cudaMalloc) {
        fprintf(stderr, "Error in `dlsym`: %s\n", dlerror());
    }

    real_cudaFree = (cudaError_t (*)(void*)) dlsym(RTLD_NEXT, "cudaFree");
    if (NULL == real_cudaFree) {
        fprintf(stderr, "Error in `dlsym`: %s\n", dlerror());
    }
}

void mem_create(CUmemGenericAllocationHandle *allocHandle, size_t size) {
//    size_t granularity = 0;
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = currentDev;
//    cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
//    padded_size = ROUND_UP(size, granularity);
    cuMemCreate(allocHandle, size, &prop, 0);
}

void mem_set_access(void* devPtr, size_t size) {
    CUmemAccessDesc accessDesc = {};
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.location.id = currentDev;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    cuMemSetAccess((CUdeviceptr)devPtr, size, &accessDesc, 1);
}

cudaError_t cudaMalloc(void** devPtr, size_t size) {
    if (real_cudaMalloc == NULL) {
        my_init();
    }

//    cudaError_t ret = real_cudaMalloc(devPtr, size);
//    std::cout << "[my_preload.cc] cudaMalloc" << " devPtr=" << devPtr << " size=" << size << " ret=" << ret << std::endl;
//    return ret;

    CUmemGenericAllocationHandle allocHandle;
    mem_create(&allocHandle, size);

    /* Reserve a virtual address range */
    cuMemAddressReserve((CUdeviceptr*)devPtr, size, 0, 0, 0);
    /* Map the virtual address range
     * to the physical allocation */
    cuMemMap((CUdeviceptr)*devPtr, size, 0, allocHandle, 0);

    mem_set_access(*devPtr, size);

    info_of_ptr_map[*devPtr] = MyInfo { allocHandle, size };

    std::cout << "[my_preload.cc] cudaMalloc"
        << " devPtr=" << devPtr << " size=" << size
        << " allocHandle=" << allocHandle
        << std::endl;

    return cudaSuccess;
}

cudaError_t cudaFree(void* devPtr) {
    if (real_cudaFree == NULL) {
        my_init();
    }

//    cudaError_t ret = real_cudaFree(devPtr);
//    std::cout << "[my_preload.cc] cudaFree" << " devPtr=" << devPtr << " ret=" << ret << std::endl;
//    return ret;

    MyInfo info = info_of_ptr_map[devPtr];
    info_of_ptr_map.erase(devPtr);

    cuMemUnmap((CUdeviceptr)devPtr, info.size);
    cuMemRelease(info.allocHandle);
    cuMemAddressFree((CUdeviceptr)devPtr, info.size);

    std::cout << "[my_preload.cc] cudaFree"
        << " devPtr=" << devPtr << " info.size=" << info.size
        << " info.allocHandle=" << info.allocHandle
        << std::endl;

    return cudaSuccess;
}

extern "C" {
    void hack_release_occupation() {
        for (auto it = info_of_ptr_map.begin(); it != info_of_ptr_map.end(); ++it) {
            void* devPtr = it->first;
            MyInfo info = it->second;
            cuMemUnmap((CUdeviceptr)devPtr, info.size);
            cuMemRelease(info.allocHandle);

            std::cout << "[my_preload.cc] hack_release_occupation"
                << " devPtr=" << devPtr << " info.size=" << info.size << " info.allocHandle=" << info.allocHandle
                << std::endl;
        }
    }

    void hack_resume_occupation() {
        for (auto it = info_of_ptr_map.begin(); it != info_of_ptr_map.end(); ++it) {
            void* devPtr = it->first;
            MyInfo &info = it->second;

            CUmemGenericAllocationHandle newAllocHandle;
            mem_create(&newAllocHandle, info.size);

            cuMemMap((CUdeviceptr)devPtr, info.size, 0, newAllocHandle, 0);

            mem_set_access(devPtr, info.size);

            std::cout << "[my_preload.cc] hack_resume_occupation"
                << " devPtr=" << devPtr << " info.size=" << info.size << " (old)info.allocHandle=" << info.allocHandle
                << " (new)newAllocHandle=" << newAllocHandle
                << std::endl;

            info.allocHandle = newAllocHandle;
        }
    }
}