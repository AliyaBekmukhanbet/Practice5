%%cuda
#include <cuda_runtime.h>
#include <stdio.h>

// =======================
// ПАРАЛЛЕЛЬНАЯ ОЧЕРЕДЬ
// =======================
struct Queue {
    int* data;
    int head;
    int tail;
    int capacity;

    __device__ void init(int* buffer, int size) {
        data = buffer;
        head = 0;
        tail = 0;
        capacity = size;
    }

    __device__ bool enqueue(int value) {
        int pos = atomicAdd(&tail, 1);
        if (pos < capacity) {
            data[pos] = value;
            return true;
        }
        return false;
    }

    __device__ bool dequeue(int* value) {
        int pos = atomicAdd(&head, 1);
        if (pos < tail) {
            *value = data[pos];
            return true;
        }
        return false;
    }
};

// =======================
// ПАРАЛЛЕЛЬНЫЙ СТЕК
// =======================
struct Stack {
    int* data;
    int top;
    int capacity;

    __device__ void init(int* buffer, int size) {
        data = buffer;
        top = 0;
        capacity = size;
    }

    __device__ bool push(int value) {
        int pos = atomicAdd(&top, 1);
        if (pos < capacity) {
            data[pos] = value;
            return true;
        }
        return false;
    }

    __device__ bool pop(int* value) {
        int pos = atomicSub(&top, 1) - 1;
        if (pos >= 0) {
            *value = data[pos];
            return true;
        }
        return false;
    }
};

// =======================
// KERNEL ИНИЦИАЛИЗАЦИИ
// =======================
__global__ void initQueueKernel(Queue* q, int* buffer, int size) {
    q->init(buffer, size);
}

__global__ void initStackKernel(Stack* s, int* buffer, int size) {
    s->init(buffer, size);
}

// =======================
// KERNEL ОЧЕРЕДИ
// =======================
__global__ void queueKernel(Queue* q, int numOps) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = 0; i < numOps; i++) {
        q->enqueue(tid * numOps + i);
    }

    __syncthreads();

    for (int i = 0; i < numOps; i++) {
        int value;
        q->dequeue(&value);
    }
}

// =======================
// KERNEL СТЕКА
// =======================
__global__ void stackKernel(Stack* s, int numOps) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = 0; i < numOps; i++) {
        s->push(tid * numOps + i);
    }

    __syncthreads();

    for (int i = 0; i < numOps; i++) {
        int value;
        s->pop(&value);
    }
}

// =======================
// ИЗМЕРЕНИЕ ВРЕМЕНИ
// =======================
template <typename Kernel, typename Struct>
float measureTime(Kernel kernel, Struct* d_struct,
                  int numOps, int blocks, int threads) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    kernel<<<blocks, threads>>>(d_struct, numOps);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float time = 0.0f;
    cudaEventElapsedTime(&time, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return time;
}

// =======================
// MAIN
// =======================
int main() {
    const int CAPACITY = 1 << 20;   // 1M элементов
    const int NUM_OPS = 10;
    const int BLOCKS = 32;
    const int THREADS = 256;

    // ---------- QUEUE ----------
    Queue* d_queue;
    int* d_queueBuffer;

    cudaMalloc(&d_queue, sizeof(Queue));
    cudaMalloc(&d_queueBuffer, CAPACITY * sizeof(int));

    initQueueKernel<<<1, 1>>>(d_queue, d_queueBuffer, CAPACITY);
    cudaDeviceSynchronize();

    float queueTime = measureTime(queueKernel, d_queue,
                                  NUM_OPS, BLOCKS, THREADS);

    // ---------- STACK ----------
    Stack* d_stack;
    int* d_stackBuffer;

    cudaMalloc(&d_stack, sizeof(Stack));
    cudaMalloc(&d_stackBuffer, CAPACITY * sizeof(int));

    initStackKernel<<<1, 1>>>(d_stack, d_stackBuffer, CAPACITY);
    cudaDeviceSynchronize();

    float stackTime = measureTime(stackKernel, d_stack,
                                  NUM_OPS, BLOCKS, THREADS);

    // ---------- RESULTS ----------
    printf("Queue time: %.4f ms\n", queueTime);
    printf("Stack time: %.4f ms\n", stackTime);

    if (queueTime < stackTime)
        printf("Queue is faster by %.4f ms\n", stackTime - queueTime);
    else
        printf("Stack is faster by %.4f ms\n", queueTime - stackTime);

    // ---------- CLEANUP ----------
    cudaFree(d_queue);
    cudaFree(d_queueBuffer);
    cudaFree(d_stack);
    cudaFree(d_stackBuffer);

    return 0;
}
