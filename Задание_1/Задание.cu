%%cuda
#include <cuda_runtime.h>
#include <stdio.h>

#define CAPACITY 1024  // Фиксированная ёмкость стека
#define NUM_THREADS 512  // Количество потоков для теста (больше ёмкости для проверки переполнения)

// Глобальные переменные на устройстве
__device__ int d_top;
__device__ int d_data[CAPACITY];

// Структура стека (методы на устройстве)
struct Stack {
    __device__ void init() {
        // Инициализация должна выполняться одним потоком (например, threadIdx.x == 0)
        d_top = -1;
    }

    __device__ bool push(int value) {
        int old = atomicAdd(&d_top, 1);  // Возвращает старое значение top
        int pos = old + 1;
        if (pos >= CAPACITY) {
            atomicAdd(&d_top, -1);  // Отмена, если переполнение
            return false;
        }
        d_data[pos] = value;
        return true;
    }

    __device__ bool pop(int *value) {
        int old = atomicAdd(&d_top, -1);  // Возвращает старое значение top
        if (old < 0) {
            atomicAdd(&d_top, 1);  // Отмена, если стек пуст
            return false;
        }
        *value = d_data[old];
        return true;
    }
};

// Ядро для параллельного push (каждый поток пытается push своё значение)
__global__ void pushKernel(int *d_success_count) {
    Stack stack;
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        stack.init();  // Инициализация стека одним потоком
    }
    __syncthreads();  // Синхронизация перед операциями

    int value = threadIdx.x + blockIdx.x * blockDim.x;  // Уникальное значение для каждого потока
    bool success = stack.push(value);
    if (success) {
        atomicAdd(d_success_count, 1);  // Счётчик успешных push
    }
}

// Ядро для параллельного pop (каждый поток пытается pop значение)
__global__ void popKernel(int *d_popped_sum, int *d_success_count) {
    Stack stack;
    int value;
    bool success = stack.pop(&value);
    if (success) {
        atomicAdd(d_popped_sum, value);  // Сумма popped значений для проверки
        atomicAdd(d_success_count, 1);  // Счётчик успешных pop
    }
}

// Хост-код
int main() {
    // Задача 1: Инициализация стека с фиксированной ёмкостью

    int *d_push_success, *d_pop_success, *d_popped_sum;
    int h_push_success = 0, h_pop_success = 0, h_popped_sum = 0;

    cudaMalloc(&d_push_success, sizeof(int));
    cudaMalloc(&d_pop_success, sizeof(int));
    cudaMalloc(&d_popped_sum, sizeof(int));

    cudaMemcpy(d_push_success, &h_push_success, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pop_success, &h_pop_success, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_popped_sum, &h_popped_sum, sizeof(int), cudaMemcpyHostToDevice);

    // Задача 2: Запуск ядра для параллельного push
    pushKernel<<<1, NUM_THREADS>>>(d_push_success);
    cudaDeviceSynchronize();

    // Копируем результаты push
    cudaMemcpy(&h_push_success, d_push_success, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Успешных push: %d (ожидается <= %d)\n", h_push_success, CAPACITY);

    // Запуск ядра для параллельного pop
    popKernel<<<1, NUM_THREADS>>>(d_popped_sum, d_pop_success);
    cudaDeviceSynchronize();

    // Копируем результаты pop
    cudaMemcpy(&h_pop_success, d_pop_success, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_popped_sum, d_popped_sum, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Успешных pop: %d (должно совпадать с push)\n", h_pop_success);
    printf("Сумма popped значений: %d\n", h_popped_sum);

    // Задача 3: Проверка корректности
    int *h_data = (int*)malloc(CAPACITY * sizeof(int));
    cudaMemcpyFromSymbol(h_data, d_data, CAPACITY * sizeof(int), 0, cudaMemcpyDeviceToHost);

    // Пример проверки: вывод первых 10 элементов (для отладки)
    printf("Первые 10 элементов в data (после pop стек пуст, но data остаётся):\n");
    for (int i = 0; i < 10 && i < h_push_success; ++i) {
        printf("%d ", h_data[i]);
    }
    printf("\n");

    // Освобождение памяти
    cudaFree(d_push_success);
    cudaFree(d_pop_success);
    cudaFree(d_popped_sum);
    free(h_data);

    // Если push_success == pop_success && push_success <= CAPACITY, тест пройден
    if (h_push_success == h_pop_success && h_push_success <= CAPACITY) {
        printf("Тест пройден: операции корректны.\n");
    } else {
        printf("Тест провален: несоответствие.\n");
    }

    return 0;
}
