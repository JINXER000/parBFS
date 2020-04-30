#ifndef GPU_TIMER_H__
#define GPU_TIMER_H__

#include <cuda_runtime.h>
#include <sys/time.h>
#include <stdio.h>
struct GpuTimer
{
  cudaEvent_t start;
  cudaEvent_t stop;

  GpuTimer()
  {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
  }

  ~GpuTimer()
  {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  void Start()
  {
    cudaEventRecord(start, 0);
  }

  void Stop()
  {
    cudaEventRecord(stop, 0);
  }

  float Elapsed()
  {
    float elapsed;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    return elapsed;
  }
};


struct timer{

  void start(){
    gettimeofday(&t_b, 0);
  }
  double elapsed()
  {
    gettimeofday(&t_e, 0);
    double timefly=(t_e.tv_sec - t_b.tv_sec) + (t_e.tv_usec - t_b.tv_usec)*1e-6;
    return timefly;
  }
  timeval t_b, t_e;
};
#define TIME_BEGIN() {timeval t_b, t_e; gettimeofday(&t_b, 0);
#define TIME_END(TAG) gettimeofday(&t_e, 0); printf("=== %s time: %lf s\n", TAG, (t_e.tv_sec - t_b.tv_sec) + (t_e.tv_usec - t_b.tv_usec)*1e-6); }

#endif  /* GPU_TIMER_H__ */
