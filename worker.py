"""
Vast.ai PyWorker for SeedVC Voice Conversion
This worker integrates with Vast.ai serverless autoscaler
"""
from vastai import (
    Worker,
    WorkerConfig,
    HandlerConfig,
    BenchmarkConfig,
    LogActionConfig,
)

worker_config = WorkerConfig(
    model_server_url="http://127.0.0.1",
    model_server_port=8080,
    model_log_file="/var/log/seedvc/server.log",
    handlers=[
        HandlerConfig(
            route="/convert",
            allow_parallel_requests=False,  # SeedVC processes one at a time
            max_queue_time=300.0,  # 5 minutes max wait
            workload_calculator=lambda payload: 100.0,  # Fixed cost per request
            benchmark_config=BenchmarkConfig(
                generator=lambda: {},  # Empty for non-LLM
                runs=1,
                concurrency=1,
            ),
        ),
        HandlerConfig(
            route="/health",
            allow_parallel_requests=True,
            max_queue_time=10.0,
            workload_calculator=lambda payload: 1.0,
        ),
    ],
    log_action_config=LogActionConfig(
        on_load=["Models loaded successfully", "Uvicorn running on"],
        on_error=["Traceback (most recent call last):", "RuntimeError:", "CUDA out of memory"],
        on_info=["Loading Seed-VC", "Converting:"],
    ),
)

if __name__ == "__main__":
    Worker(worker_config).run()
