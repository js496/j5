from fastapi import FastAPI, Request, HTTPException
import json
import subprocess
import docker
from docker.types import DeviceRequest
import time
import os
import redis.asyncio as redis
import sys
from fastapi.responses import JSONResponse
import asyncio
from datetime import datetime
from contextlib import asynccontextmanager
import pynvml
import logging
from typing import Dict, List, Optional, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
client = docker.from_env()
r = redis.Redis(host="redis", port=6379, db=0)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def get_gpu_info():
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        gpu_info = []
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        gpu_util = utilization.gpu  # GPU utilization percentage

        # Get GPU memory information
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        mem_used = memory_info.used / 1024**2  # Convert bytes to MB
        mem_total = memory_info.total / 1024**2  # Convert bytes to MB
        mem_util = (mem_used / mem_total) * 100  # Memory utilization percentage

        gpu_info.append({
                "gpu_count": device_count,  # Total number of GPUs
                "gpu_util": float(gpu_util),
                "mem_used": float(mem_used),
                "mem_total": float(mem_total),
                "mem_util": float(mem_util)
        })
        pynvml.nvmlShutdown()
     
        return gpu_info
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return 0

current_gpu_info = get_gpu_info()
gpu_int_arr = [0]


async def redis_timer():
    """
    Periodically update GPU information in Redis.
    """
    while True:
        try:
            current_gpu_info = get_gpu_info()
            if not current_gpu_info:
                logger.warning("No GPU info available")
                await asyncio.sleep(0.2)
                continue

            res_db_gpu = await r.get("db_gpu")
            db_gpu = json.loads(res_db_gpu) if res_db_gpu else []

            updated_gpu_data = []
            for gpu_int, gpu_data in enumerate(db_gpu):
                updated_gpu_data.append({
                    "gpu": gpu_int,
                    "gpu_info": json.dumps(current_gpu_info),
                    "running_model": gpu_data.get("running_model", "0"),
                    "timestamp": datetime.now().isoformat(),
                    "port_vllm": gpu_data.get("port_vllm", "0"),
                    "port_model": gpu_data.get("port_model", "0"),
                    "used_ports": gpu_data.get("used_ports", "0"),
                    "used_models": gpu_data.get("used_models", "0"),
                })

            await r.set("db_gpu", json.dumps(updated_gpu_data))
            await asyncio.sleep(0.2)

        except Exception as e:
            logger.error(f"Error in Redis timer: {e}")
            await asyncio.sleep(0.2)


async def redis_add(gpu: int, running_model: str, port_vllm: str, port_model: str, used_ports: str, used_models: str):
    """
    Add GPU data to Redis.

    Args:
        gpu (int): GPU index.
        running_model (str): Name of the running model.
        port_vllm (str): vLLM port.
        port_model (str): Model port.
        used_ports (str): Used ports.
        used_models (str): Used models.
    """
    try:
        current_gpu_info = get_gpu_info()
        res_db_gpu = await r.get("db_gpu")
        db_gpu = json.loads(res_db_gpu) if res_db_gpu else []

        add_data = {
            "gpu": gpu,
            "gpu_info": json.dumps(current_gpu_info),
            "running_model": running_model,
            "timestamp": datetime.now().isoformat(),
            "port_vllm": port_vllm,
            "port_model": port_model,
            "used_ports": used_ports,
            "used_models": used_models,
        }
        db_gpu.append(add_data)
        await r.set("db_gpu", json.dumps(db_gpu))

    except Exception as e:
        logger.error(f"Error adding data to Redis: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan manager for FastAPI app.
    """
    asyncio.create_task(redis_timer())
    yield


# FastAPI app
app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    """
    Root endpoint.
    """
    return {"message": f"Hello from server {os.getenv('CONTAINER_PORT')}!"}


@app.post("/dockerrest")
async def docker_rest(request: Request):
    """
    Handle Docker-related requests.
    """
    try:
        req_data = await request.json()
        req_method = req_data.get("req_method")

        if req_method == "logs":
            req_container = client.containers.get(req_data["req_model"])
            res_logs = req_container.logs().decode("utf-8")
            return JSONResponse({"result": 200, "result_data": res_logs})

        elif req_method == "network":
            req_container = client.containers.get(req_data["req_container_name"])
            stats = req_container.stats(stream=False)
            return JSONResponse({"result": 200, "result_data": stats})

        elif req_method == "list":
            res_container_list = client.containers.list(all=True)
            return JSONResponse([container.attrs for container in res_container_list])

        elif req_method == "delete":
            req_container = client.containers.get(req_data["req_model"])
            req_container.stop()
            req_container.remove(force=True)
            return JSONResponse({"result": 200})

        elif req_method == "stop":
            req_container = client.containers.get(req_data["req_model"])
            req_container.stop()
            return JSONResponse({"result": 200})

        elif req_method == "start":
            req_container = client.containers.get(req_data["req_model"])
            req_container.start()
            return JSONResponse({"result": 200})

        elif req_method == "create":
            container_name = str(req_data["req_model"]).replace("/", "_")
            res_db_gpu = await r.get("db_gpu")
            db_gpu = json.loads(res_db_gpu) if res_db_gpu else []

            # Check if model is already downloaded
            if req_data["req_model"] in [g["used_models"] for g in db_gpu]:
                return JSONResponse({"result": 302, "result_data": "Model already downloaded. Trying to start container..."})

            # Check if ports are already in use
            if req_data["req_port_vllm"] in [g["used_ports"] for g in db_gpu] or req_data["req_port_model"] in [g["used_ports"] for g in db_gpu]:
                return JSONResponse({"result": 409, "result_data": "Error: Port already in use"})

            # Check memory usage
            current_gpu_info = get_gpu_info()
            if current_gpu_info and current_gpu_info[0]["mem_util"] > 50:
                for running_model in [g["running_model"] for g in db_gpu]:
                    req_container = client.containers.get(running_model)
                    req_container.stop()

            # Wait for memory to free up
            for _ in range(10):
                current_gpu_info = get_gpu_info()
                if current_gpu_info and current_gpu_info[0]["mem_util"] <= 80:
                    break
                await asyncio.sleep(1)
            else:
                return JSONResponse({"result": 500, "result_data": "Error: Memory > 80%"})

            # Add new container data to Redis
            await redis_add(
                gpu=666,
                running_model=container_name,
                port_vllm=req_data["req_port_vllm"],
                port_model=req_data["req_port_model"],
                used_ports=f'{req_data["req_port_vllm"]},{req_data["req_port_model"]}',
                used_models=req_data["req_model"],
            )

            # Start the container
            res_container = client.containers.run(
                "vllm/vllm-openai:latest",
                command=f'--model {req_data["req_model"]}',
                name=container_name,
                runtime=req_data["req_runtime"],
                volumes={"/home/cloud/.cache/huggingface": {"bind": "/root/.cache/huggingface", "mode": "rw"}},
                environment={"HUGGING_FACE_HUB_TOKEN": os.getenv("HFTK")},
                ports={f'{req_data["req_port_vllm"]}/tcp': ("0.0.0.0", req_data["req_port_model"])},
                ipc_mode="host",
                device_requests=[device_request],
                detach=True,
            )
            return JSONResponse({"result": 200, "result_data": res_container.id})

        else:
            raise HTTPException(status_code=400, detail="Invalid request method")

    except Exception as e:
        logger.error(f"Error in Docker REST endpoint: {e}")
        return JSONResponse({"result": 500, "result_data": str(e)})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("CONTAINER_PORT")))