import os, sys
import logging
import yaml
#os.environ["RAY_LOG_TO_STDERR"] = "1"
#os.environ["RAY_BACKEND_LOG_LEVEL"] = "info"  # 或 debug
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)
#root_logger = logging.getLogger()
#root_logger.setLevel(logging.DEBUG)
#handler = logging.StreamHandler(sys.stdout)
#handler.setLevel(logging.DEBUG)
#formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
#handler.setFormatter(formatter)
#root_logger.addHandler(handler)



import ray
import subprocess
import argparse


# 初始化 ray
#ray.init(log_to_driver=True, logging_level=logging.DEBUG, configure_logging=True)
ray.init(log_to_driver=True, logging_level=logging.DEBUG)
#logging.getLogger("vllm").setLevel(logging.DEBUG) 

# 定义一个 Ray 任务来启动 vllm serve
@ray.remote(num_gpus=1)
def start_vllm_serve(args):
    # 构建命令
    # command = [
    #     'vllm', 'serve',
    #     model_path,
    #     '--trust-remote-code' if trust_remote_code else '',
    #     f'--max-model-len={max_model_len}',
    #     f'--max-num-seqs={max_num_seqs}',
    #     '--enable-chunked-prefill' if enable_chunked_prefill else '',
    #     f'--tensor-parallel-size={tensor_parallel_size}',
    #     f'--port={port}'
    # ]
    
    # 过滤掉空字符串
    breakpoint()
    command = [arg for arg in command if arg]

#    command = [
#        "vllm", "serve", "/models/Qwen2.5-7B-Instruct/",
#        "--trust-remote-code",
#        "--max-model-len=32768",
#        "--max-num-seqs=256",
#        "--enable-chunked-prefill",
#        "--tensor-parallel-size=1",
#        "--port=9013"
#    ]

    # Start the subprocess

    # 打印命令以便调试
    print(f"Starting vllm serve with command: {' '.join(command)}")
    process = subprocess.Popen(command, stdout=sys.stdout, stderr=sys.stderr)
    # 启动 vllm serve
    #process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
   
    # 捕获输出和错误
    stdout, stderr = process.communicate()
    
    # 打印输出和错误
    print("Standard Output:")
    print(stdout.decode())
    print("Standard Error:")
    print(stderr.decode())
   
    return process.returncode

# 定义主函数
def main():
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description='Start vllm serve with Ray')
    
    # 添加命令行参数
    parser.add_argument('--model-path', type=str, required=True, help='Path to the model')
    
    args = parser.parse_args()
    vllm_args = yaml.safe_load(args.model_path)
    breakpoint()
    # 启动 vllm serve
    result = start_vllm_serve.remote(vllm_args)# 获取结果
    return_code = ray.get(result)
    
    # 打印返回码
    print(f"vllm serve exited with return code: {return_code}")

if __name__ == "__main__":
    main()