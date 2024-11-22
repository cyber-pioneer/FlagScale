## Setup

[Install vLLM](../../../README.md#setup)


[Prepare Qwen data](https://www.modelscope.cn/models/Qwen/Qwen2.5-7B-Instruct/summary)

```shell
pip install modelscope
modelscope download --model Qwen/Qwen2.5-7B-Instruct --local_dir /models/
```

## Serve Qwen

```shell
cd FlagScale
python run.py -cp examples/qwen/conf/ -cn config.yaml
```


## Call Qwen
```shell
curl http://127.0.0.1:4567/v1/chat/completions -H "Content-Type: application/json" -d '{
        "model": "/models/qwen/weights/",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Introduce Bruce Lee in details."}
        ]
    }'
```




