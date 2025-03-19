import vllm
from vllm import LLM
from vllm.sampling_params import SamplingParams

from flagscale.inference.arguments import parse_config


def inference_qwen(config):

    prompts = config.generate.get("prompts", "")

    llm_cfg = config.get("llm", {})
    llm = LLM(**llm_cfg)

    sampling_cfg = config.generate.get("sampling", {})
    sampling_params = SamplingParams(**sampling_cfg)

    result = llm.generate(prompts, sampling_params)
    print("========= prompt ========= ", prompts, flush=True)
    print("===== generate result ==== ", result[0].outputs[0].text, flush=True)


if __name__ == "__main__":
    config = parse_config()
    print(f"[vllm.__file__] {vllm.__file__}")
    inference_qwen(config)
