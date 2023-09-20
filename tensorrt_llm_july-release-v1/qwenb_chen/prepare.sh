cp -r /root/workspace/trt2023_qwen7-b/tensorrt_llm_july-release-v1/qwenb_chen /root/workspace/tensorrt_llm_july-release-v1/
cp -r /root/workspace/tensorrt_llm_july-release-v1/tensorrt_llm/libs /root/workspace/tensorrt_llm_july-release-v1/qwenb_chen/tensorrt_llm
rm /root/workspace/trt2023/QWen-7B-Chat/config.json
cp -r /root/workspace/trt2023_qwen7-b/tensorrt_llm_july-release-v1/qwenb_chen/qwen_7b_chat/config.json /root/workspace/trt2023/QWen-7B-Chat