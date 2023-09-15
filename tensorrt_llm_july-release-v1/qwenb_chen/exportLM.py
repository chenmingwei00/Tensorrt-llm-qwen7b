from transformers import AutoModelForCausalLM, AutoTokenizer
from qwen_7b_chat.modeling_qwen import QWenLMHeadModel
from transformers.generation import GenerationConfig
from qwen_7b_chat.configuration_qwen import QWenConfig
import argparse
# checkpoint_path="/root/workspace/trt2023/QWen-7B-Chat"

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--checkpoint_path',
                    type=str,
                    default='./qwenftModel',
                    help='file name of output directory',
                    required=False)
args = parser.parse_args()
checkpoint_path=args.checkpoint_path
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(checkpoint_path, device_map="auto", trust_remote_code=True).eval()
model = QWenLMHeadModel.from_pretrained(checkpoint_path, device_map="auto", trust_remote_code=True, fp16=True).eval()

model.generation_config = GenerationConfig.from_pretrained(
        checkpoint_path, trust_remote_code=True, resume_download=True,
    )

# os.system("rm -rv /root/.cache/huggingface/modules/transformers_modules/QWen-7B-Chat/modeling_qwen.py")
# os.system("copy /root/workspace/trt2023/QWen-7B-Chat/modeling_qwen.py /root/.cache/huggingface/modules/transformers_modules/QWen-7B-Chat/modeling_qwen.py")

prompt = "续写：RTX4090具有760亿个晶体管，16384个CUDA核心"

history, response = [], ''
import time
start_time=time.time()
response=model.chat(tokenizer, prompt, history=history)
print(response)
end_time=time.time()
consum_time=end_time-start_time
avge=consum_time/len(response[0])
print("函数运行时间：",avge)

print("函数运行时间：",consum_time)
