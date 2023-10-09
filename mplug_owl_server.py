import torch
import numpy as np
import random
import os
import requests
from pathlib import Path
from flask import Flask, json, request, send_file
from mplug_owl_video.modeling_mplug_owl import MplugOwlForConditionalGeneration
from transformers import AutoTokenizer
from mplug_owl_video.processing_mplug_owl import MplugOwlImageProcessor, MplugOwlProcessor


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# 设置随机数种子
setup_seed(0)

pretrained_ckpt = '/data/mfishzhang/model_weigths/mplug-owl-llama-7b-video'
vision_model = MplugOwlForConditionalGeneration.from_pretrained(
    pretrained_ckpt,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
image_processor = MplugOwlImageProcessor.from_pretrained(pretrained_ckpt)
tokenizer = AutoTokenizer.from_pretrained(pretrained_ckpt)
processor = MplugOwlProcessor(image_processor, tokenizer)

# device = torch.device("cuda:2")
# vision_model.to(device)

print("success")

api = Flask(__name__)

@api.route('/api/mplug_owl_chat', methods=['POST'])
def chat_main():
    # print(request)
    # 保存上传的文件，并构造输入样本格式
    video_file = request.files['video']
    question = request.form['question']
    print(question)

    # with open("./temp.mp4", "wb") as f:
    #     f.write(video_file)
    video_file.save("./temp.mp4")
    video_list = ['./temp.mp4']

    prompts = [
        f'''The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
        Human: <|video|> 
        Human: {question}
        AI: ''']

    # print(prompts)

    generate_kwargs = {
        'do_sample': True,
        'top_k': 5,
        'max_length': 512,
        'temperature': 0.1
    }

    inputs = processor(text=prompts, videos=video_list, num_frames=4, return_tensors='pt')
    inputs = {k: v.bfloat16() if v.dtype == torch.float else v for k, v in inputs.items()}
    inputs = {k: v.to(vision_model.device) for k, v in inputs.items()}

    # 模型预估
    with torch.no_grad():
        res = vision_model.generate(**inputs, **generate_kwargs)
    answer = tokenizer.decode(res.tolist()[0], skip_special_tokens=True)

    print(answer)
    # 返回结果
    if os.path.exists("./temp.mp4"):
        os.remove("./temp.mp4")
    return answer


if __name__ == '__main__':
    api.run(host='0.0.0.0', port=30400)
