from flask import Flask, request, jsonify
import torch
import datetime

from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
app = Flask(__name__)

REPO = "replit/replit-code-v1-3b"
token = ''
MAX_INPUT_TOKENS = 1024
device = "cuda" if torch.cuda.is_available() else "cpu"

if device == "cuda":
    device_ids = [0, 1]  # 指定两个GPU
    model = AutoModelForCausalLM.from_pretrained(REPO, use_auth_token=token, trust_remote_code=True,
                                                 low_cpu_mem_usage=True).to(device_ids[0], dtype=torch.bfloat16)
    # 将模型复制到多个 GPU 上
    model = torch.nn.DataParallel(model, device_ids=device_ids).module
else:
    model = AutoModelForCausalLM.from_pretrained(REPO, use_auth_token=token, trust_remote_code=True,
                                                 low_cpu_mem_usage=True)
print('init')
model.eval()

tokenizer = AutoTokenizer.from_pretrained(REPO, use_auth_token=token, trust_remote_code=True)
tokenizer.truncation_side = "left"
set_seed(42)

# from flask import Flask, request, jsonify
# import torch
# import datetime
#
# from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
# app = Flask(__name__)
#
# REPO = "replit/replit-code-v1-3b"
# token = ''
# MAX_INPUT_TOKENS = 1024
# device = "cuda" if torch.cuda.is_available() else "cpu"
# if device == "cuda":
#     model = AutoModelForCausalLM.from_pretrained(REPO, use_auth_token=token, trust_remote_code=True,
#                                                  low_cpu_mem_usage=True).to(device, dtype=torch.bfloat16)
# else:
#     model = AutoModelForCausalLM.from_pretrained(REPO, use_auth_token=token, trust_remote_code=True,
#                                                  low_cpu_mem_usage=True)
# print('init')
# model.eval()
#
# tokenizer = AutoTokenizer.from_pretrained(REPO, use_auth_token=token, trust_remote_code=True)
# tokenizer.truncation_side = "left"  #
# set_seed(42)


@app.route('/generate_code', methods=['POST'])
def generate_code():
    print('1t' + str(datetime.datetime.now()))
    data = request.get_json()
    prompt = data['prompt']
    max_new_tokens = data['max_new_tokens']
    temperature = data.get('temperature', 0.2)
    top_p = data.get('top_p', 0.9)
    top_k = data.get('top_k', None)
    use_cache = data.get('use_cache', True)
    repetition_penalty = data.get('repetition_penalty', 1.0)

    # Call the code_generation function with the given parameters
    result = code_generation(prompt, max_new_tokens, temperature, top_p, top_k, use_cache, repetition_penalty)

    # Return the result as a JSON response
    return jsonify({'result': result})


def code_generation(prompt, max_new_tokens, temperature=0.2, top_p=0.9, top_k=None, use_cache=True, repetition_penalty=1.0):
   # truncates the prompt to MAX_INPUT_TOKENS if its too long
    print('2t' + str(datetime.datetime.now()))
    x = tokenizer.encode(prompt, return_tensors="pt", max_length=MAX_INPUT_TOKENS, truncation=True).to(device)

    print('3t' + str(datetime.datetime.now()))
    print('max_new_tokens ', max_new_tokens)
    print('temperature ', temperature)
    print('tokenizer.pad_token_id ', tokenizer.pad_token_id)
    print('tokenizer.eos_token_id ', tokenizer.eos_token_id)
    print('top_p ', top_p)
    print('top_k ', top_k)
    print('use_cache ', use_cache)
    print('repetition_penalty ', repetition_penalty)
    print("Prompt shape: ", x.shape) # just adding to see in the space logs in prod
    y = model.generate(x,
                       max_new_tokens=max_new_tokens,
                       temperature=temperature,
                       pad_token_id=tokenizer.pad_token_id,
                       eos_token_id=tokenizer.eos_token_id,
                       top_p=top_p,
                       top_k=top_k,
                       use_cache=use_cache,
                       repetition_penalty=repetition_penalty
                    )
    print('4t' + str(datetime.datetime.now()))
    completion = tokenizer.decode(y[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    print('5t' + str(datetime.datetime.now()))
    completion = completion[len(prompt):]
    print('6t' + str(datetime.datetime.now()))
    return post_processing(prompt, completion)

def post_processing(prompt, completion):
    return prompt + completion

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8801, debug=True)