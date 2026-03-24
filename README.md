# install dep
```shell
pip install openai
pip install python-dotenv
```
# donwloade model
use qwen3.5-0.8b model
https://www.modelscope.cn/models/unsloth/Qwen3.5-0.8B-GGUF/files

# inference framework
The large model inference framework use llama.cpp

https://github.com/ggml-org/llama.cpp/tags

# start server
```
llama-server.exe -m  Qwen3.5-0.8B-Q8_0.gguf --host 0.0.0.0 -fa off -c 262144
```