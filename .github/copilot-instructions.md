# LLMs Engineering - Copilot Instructions

## Project Overview
Professional training program for LLM Engineering with 8 hands-on projects demonstrating OpenAI API integration, Ollama local deployment, function calling, multimodal AI, RAG, fine-tuning, and autonomous agents. Content is primarily in Spanish (es-MX).

## Architecture & Structure
- **Root**: Weekly project notebooks (`week1-Project1.ipynb`, `week2-Project2.ipynb`)
- **Weekly Folders** (`week1/`, `week2/`, `week3/`): Daily lessons (`day1.ipynb`, `day2.ipynb`) and utilities (`scraper.py`)
- **Projects Completed**: Project 1 (Programming Tutor), Project 2 (FlightAI Multimodal Assistant)
- **Week 3 Focus**: HuggingFace Transformers library, tokenizers, local model loading with quantization

## Environment Configuration Pattern
Critical: Always use this exact pattern for environment setup in notebooks:

```python
from dotenv import load_dotenv
import os

# Load from global workspace .env
load_dotenv(dotenv_path='/workspace/.env', override=True)

# Validate immediately
ollama_base_url = os.getenv('OLLAMA_BASE_URL')
ollama_api_key = os.getenv('OLLAMA_API_KEY')
openai_api_key = os.getenv('OPENAI_API_KEY')

print(f"Ollama: {ollama_base_url}")  # Always confirm loading
```

**Never** assume env vars are set - always validate and print confirmation.

## LLM Integration Patterns

### Ollama (Local/Cloud) - Standard Pattern
```python
from openai import OpenAI

client = OpenAI(
    base_url=f"{ollama_base_url}/v1",  # OpenAI-compatible endpoint
    api_key=ollama_api_key
)

# Models used: deepseek-v3.1:671b-cloud, qwen3-coder:480b-cloud
```

### OpenAI (Frontier Models)
```python
client = OpenAI(api_key=openai_api_key)

# Cost-conscious model choices:
# - gpt-4o-mini: $0.15/$0.60 per 1M tokens (chat)
# - dall-e-3: ~$0.04 per image
# - tts-1: $15 per 1M chars (cheaper than tts-1-hd)
```

### Streaming Responses (Project 1 Pattern)
```python
response = client.chat.completions.create(
    model=model_name,
    messages=messages,
    stream=True  # Enable typewriter effect
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end='')
```

## Function Calling / Tools (Project 2)
Define tools with OpenAI schema, execute locally:

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_ticket_price",
        "description": "Get flight ticket price",
        "parameters": {
            "type": "object",
            "properties": {"destination": {"type": "string"}},
            "required": ["destination"]
        }
    }
}]

response = client.chat.completions.create(
    model=model,
    messages=messages,
    tools=tools
)

# Check tool_calls in response, execute function, append result to messages
```

**Critical**: DeepSeek-V3.1 671B (Ollama) has strong tool support; verify compatibility for other models.

## Multimodal Patterns

### Image Generation (Pollinations.AI - Free)
```python
url = f"https://image.pollinations.ai/prompt/{prompt}"
response = requests.get(url)
image = Image.open(BytesIO(response.content))
```

### Text-to-Speech (gTTS with Speed Control)
```python
from gtts import gTTS
from pydub import AudioSegment

tts = gTTS(text=message, lang='es', slow=False)
audio_fp = BytesIO()
tts.write_to_fp(audio_fp)
audio_fp.seek(0)

# Speed adjustment (1.1x = 10% faster)
audio = AudioSegment.from_file(audio_fp, format="mp3")
audio_fast = audio._spawn(audio.raw_data, overrides={'frame_rate': int(audio.frame_rate * 1.1)})
```

### Audio Transcription (Whisper)
```python
import whisper
model = whisper.load_model("base")
result = model.transcribe(audio_path)
```

## HuggingFace Transformers (Week 3)

### Authentication Pattern
```python
from google.colab import userdata  # Colab-specific; adapt for local
from huggingface_hub import login

hf_token = userdata.get('HUGGINGFACE_API_KEY')
if hf_token.startswith('Bearer '):
    hf_token = hf_token.replace('Bearer ', '')
login(hf_token.strip())
```

### Quantization for Large Models (4-bit)
```python
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    device_map="auto",
    quantization_config=quant_config
)

# Apply chat template for instruct models
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
```

**Models Used**: Llama 3.1, Phi-3, Gemma 3, Qwen 3, Mixtral

### Memory Management
```python
# Check memory footprint
memory_mb = model.get_memory_footprint() / 1e6
print(f"Memory: {memory_mb:,.1f} MB")

# Cleanup after inference
del inputs, outputs, model
torch.cuda.empty_cache()
```

## Web Scraping Pattern (week1/scraper.py)
```python
from bs4 import BeautifulSoup
import requests

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36..."
}

response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.content, "html.parser")

# Clean irrelevant tags
for tag in soup.body(["script", "style", "img", "input"]):
    tag.decompose()

text = soup.body.get_text(separator="\n", strip=True)
```

**Always** include `User-Agent` header to avoid 403/blocking.

## Gradio UI Pattern (Project 2)
```python
import gradio as gr

def chat_interface(message, history):
    # Process message, call LLM, return response
    return response_text, generated_image, audio_bytes

demo = gr.ChatInterface(
    chat_interface,
    type="messages",
    additional_outputs=[gr.Image(), gr.Audio()]
)
demo.launch()
```

## Notebook Conventions
- **Markdown First**: Explain concept before code
- **Cell Modularity**: Each cell should run independently when possible
- **Rich Display**: Use `IPython.display.Markdown(response)` for formatted LLM outputs
- **Provider Selection**: Use `USE_PROVIDER` variable to toggle between 'ollama' and 'openai'
- **Error Handling**: Validate API keys/services before calls, provide clear error messages

## Dependencies by Project
- **Project 1**: `openai`, `python-dotenv`, `beautifulsoup4`, `requests`, `IPython`
- **Project 2**: Add `gradio`, `gtts`, `whisper`, `pydub`, `Pillow`
- **Week 3**: Add `transformers`, `torch`, `bitsandbytes`, `sentencepiece`, `accelerate`, `huggingface-hub`

## Communication Style
- **Language**: Spanish (es-MX) for markdown cells and user-facing content
- **Tone**: Professional, educational, no emojis
- **Comments**: Technical and concise, in Spanish or English
- **Docstrings**: English preferred for code reusability
