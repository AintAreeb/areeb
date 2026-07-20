# German Tutor — Your Own Fine-Tuned Model (Free Stack)

## What this is
A pipeline to turn your German-tutor dialogue script into a small fine-tuned
language model, trained for free on Google Colab, and run either locally or
hosted for free.

## Files
- `prepare_dataset.py` — converts a `{{user}}` / `{{char}}` transcript into
  a JSONL fine-tuning dataset.
- `raw_dialogue.txt` — sample transcript (replace with your full one).
- `dataset.jsonl` — generated training data (output of the script above).
- `train_colab.ipynb` — Colab notebook that fine-tunes a small open model
  (Llama 3.2 3B) on your dataset using LoRA (via Unsloth).

## Step-by-step

### 1. Build your dataset
Put your full dialogue transcript (all your example conversations, more the
better — aim for 200+ turns if you can) into a `.txt` file using the
`{{user}}: ...` / `{{char}}: ...` format, then run:

```bash
python prepare_dataset.py my_full_transcript.txt dataset.jsonl --system system_prompt.txt
```

`system_prompt.txt` should contain your tutor's rules (you can reuse the
lean prompt from before, or leave `--system` off for no system message).

More data = better results. If you only have the one sample I used, the
model will barely learn anything — you'll want to generate a lot more
example turns (I can help you generate synthetic ones once you tell me
what scenarios/books you want covered).

### 2. Push to GitHub
```bash
git init
git add prepare_dataset.py raw_dialogue.txt dataset.jsonl train_colab.ipynb README.md
git commit -m "German tutor fine-tuning pipeline"
git remote add origin https://github.com/<your-username>/german-tutor.git
git push -u origin main
```

### 3. Train on Colab (free GPU)
- Go to https://colab.research.google.com
- Open `train_colab.ipynb` (upload it, or open directly from your GitHub repo via File -> Open notebook -> GitHub tab)
- Runtime -> Change runtime type -> **T4 GPU**
- Run cells top to bottom. Upload your `dataset.jsonl` when prompted.
- Training a few hundred examples for 3 epochs on a free T4 takes roughly 10–30 minutes.

### 4. Get your model out
Two options, both in the notebook:
- **LoRA adapters** (small, few hundred MB): push to Hugging Face Hub (free) for hosted use.
- **GGUF export**: download the `.gguf` file and run it **locally for free** with [Ollama](https://ollama.com):
  ```bash
  ollama create german-tutor -f Modelfile
  ollama run german-tutor
  ```
  Where `Modelfile` just contains:
  ```
  FROM ./german_tutor_gguf/model-q4_k_m.gguf
  ```

### 5. Deploy (optional, if you want it on a server instead of local)
- **Hugging Face Spaces** (free): host a small Gradio/Streamlit chat UI backed by your model.
- Or keep it fully local with Ollama + a simple script — no server needed if it's just for you.

## Reality check
A 3B model fine-tuned on a small custom dataset won't be as fluent or
reliable as calling a big hosted model (GPT/Claude/Gemini) through a free
API. If lesson-generation *quality* matters more to you than "I trained it
myself," the prompt-based approach from earlier (free API + your system
prompt) will give a noticeably smoother experience. This fine-tuning path
is worth it if the goal is specifically learning how model training works
or having something that runs 100% offline/locally.
