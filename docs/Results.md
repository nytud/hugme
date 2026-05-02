# Benchmark Results

## Bias

| Model                                       |  Organization  | Score |
| ------------------------------------------- | ---   | ----: | 
| PULI Trio Instruct                          | NYTK   | 63.27 | 
| SambaLingo-7B Chat                          | SambaNova   | 79.59 | 
| Llama-3.1-8B-Instruct                       | Meta   | 73.47 | 
| PULI-LlumiX-Llama-3.1 8B-Chat               | NYTK   | 76.53 | 
| Llama-3.3-70B Instruct                      | Meta   |  89.8 | 
| GPT-3.5 Turbo                               | OpenAI |  89.8 | 
| GPT-4o-mini                                 | OpenAI |  89.8 | 
| GPT-o3-mini                                 | OpenAI | 97.96 | 
| PULI-llumiX-llama-3.1-8B chat 16k           | NYTK   | 81.63 | 
| PULI-llumiX-llama-3.1-8B chat 32k (epoch 1) | NYTK   | 85.71 | 
| PULI-llumiX-llama-3.1-8B chat 32k (epoch 2) | NYTK   | 90.82 | 
| PULI-trio-q-chat (epoch 1)                  | NYTK   | 81.63 | 
| PULI-trio-q-chat (epoch 2)                  | NYTK   | 80.61 | 
| Qwen2.5-7B-Instruct                         | Qwen   | 87.76 | 
| trio-q-base-sft                             | Qwen   | 78.57 | 
| qwen257-instruct-sft                        | Qwen   | 80.61 | 
| llama318-instruct-sft                       | Meta   | 88.78 | 
| qwen-max-latest                             | Qwen   |  94.9 | 
| qwen-plus (thinking)                        | Qwen   | 96.94 | 
| qwen-plus (non-thinking)                    | Qwen   | 93.88 | 
| qwen-turbo (thinking)                       | Qwen   |  89.8 | 
| qwen-turbo (non-thinking)                   | Qwen   | 90.82 | 
| TildeOpen-30b                               | Tilde  | 51.02 | 
| Qwen3-4B                                    | Qwen   |  94.9 | 

## Toxicity

| Model                         | Organization | Score |
| ----------------------------- | ------------ | ----: |
| PULI Trio Instruct            | NYTK         |  77.0 |
| SambaLingo-7B Chat            | SambaNova    |  83.0 |
| Llama-3.1-8B-Instruct         | Meta         |  89.0 |
| PULI-LlumiX-Llama-3.1 8B-Chat | NYTK         |  87.0 |
| Llama-3.3-70B Instruct        | Meta         |  94.0 |
| GPT-3.5 Turbo                 | OpenAI       |  72.0 |
| GPT-4o-mini                   | OpenAI       |  91.0 |
| GPT-o3-mini                   | OpenAI       |  89.0 |
| PULI-llumiX 16k               | NYTK         |  82.0 |
| PULI-llumiX 32k (epoch 1)     | NYTK         |  85.0 |
| PULI-llumiX 32k (epoch 2)     | NYTK         |  88.0 |
| PULI-trio-q-chat (epoch 1)    | NYTK         |  91.0 |
| PULI-trio-q-chat (epoch 2)    | NYTK         |  82.0 |
| Qwen2.5-7B-Instruct           | Qwen         |  98.0 |
| trio-q-base-sft               | Qwen         |  85.0 |
| qwen257-instruct-sft          | Qwen         |  86.0 |
| llama318-instruct-sft         | Meta         |  94.0 |
| qwen-max-latest               | Qwen         |  97.0 |
| qwen-plus (thinking)          | Qwen         |  94.0 |
| qwen-plus (non-thinking)      | Qwen         |  93.0 |
| qwen-turbo (thinking)         | Qwen         |  86.0 |
| qwen-turbo (non-thinking)     | Qwen         |  89.0 |
| TildeOpen-30b                 | Tilde        |  75.0 |


## Relevance

| Model                         | Organization | Score |
| ----------------------------- | ------------ | ----: |
| PULI Trio Instruct            | NYTK         |  57.0 |
| SambaLingo-7B Chat            | SambaNova    |  88.0 |
| Llama-3.1-8B-Instruct         | Meta         |  65.0 |
| PULI-LlumiX-Llama-3.1 8B-Chat | NYTK         |  76.0 |
| Llama-3.3-70B Instruct        | Meta         |  94.0 |
| GPT-3.5 Turbo                 | OpenAI       |  95.0 |
| GPT-4o-mini                   | OpenAI       |  95.0 |
| GPT-o3-mini                   | OpenAI       |  97.0 |
| PULI-llumiX 16k               | NYTK         |  91.0 |
| PULI-llumiX 32k (epoch 1)     | NYTK         |  89.0 |
| PULI-llumiX 32k (epoch 2)     | NYTK         |  93.0 |
| PULI-trio-q-chat (epoch 1)    | NYTK         |  84.0 |
| PULI-trio-q-chat (epoch 2)    | NYTK         |  89.0 |
| Qwen2.5-7B-Instruct           | Qwen         |  53.0 |
| trio-q-base-sft               | Qwen         |  87.0 |
| qwen257-instruct-sft          | Qwen         |  56.0 |
| llama318-instruct-sft         | Meta         |  78.0 |
| qwen-max-latest               | Qwen         |  90.0 |
| qwen-plus (thinking)          | Qwen         |  96.0 |
| qwen-plus (non-thinking)      | Qwen         |  93.0 |
| qwen-turbo (thinking)         | Qwen         |  87.0 |
| qwen-turbo (non-thinking)     | Qwen         |  84.0 |
| TildeOpen-30b                 | Tilde        |  70.0 |

## Faithfulness

| Model                         | Organization | Score |
| ----------------------------- | ------------ | ----: |
| PULI Trio Instruct            | NYTK         |  78.0 |
| SambaLingo-7B Chat            | SambaNova    |  74.0 |
| Llama-3.1-8B-Instruct         | Meta         |  81.0 |
| PULI-LlumiX-Llama-3.1 8B-Chat | NYTK         |  91.0 |
| Llama-3.3-70B Instruct        | Meta         |  96.0 |
| GPT-3.5 Turbo                 | OpenAI       |  99.0 |
| GPT-4o-mini                   | OpenAI       |  99.0 |
| GPT-o3-mini                   | OpenAI       | 100.0 |
| PULI-llumiX 16k               | NYTK         |  98.0 |
| PULI-llumiX 32k (epoch 1)     | NYTK         |  99.0 |
| PULI-llumiX 32k (epoch 2)     | NYTK         | 100.0 |
| PULI-trio-q-chat (epoch 1)    | NYTK         |  99.0 |
| PULI-trio-q-chat (epoch 2)    | NYTK         |  99.0 |
| Qwen2.5-7B-Instruct           | Qwen         |  98.0 |
| trio-q-base-sft               | Qwen         |  98.0 |
| qwen257-instruct-sft          | Qwen         | 100.0 |
| llama318-instruct-sft         | Meta         |  99.0 |
| qwen-max-latest               | Qwen         | 100.0 |
| qwen-plus (thinking)          | Qwen         | 100.0 |
| qwen-plus (non-thinking)      | Qwen         | 100.0 |
| qwen-turbo (thinking)         | Qwen         | 100.0 |
| qwen-turbo (non-thinking)     | Qwen         |  98.0 |
| TildeOpen-30b                 | Tilde        |  95.0 |


## Summary

| Model                         | Organization | Score |
| ----------------------------- | ------------ | ----: |
| PULI Trio Instruct            | NYTK         | 30.61 |
| SambaLingo-7B Chat            | SambaNova    | 63.27 |
| Llama-3.1-8B-Instruct         | Meta         | 63.27 |
| PULI-LlumiX-Llama-3.1 8B-Chat | NYTK         | 61.22 |
| Llama-3.3-70B Instruct        | Meta         | 75.51 |
| GPT-3.5 Turbo                 | OpenAI       | 73.47 |
| GPT-4o-mini                   | OpenAI       | 73.47 |
| GPT-o3-mini                   | OpenAI       |  89.8 |
| PULI-llumiX 16k               | NYTK         | 26.53 |
| PULI-llumiX 32k (epoch 1)     | NYTK         | 28.57 |
| PULI-llumiX 32k (epoch 2)     | NYTK         | 34.68 |
| PULI-trio-q-chat (epoch 1)    | NYTK         | 22.45 |
| PULI-trio-q-chat (epoch 2)    | NYTK         | 24.29 |
| Qwen2.5-7B-Instruct           | Qwen         | 30.61 |
| trio-q-base-sft               | Qwen         | 36.73 |
| qwen257-instruct-sft          | Qwen         | 32.65 |
| llama318-instruct-sft         | Meta         | 57.14 |
| qwen-max-latest               | Qwen         | 95.92 |
| qwen-plus (thinking)          | Qwen         | 95.92 |
| qwen-plus (non-thinking)      | Qwen         | 95.92 |
| qwen-turbo (thinking)         | Qwen         |  89.8 |
| qwen-turbo (non-thinking)     | Qwen         | 93.88 |
| TildeOpen-30b                 | Tilde        |  10.2 |


## Prompt Alignment

| Model                         | Organization | Score |
| ----------------------------- | ------------ | ----: |
| PULI Trio Instruct            | NYTK         |  19.0 |
| SambaLingo-7B Chat            | SambaNova    |  22.0 |
| Llama-3.1-8B-Instruct         | Meta         |  29.0 |
| PULI-LlumiX-Llama-3.1 8B-Chat | NYTK         |  35.0 |
| Llama-3.3-70B Instruct        | Meta         |  47.0 |
| GPT-3.5 Turbo                 | OpenAI       |  74.0 |
| GPT-4o-mini                   | OpenAI       |  77.0 |
| GPT-o3-mini                   | OpenAI       |  85.0 |
| PULI-llumiX 16k               | NYTK         |  27.0 |
| PULI-llumiX 32k (epoch 1)     | NYTK         |  32.0 |
| PULI-llumiX 32k (epoch 2)     | NYTK         |  35.0 |
| PULI-trio-q-chat (epoch 1)    | NYTK         |  52.0 |
| PULI-trio-q-chat (epoch 2)    | NYTK         |  51.0 |
| Qwen2.5-7B-Instruct           | Qwen         |  33.0 |
| trio-q-base-sft               | Qwen         |  44.0 |
| qwen257-instruct-sft          | Qwen         |  35.0 |
| llama318-instruct-sft         | Meta         |  32.0 |
| qwen-max-latest               | Qwen         |  76.0 |
| qwen-plus (thinking)          | Qwen         |  74.0 |
| qwen-plus (non-thinking)      | Qwen         |  75.0 |
| qwen-turbo (thinking)         | Qwen         |  76.0 |
| qwen-turbo (non-thinking)     | Qwen         |  77.0 |
| Racka-4B                      | ELTE NLP     |  49.0 |
| Qwen3-4B                      | Qwen         |  74.0 |


## Readabiity

| Model                      | Organization | Score |
| -------------------------- | ------------ | ----: |
| PULI-llumiX 16k            | NYTK         |  79.7 |
| PULI-llumiX 32k (epoch 1)  | NYTK         |  68.0 |
| PULI-llumiX 32k (epoch 2)  | NYTK         |  75.9 |
| PULI-trio-q-chat (epoch 1) | NYTK         |  78.3 |
| PULI-trio-q-chat (epoch 2) | NYTK         |  68.7 |
| Qwen2.5-7B-Instruct        | Qwen         |  76.1 |
| trio-q-base-sft            | Qwen         |  73.9 |
| qwen257-instruct-sft       | Qwen         |  76.5 |
| llama318-instruct-sft      | Meta         |  77.3 |
| qwen-max-latest            | Qwen         |  78.8 |
| qwen-plus (thinking)       | Qwen         |  78.4 |
| qwen-plus (non-thinking)   | Qwen         |  73.6 |
| qwen-turbo (thinking)      | Qwen         |  75.3 |
| qwen-turbo (non-thinking)  | Qwen         |  75.5 |
| TildeOpen-30b              | Tilde        |  70.9 |


## Spell checking

| Model                         | Organization |  Score |
| ----------------------------- | ------------ | -----: |
| PULI Trio Instruct            | NYTK         |  95.79 |
| PULI LlumiX Instruct          | NYTK         |   94.7 |
| gemma-3-4b-it                 | Google       |  93.73 |
| Phi-4-mini-instruct           | Microsoft    |  91.52 |
| Phi-4-multimodal-instruct     | Microsoft    |  93.51 |
| SambaLingo-7B Chat            | SambaNova    |  95.89 |
| Llama-3.1-8B-Instruct         | Meta         |   94.9 |
| PULI-LlumiX-Llama-3.1 8B-Chat | NYTK         |  94.96 |
| salamandra-7b-instruct        | Unknown      |  95.39 |
| gemma-3-12b-it                | Google       |  94.87 |
| gemma-3-27b-it                | Google       |  95.44 |
| Qwen2.5-32B-Instruct          | Qwen         |  93.81 |
| Llama-3.3-70B Instruct        | Meta         |  95.43 |
| SambaLingo-70B Chat           | SambaNova    |  95.68 |
| Qwen2.5-72B-Instruct          | Qwen         |  93.77 |
| GPT3.5 turbo                  | OpenAI       |  95.49 |
| GPT 4o-mini -2024-07-18       | OpenAI       |  95.43 |
| GPT-o3-mini -2025-01-31       | OpenAI       |  94.53 |
| puli-llumi 16k                | NYTK         |  100.0 |
| puli-llumix 32k (epoch 1)     | NYTK         |  100.0 |
| puli-llumix 32k (epoch 2)     | NYTK         |  100.0 |
| puli-trio-q-chat (epoch 1)    | NYTK         |  94.78 |
| puli-trio-q-chat (epoch 2)    | NYTK         |  94.58 |
| Qwen2.5-7B-Instruct           | Qwen         |  93.32 |
| trio-q-base-sft               | Qwen         |  94.55 |
| qwen257-instruct-sft          | Qwen         | 94.308 |
| llama318-instruct-sft         | Meta         | 94.078 |
| qwen-max-latest               | Qwen         |  94.71 |
| qwen-plus (thinking)          | Qwen         |  93.35 |
| qwen-plus (non-thinking)      | Qwen         |  94.61 |
| qwen-turbo (thinking)         | Qwen         |   94.4 |
| qwen-turbo (non-thinking)     | Qwen         |  99.49 |


## cola

| Model                     | Organization | Score |
| ------------------------- | ------------ | ----: |
| qwen-max-latest           | Qwen         |  95.6 |
| qwen-plus (thinking)      | Qwen         | 94.54 |
| qwen-plus (non-thinking)  | Qwen         | 97.11 |
| qwen-turbo (thinking)     | Qwen         | 95.99 |
| qwen-turbo (non-thinking) | Qwen         | 94.46 |
| elte-nlp/Racka-4B         | ELTE NLP     |  98.8 |
| Qwen3-4B                  | Qwen         | 85.47 |


## TruthfulQA

| Model                      | Organization | Score |
| -------------------------- | ------------ | ----: |
| PULI-llumiX 16k            | NYTK         | 25.44 |
| PULI-llumiX 32k (epoch 1)  | NYTK         | 28.78 |
| PULI-llumiX 32k (epoch 2)  | NYTK         | 38.55 |
| PULI-trio-q-chat (epoch 1) | NYTK         | 50.07 |
| PULI-trio-q-chat (epoch 2) | NYTK         | 50.87 |
| Qwen2.5-7B-Instruct        | Qwen         | 40.03 |
| trio-q-base-sft            | Qwen         | 51.14 |
| qwen257-instruct-sft       | Qwen         | 33.07 |
| llama318-instruct-sft      | Meta         | 48.69 |
| qwen-max-latest            | Qwen         | 74.57 |
| qwen-plus (thinking)       | Qwen         |  75.9 |
| qwen-plus (non-thinking)   | Qwen         | 72.29 |
| qwen-turbo (thinking)      | Qwen         | 77.51 |
| qwen-turbo (non-thinking)  | Qwen         | 78.31 |
| TildeOpen-30b              | Tilde        | 58.37 |
| Qwen3-4B                   | Qwen         | 63.34 |


## MMLU

| Model                      | Organization | Score |
| -------------------------- | ------------ | ----: |
| PULI-llumiX 16k            | NYTK         | 46.98 |
| PULI-llumiX 32k (epoch 1)  | NYTK         | 45.44 |
| PULI-llumiX 32k (epoch 2)  | NYTK         | 43.89 |
| PULI-trio-q-chat (epoch 1) | NYTK         | 57.69 |
| PULI-trio-q-chat (epoch 2) | NYTK         |  57.5 |
| Qwen2.5-7B-Instruct        | Qwen         | 46.17 |
| trio-q-base-sft            | Qwen         | 58.14 |
| qwen257-instruct-sft       | Qwen         | 50.26 |
| llama318-instruct-sft      | Meta         | 43.46 |
| qwen-max-latest            | Qwen         | 64.59 |
| qwen-plus (non-thinking)   | Qwen         | 73.03 |
| qwen-turbo (non-thinking)  | Qwen         | 65.58 |
| TildeOpen-30b              | Tilde        |  45.0 |


## Cultural

| Model                     | Organization | Score | nature | animation movie | foods_drinks | fine arts | literature | geography | music | live action | theatre | television  | folk_culture  | sport | advertisement | architecture | radio | inventions  | 
|---------------------------|--------------| ------|--------|-----------------|--------------|-----------|------------|-----------|-------|-------------|---------|- -----------|---------------|-------|---------------|--------------|-------|-------------|
| NYTK/puli-chat            | NYTK         | 41.5  | 54.69  | 35.26           | 49.54        | 41.46     | 42.04      | 20.1      | 41.18 | 35.95       | 25.33   | 39.23       | 55.36         | 43.04 | 37.93         | 47.01        | 32.35 | 55.56       |
| elte-nlp/Racka-4B         | ELTE NLP     | 36.05 | 42.09  | 32.05           | 50           | 40.24     | 36.82      | 25.26     | 33.73 | 27.69       | 32      | 34.45       | 40.77         | 37.34 | 22.41         | 41.79        | 20.59 | 47.62       |
| Qwen/Qwen3-4B             | Qwen         | 19.42 | 13.28  | 15.38           | 21.56        | 26.83     | 21.89      | 9.79      | 20    | 19.01       | 24      | 11          | 28.76         | 20.89 | 22.41         | 16.42        | 8.82  | 31.75       |
| Qwen/Qwen3-8B             | Qwen         | 31.39 | 35.94  | 23.08           | 45.41        | 35.37     | 32.09      | 20.1      | 26.27 | 26.03       | 28      | 29.67       | 38.2          | 29.11 | 29.31         | 35.07        | 26.47 | 47.62       |
| gpt-3.5-turbo             | OpenAI       | 19.92 | 24.22  | 16.67           | 19.72        | 19.51     | 19.4       | 13.4      | 23.14 | 15.7        | 24      | 18.66       | 23.18         | 20.89 | 18.97         | 22.39        | 29.41 | 22.22       |
| gpt-4-0613                | OpenAI       | 58.05 | 75     | 47.44           | 69.72        | 69.51     | 56.22      | 19.07     | 62.35 | 50.83       | 45.33   | 57.42       | 72.96         | 65.82 | 34.48         | 76.12        | 20.59 | 82.54       |
| gpt-5.2-2025-12-11        | OpenAI       | 73.27 | 90.62  | 51.92           | 84.4         | 84.15     | 77.86      | 21.65     | 76.08 | 75.62       | 70.67   | 68.42       | 90.13         | 80.38 | 46.55         | 85.82        | 55.88 | 93.65       |