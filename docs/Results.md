# Benchmark Results

## LLM-as-a judge Modules

| Model                                       |  Organization   | Bias  | Toxicity  | Relevance | Faithfullness | Summary   | Prompt Alignment  |
| ------------------------------------------- | ---             | ----: | ---:      | ---:      | ---:          | ---:      | ---:              |
| PULI Trio Instruct                          | NYTK            | 63.27 | 77.0      | 57.0      |  78.0         | 30.61     | 19.0              |
| PULI-llumiX-llama-3.1-8B chat 16k           | NYTK            | 81.63 | 82.0      | 91.0      |  98.0         | 26.53     | 27.0              |
| PULI-LlumiX-Llama-3.1 8B-Chat               | NYTK            | 76.53 | 87.0      | 76.0      |  91.0         | 61.22     | 35.0              |
| PULI-llumiX-llama-3.1-8B chat 32k (epoch 1) | NYTK            | 85.71 | 85.0      | 89.0      |  99.0         | 28.57     | 32.0              |
| PULI-llumiX-llama-3.1-8B chat 32k (epoch 2) | NYTK            | 90.82 | 88.0      | 93.0      | 100.0         | 34.68     | 35.0              |
| PULI-trio-q-chat (epoch 1)                  | NYTK            | 81.63 | 91.0      | 84.0      |  99.0         | 22.45     | 52.0              |
| PULI-trio-q-chat (epoch 2)                  | NYTK            | 80.61 | 82.0      | 89.0      |  99.0         | 24.29     | 51.0              |
| Racka-4B                                    | ELTE NLP        | 90.82 | 97        | 66        | 97            | 85.71     | 45.0              |
| Llama-3.1-8B-Instruct                       | Meta            | 73.47 | 89.0      | 65.0      |  81.0         | 63.27     | 29.0              |
| Llama-3.3-70B Instruct                      | Meta            |  89.8 | 94.0      | 94.0      |  96.0         | 75.51     | 47.0              |
| llama318-instruct-sft                       | Meta            | 88.78 | 94.0      | 78.0      |  99.0         | 57.14     | 32.0              |
| GPT-5.2-2025-12-11                          | OpenAI          | -     | -         | -         | -             | -         | 87                |
| GPT-5.4-2026-03-05                          | OpenAI          | -     | -         | -         | -             | -         | 87                |
| GPT-4o-mini                                 | OpenAI          |  89.8 | 91.0      | 95.0      |  99.0         | 73.47     | 77.0              |
| GPT-3.5 Turbo                               | OpenAI          |  89.8 | 72.0      | 95.0      |  99.0         | 73.47     | 74.0              |
| GPT-o3-mini                                 | OpenAI          | 97.96 | 89.0      | 97.0      | 100.0         |  89.8     | 85.0              |
| Qwen3-4B                                    | Qwen            |  94.9 | 92        | 66        | 100           | 85.71     | 74.0              |
| Qwen2.5-7B-Instruct                         | Qwen            | 87.76 | 98.0      | 53.0      |  98.0         | 30.61     | 33.0              |
| qwen257-instruct-sft                        | Qwen            | 80.61 | 86.0      | 56.0      | 100.0         | 32.65     | 35.0              |
| qwen-max-latest                             | Qwen            |  94.9 | 97.0      | 90.0      | 100.0         | 95.92     | 76.0              |
| qwen-plus (thinking)                        | Qwen            | 96.94 | 94.0      | 96.0      | 100.0         | 95.92     | 74.0              |
| qwen-plus (non-thinking)                    | Qwen            | 93.88 | 93.0      | 93.0      | 100.0         | 95.92     | 75.0              |
| qwen-turbo (thinking)                       | Qwen            |  89.8 | 86.0      | 87.0      | 100.0         |  89.8     | 76.0              |
| qwen-turbo (non-thinking)                   | Qwen            | 90.82 | 89.0      | 84.0      |  98.0         | 93.88     | 77.0              |
| trio-q-base-sft                             | Qwen            | 78.57 | 85.0      | 87.0      |  98.0         | 36.73     | 44.0              |
| SambaLingo-7B Chat                          | SambaNova       | 79.59 | 83.0      | 88.0      |  74.0         | 63.27     | 22.0              |
| TildeOpen-30b                               | Tilde           | 51.02 | 75.0      | 70.0      |  95.0         |  10.2     | -                 |

## Language Proficiency & World Knowledge

| Model                                 | Organization | Readability |  Spell checking  | Cola  | TruthfulQA    | MMLU  |
| -----------------------------         | ------------ | -----:      |  ---:            | ---:  | ---:          | ---:  |
| PULI Trio Instruct                    | NYTK         | -           |  95.79           | -     | -             | -     |
| PULI LlumiX Instruct                  | NYTK         | -           |   94.7           | -     | -             | -     |
| PULI-LlumiX-Llama-3.1 8B-Chat         | NYTK         | -           |  94.96           | -     | -             | -     |
| puli-llumix-llama-3.1-8B chat 16k     | NYTK         | 79.7        |  100.0           | -     | 25.44         | 46.98 |
| puli-llumix 32k (epoch 1)             | NYTK         | 68.0        |  100.0           | -     | 28.78         | 45.44 |
| puli-llumix 32k (epoch 2)             | NYTK         | 75.9        |  100.0           | -     | 38.55         | 43.89 |
| puli-trio-q-chat (epoch 1)            | NYTK         | 78.3        |  94.78           | -     | 50.07         | 57.69 |
| puli-trio-q-chat (epoch 2)            | NYTK         | 68.7        |  94.58           | -     | 50.87         | 57.5  |
| Racka-4B                              | ELTE NLP     | 68.7        |  96.61           | 98.8  | 23.05         | 19.65 |
| Llama-3.1-8B-Instruct                 | Meta         | -           |   94.9           | -     | -             | -     |
| Llama-3.3-70B Instruct                | Meta         | -           |  95.43           | -     | -             | -     |
| llama318-instruct-sft                 | Meta         | 77.3        | 94.078           | -     | 48.69         | 43.46 |
| GPT 5.2-2025-12-11                    | OpenAI       | 75.7        | 93.41            | -     | 88.14         | 83.97 |
| GPT 5.4-2026-03-05                    | OpenAI       | 78.1        | 95.527           | -     | 88.68         | 86.49 |
| GPT 4o-mini -2024-07-18               | OpenAI       | -           |  95.43           | -     | -             | -     |
| GPT3.5 turbo                          | OpenAI       | -           |  95.49           | -     | -             | -     |
| GPT-o3-mini -2025-01-31               | OpenAI       | -           |  94.53           | -     | -             | -     |
| Qwen3-4B                              | Qwen         | -           |  -               | 85.47 | 63.34         | -     |
| Qwen2.5-72B-Instruct                  | Qwen         | -           |  93.77           | -     | -             | -     |
| Qwen2.5-32B-Instruct                  | Qwen         | -           |  93.81           | -     | -             | -     |
| Qwen2.5-7B-Instruct                   | Qwen         | 76.1        |  93.32           | -     | 40.03         | 46.17 |
| qwen257-instruct-sft                  | Qwen         | 76.5        | 94.308           | -     | -             | 50.26 |
| qwen-max-latest                       | Qwen         | 78.8        |  94.71           | 95.6  | 74.57         | 64.59 |
| qwen-plus (thinking)                  | Qwen         | 78.4        |  93.35           | 94.54 | 75.9          | -     |
| qwen-plus (non-thinking)              | Qwen         | 73.6        |  94.61           | 97.11 | 72.29         | 73.03 |
| qwen-turbo (thinking)                 | Qwen         | 75.3        |   94.4           | 95.99 | 77.51         | -     |
| qwen-turbo (non-thinking)             | Qwen         | 75.5        |  99.49           | 94.46 | 78.31         | 65.58 |
| trio-q-base-sft                       | Qwen         | 73.9        |  94.55           | -     | 51.14         | 58.14 |
| SambaLingo-70B Chat                   | SambaNova    | -           |  95.68           | -     | -             | -     |
| SambaLingo-7B Chat                    | SambaNova    | -           |  95.89           | -     | -             | -     |
| TileOpen-30b                          | Tilde        | 70.9        | -                | -     | 58.37         | -     |
| salamandra-7b-instruct                | BSC          | -           |  95.39           | -     | -             | -     |
| Phi-4-mini-instruct                   | Microsoft    | -           |  91.52           | -     | -             | -     |
| Phi-4-multimodal-instruct             | Microsoft    | -           |  93.51           | -     | -             | -     |
| gemma-3-12b-it                        | Google       | -           |  94.87           | -     | -             | -     |
| gemma-3-27b-it                        | Google       | -           |  95.44           | -     | -             | -     |
| gemma-3-4b-it                         | Google       | -           |  93.73           | -     | -             | -     |

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
