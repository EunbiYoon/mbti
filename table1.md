# Table 1: Evaluation results on downstream reasoning benchmarks within multi-agent systems

Competitive game-trained agents excel in the competitive MAD framework, while the cooperative-trained agent excels in the cooperative AutoGen framework. The generalist model performs robustly across both. **Bold** and <u>underlined</u> indicate the best and second-best scores, respectively.

| Setting | Model | Average | MATH | GSM8K | AQUA | AIME | AMC | MMLU | GPQA |
|---------|-------|-------:|------:|------:|------:|------:|------:|------:|------:|
| **Single Agent** | Qwen3-4B | 60.74 | 87.60 | 94.60 | 39.80 | 36.70 | 70.00 | 57.10 | **39.39** |
|  | SPIRAL | **63.75** | 87.50 | <u>94.80</u> | <u>51.20</u> | 36.70 | **80.00** | 58.70 | 37.37 |
|  | **MARSHAL** |  |  |  |  |  |  |  |  |
|  | Tic-Tac-Toe | <u>63.54</u> | <u>89.10</u> | **95.20** | 46.50 | <u>40.00</u> | <u>77.50</u> | 57.60 | <u>38.89</u> |
|  | Kuhn Poker | 61.38 | 87.80 | 94.50 | 48.40 | 33.30 | 72.50 | <u>59.30</u> | 33.84 |
|  | Mini Hanabi | 62.05 | 88.10 | 94.70 | 48.00 | **43.30** | 65.00 | 58.90 | 36.36 |
|  | **Generalist** | 62.79 | **89.90** | 94.60 | **52.00** | 33.30 | 75.00 | **59.90** | 34.85 |
| **MAD (Competitive)** | Qwen3-4B | 72.45 | 90.20 | 95.91 | 80.71 | 40.00 | 75.00 | **87.42** | 37.88 |
|  | SPIRAL | 73.41 | 91.60 | 95.45 | 81.89 | 40.00 | 77.50 | 87.01 | 40.40 |
|  | **MARSHAL** |  |  |  |  |  |  |  |  |
|  | Tic-Tac-Toe | <u>75.01</u> | 92.20 | 96.06 | 83.07 | 43.33 | **82.50** | 86.76 | 41.12 |
|  | Kuhn Poker | 74.54 | 91.60 | **96.21** | 82.68 | 40.00 | **82.50** | 87.39 | <u>41.41</u> |
|  | Mini Hanabi | 73.70 | 91.40 | 95.60 | 82.68 | 43.33 | 77.50 | 87.04 | 38.38 |
|  | **Generalist** | **75.96** | **92.80** | 95.60 | **83.86** | **46.67** | <u>80.00</u> | <u>87.36</u> | **45.45** |
| **AutoGen (Cooperative)** | Qwen3-4B | 79.14 | 93.40 | **94.69** | 85.04 | 56.67 | 87.50 | 89.21 | 47.47 |
|  | SPIRAL | 80.05 | 94.20 | 94.47 | 86.61 | 60.00 | 87.50 | **91.60** | 45.96 |
|  | **MARSHAL** |  |  |  |  |  |  |  |  |
|  | Tic-Tac-Toe | 80.15 | 94.40 | **94.69** | **87.01** | 60.00 | 90.00 | 89.53 | 45.45 |
|  | Kuhn Poker | 81.54 | **95.80** | 94.39 | <u>86.61</u> | <u>63.33</u> | <u>92.50</u> | 89.65 | <u>48.48</u> |
|  | Mini Hanabi | 81.54 | 94.40 | <u>94.54</u> | 86.22 | **66.67** | **95.00** | 88.98 | 44.95 |
|  | **Generalist** | **82.15** | <u>95.20</u> | <u>94.54</u> | <u>86.61</u> | **66.67** | <u>92.50</u> | 89.53 | **50.00** |