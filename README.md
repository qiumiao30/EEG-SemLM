# EEG Seizure Prediction Semantic Prompt Framework

## Model Architecture
![Model Architecture](./assets/model_architecture.png)

## Abstract
Accurate seizure prediction from electroencephalography (EEG) is critical for epilepsy management. However, due to the high dimensionality, non-stationarity, and complex spatiotemporal dependencies of EEG signals, existing approaches primarily emphasize temporal dynamics or spatial patterns, while overlooking higher-level semantic information associated with coordinated brain region activity. We propose a semantic prompt framework guided EEG representation learning for seizure prediction. First, we introduce structured semantic prompts by aggregating EEG within anatomically defined brain regions, enabling noise suppression and encoding region-level neurophysiological priors. Then, we design a temporal-semantic dual-stream architecture, where the temporal stream captures long-range dependencies using a convolution–Transformer hybrid model, and the semantic stream employs a pre-trained large language model (LLM) with a dedicated prompt encoder to generate region-aware contextual representations. To unify both modalities, a contrastive mutual information alignment module is introduced to integrate temporal dynamics and semantic priors. Finally, the fused representation is decoded by a pre-layer normalization linear decoder for seizure prediction. Experimental results demonstrate that the proposed method outperforms state-of-the-art approaches, achieving improvements of 2.6% in Precision and 1.3% in Recall, while exhibiting strong robustness in few-shot and zero-shot settings. This work shows that the proposed semantic prompt framework provides an effective mechanism for seizure prediction, improving both predictive performance and robustness for seizure prediction, with promising potential for effective EEG-based monitoring.

## Training Command
```bash
python train.py --lookback 200
```

## Quick Start
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run default training:
```bash
python train.py
```

3. Train with custom parameters:
```bash
python train.py --lookback 300 --batch_size 64 --epochs 50
```
