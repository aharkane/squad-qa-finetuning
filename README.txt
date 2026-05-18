
# Fine-tuning Question-Answering Models on SQuAD


This project fine-tunes and compares three transformer-based models for extractive Question Answering on the **Stanford Question Answering Dataset (SQuAD v1.1)**:

- **DistilBERT** (`distilbert-base-uncased`)
- **MobileBERT** (`google/mobilebert-uncased`)
- **ALBERT** (`albert-base-v2`)

An interactive web app deployed on Hugging Face Spaces lets users compare the three models side-by-side on their own context and questions.

## 🚀 Live Demo

**Try the app:** [Ask Licorn 🦄 on Hugging Face Spaces](https://huggingface.co/spaces/aharkane/squad-qa-app)

**Fine-tuned models on Hugging Face Hub:**
- [`aharkane/distilbert-squad`](https://huggingface.co/aharkane/distilbert-squad)
- [`aharkane/mobilebert-squad`](https://huggingface.co/aharkane/mobilebert-squad)
- [`aharkane/albert-squad`](https://huggingface.co/aharkane/albert-squad)

## 📊 Results

Evaluated on the official SQuAD v1.1 validation set (10 570 examples):

| Model | F1 | Exact Match | Time/question (ms) | Parameters (M) |
|---|---|---|---|---|
| DistilBERT | 81.68 | 72.39 | 11.34 | 66.4 |
| MobileBERT | 0.37 | 0.03 | 8.84 | 24.6 |
| ALBERT | **90.32** | **83.31** | 27.43 | **11.1** |

**Key finding:** ALBERT achieves the highest scores with the smallest model (11M parameters vs 66M for DistilBERT). MobileBERT failed to converge with standard hyperparameters, an interesting illustration that fine-tuning is not a universal recipe.

![Model Comparison](reports/comparison.png)

## 📁 Project Structure


DNF3ED04-Projet-SQuAD_FineTuning/
├── data/                    (vide, généré au runtime)
├── reports/
│   ├── evaluation_results.csv
│   └── comparison.png
├── docs/
├── notebooks/
│   ├── 00_environment_setup.ipynb
│   ├── 01_data_exploration.ipynb
│   ├── 02_finetuning.ipynb
│   ├── 03_evaluation.ipynb
│   └── 04_push_to_hub.ipynb
├── streamlit_app/
│   ├── app.py
│   ├── requirements.txt
│   └── README.md
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_processing.py
│   ├── training.py
│   ├── evaluation.py
│   └── utils.py
├── .gitignore
├── README.md
└── requirements.txt


## 🛠️ Setup

### 1. Clone the repo

```bash
git clone https://github.com/aharkane/squad-finetuning.git
cd squad-finetuning
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the notebooks (Google Colab recommended)

1. Open `notebooks/00_environment_setup.ipynb` and run it once
2. Explore the dataset with `notebooks/01_data_exploration.ipynb`
3. Fine-tune each model with `notebooks/02_finetuning.ipynb` (change `MODEL_TO_TRAIN`)
4. Push models to HF Hub with `notebooks/04_push_to_hub.ipynb`
5. Evaluate all models with `notebooks/03_evaluation.ipynb`

### 4. Run the Streamlit app locally

```bash
cd streamlit_app
pip install -r requirements.txt
streamlit run app.py
```

## 🧠 Methodology

### Fine-tuning
- **Dataset:** SQuAD v1.1 (87 599 training examples)
- **Tokenization:** sliding window with `max_length=384`, `stride=128`
- **Training:** 1 epoch, batch size 16, learning rate 2e-5, fp16, T4 GPU
- **Time:** ~17 min for DistilBERT on Colab T4

### Evaluation
- **Metrics:** F1, Exact Match (official SQuAD metric)
- **Post-processing:** top-20 best (start, end) combinations per window
- **Reference:** [Hugging Face QA tutorial](https://github.com/huggingface/notebooks/blob/main/examples/question_answering.ipynb)

## 🎓 Key Learnings

1. **Smaller can be better.** ALBERT (11M params) outperforms DistilBERT (66M) thanks to its parameter-sharing architecture.
2. **Fine-tuning is not universal.** MobileBERT failed to converge with standard hyperparameters, highlighting model-specific tuning needs.
3. **Sliding window matters.** A stride of 128 tokens ensures that answers near window boundaries are never split.
4. **DistilBERT and `token_type_ids`.** A known bug where DistilBERT's tokenizer produces `token_type_ids` that the model rejects. Fixed by overriding `tokenizer.model_input_names`.

## 📚 References

- [SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://arxiv.org/abs/1606.05250)
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- [DistilBERT, a distilled version of BERT](https://arxiv.org/abs/1910.01108)
- [ALBERT: A Lite BERT for Self-supervised Learning](https://arxiv.org/abs/1909.11942)
- [MobileBERT: a Compact Task-Agnostic BERT](https://arxiv.org/abs/2004.02984)

## 👤 Author

**Amine Harkane**  
M2 Informatique - Université Paris 8  
[Hugging Face profile](https://huggingface.co/aharkane)

## 📄 License

MIT


