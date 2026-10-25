# Tweet Sentiment Classification

Classifies tweets as positive, negative or neutral on the [SemEval 2017 Task 4](https://arxiv.org/abs/1912.00741) (Subtask A) data. It compares classic models on word-count and TF-IDF features with a bidirectional LSTM with attention on GloVe embeddings.

## Background

This was originally completed as coursework for a university NLP module. Since then, three bugs in the LSTM training have been fixed and the notebook has been re-run (see [Changes since the original](#changes-since-the-original)).

## Data

The SemEval 2017 Task 4A tweets, in [`semeval-tweets/`](semeval-tweets/) (one tab-separated `tweet_id`, `sentiment`, `text` row per tweet):

| Set | Tweets | Positive | Negative | Neutral |
|---|---|---|---|---|
| Train | 45,026 | 35% | 18% | 46% |
| Dev | 2,000 | 35% | 19% | 46% |
| Test 1 | 3,531 | 42% | 16% | 43% |
| Test 2 | 1,853 | 53% | 11% | 36% |
| Test 3 | 2,379 | 43% | 15% | 41% |

Negative is the smallest class in every set, so every model except Naive Bayes uses class weighting.

## Method

### Preprocessing

Each tweet is:
1. lowercased, with contractions expanded ("can't" → "cannot");
2. stripped of URLs, `@mentions` and the `#` symbol (the hashtag word is kept);
3. given text names for its emojis (😩 → "weary");
4. stripped of punctuation, with repeated letters shortened ("soooo" → "so");
5. stripped of stop words, except the negations *no*, *not* and *nor*;
6. lemmatised.

Placeholder tokens (`<URL>`, `<MENTION>`, `<CAPS>`, `<REPEAT>`) and SMOTE oversampling were also tried, but they didn't help and were dropped.

### Features (classic models)

- **V1**: unigram and bigram counts, plus counts of positive and negative words from the [Bing Liu opinion lexicon](https://www.nltk.org/api/nltk.corpus.reader.opinion_lexicon.html).
- **V2**: TF-IDF, plus the same lexicon counts.

### Models

- **Classic**: MaxEnt (logistic regression), LinearSVC, Multinomial Naive Bayes and an SGD classifier, each grid-searched on both feature sets. The search uses stratified 3-fold cross-validation and scores by macro-F1.
- **LSTM**: frozen 100-dimension GloVe embeddings (top 5,000 training words) feed a 2-layer bidirectional LSTM (256 hidden units). An attention layer weights each word's output, followed by dropout (0.5) and a linear classifier. The model has 2.6M trainable parameters. It is trained with class-weighted cross-entropy and Adam (learning rate 1e-4), with early stopping on dev loss (patience 3), keeping the best epoch's weights.

The two best classic models on dev, MaxEnt and LinearSVC, are carried forward, both on V1.

### Evaluation

The score is SemEval's official metric: macro-F1 over the **positive and negative** classes only. Neutral isn't scored directly, but mistakes involving it still lower the positive and negative scores.

## Results

| Model | Dev | Test 1 | Test 2 | Test 3 | Test avg |
|---|---|---|---|---|---|
| **MaxEnt (V1)** | **0.631** | 0.616 | 0.630 | **0.562** | **0.603** |
| LinearSVC (V1) | 0.623 | 0.617 | **0.632** | 0.553 | 0.601 |
| LSTM + attention | 0.614 | **0.626** | 0.599 | 0.547 | 0.591 |

- **MaxEnt has the best dev score and test average**, but the three models are within about 0.01 of each other on average.
- **The LSTM does no better than the classic models.** It confuses positive and neutral tweets most, despite the class weights ([confusion matrix](figures/LSTM_ConfMat.png)).
- **Every model drops on test 3**, by a similar amount, which points to a different mix of tweets rather than chance.
- **The top systems in the original competition scored higher** (see the [task paper](https://arxiv.org/abs/1912.00741)), likely through different preprocessing and larger neural models.

Full grid-search results on dev, from [`Model Stats.csv`](Model%20Stats.csv):

| Model | V1 (counts) | V2 (TF-IDF) |
|---|---|---|
| MaxEnt | **0.631** | 0.618 |
| LinearSVC | 0.623 | 0.614 |
| SGD | 0.621 | 0.610 |
| Multinomial NB | 0.565 | 0.612 |

Figures: [MaxEnt](figures/MaxEnt_ConfMat.png), [LinearSVC](figures/LSVC_ConfMat.png) and [LSTM](figures/LSTM_ConfMat.png) dev confusion matrices, and the [LSTM loss curve](figures/LSTM_ConvPlot.png).

## Changes since the original

- **The LSTM was validated on raw tweets.** A later cell reloaded the dev set without cleaning it, so the validation loss and early stopping ran on uncleaned text, unlike training.
- **The best epoch was never kept.** `state_dict().copy()` is a shallow copy, so the "saved" weights kept changing, and they were never loaded back. The model is now deep-copied and restored.
- **Training batches were the same every epoch.** The data is now shuffled each epoch, with a fixed random seed.
- **LinearSVC now uses V1**, which scored higher on dev than the V2 features it used before.
- **Smaller fixes**: the embeddings path now works on Mac and Linux, only the NLTK data that's needed is downloaded, and the full grid search is off by default.

The classic models' scores are unchanged. The LSTM's dev score is about the same (0.613 → 0.614), with small changes on the test sets.

## Layout

```
nlp-sentiment-analysis/
├── sentiment_analysis.ipynb       # preprocessing, features, models, evaluation
├── environment.yml                # conda environment
├── Model Stats.csv                # full grid-search results on dev
├── figures/                       # confusion matrices and LSTM loss curve
├── semeval-tweets/
│   ├── twitter-*.txt              # train, dev and test 1-3
│   ├── Read in.ipynb              # early exploration notebook
│   └── embeddings/                # not in git: GloVe vectors (see below)
└── Docs/
    ├── 1912.00741v1.pdf           # SemEval 2017 Task 4 paper
    └── semeval-tweets.tar.bz2     # original data archive
```

## Running

```
conda env create -f environment.yml
conda activate nlp-sentiment-analysis
```

1. **Get GloVe**: download `glove.6B.zip` from [Stanford NLP](https://nlp.stanford.edu/projects/glove/) and put `glove.6B.100d.txt` in `semeval-tweets/embeddings/`. It's ~350 MB, too large for git.
2. **Run** `sentiment_analysis.ipynb` from the repo root, with the `nlp-sentiment-analysis` environment as the kernel.

A full run takes about 6 minutes with a GPU (RTX 4070 SUPER). Set `full_model_search = True` to repeat the grid search over every model and feature set. That takes much longer, since the SGD search alone is 972 fits per feature set.

`environment.yml` installs the CUDA 12.8 build of PyTorch. For a CPU-only machine, remove the `--extra-index-url` line; the LSTM still trains, just more slowly.
