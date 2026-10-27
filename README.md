# Tweet Sentiment Classification

Classifies tweets as positive, negative or neutral on the [SemEval 2017 Task 4](https://arxiv.org/abs/1912.00741) (Subtask A) data. It compares classic models on word-count, TF-IDF and character n-gram features, a bidirectional LSTM with attention on GloVe embeddings, and a fine-tuned BERTweet transformer.

## Background

This was originally completed as coursework for a university NLP module. Since then, several bugs have been fixed (including one that under-scored every model), new features and a BERTweet model have been added, and the notebook has been re-run (see [Changes since the original](#changes-since-the-original)).

## Data

The SemEval 2017 Task 4A tweets, in [`semeval-tweets/`](semeval-tweets/) (one tab-separated `tweet_id`, `sentiment`, `text` row per tweet):

| Set | Tweets | Positive | Negative | Neutral |
|---|---|---|---|---|
| Train | 45,101 | 35% | 18% | 46% |
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
3. given text names for its emojis (😩 → "weary face");
4. stripped of punctuation, with repeated letters shortened ("soooo" → "so");
5. stripped of stop words, except the negations *no*, *not* and *nor*;
6. lemmatised.

Placeholder tokens (`<URL>`, `<MENTION>`, `<CAPS>`, `<REPEAT>`) and SMOTE oversampling were also tried, but they didn't help and were dropped.

### Features (classic models)

- **V1**: unigram and bigram counts, plus counts of positive and negative words from the [Bing Liu opinion lexicon](https://www.nltk.org/api/nltk.corpus.reader.opinion_lexicon.html).
- **V2**: TF-IDF, plus the same lexicon counts.
- **V3**: word TF-IDF (1–2 grams) and character TF-IDF (3–5 characters within words), plus the same lexicon counts. Character n-grams cope better with misspellings and slang.
- **V4**: V3 built on negation-marked text ("not like this movie" → "not like_NEG movie_NEG", up to the next punctuation mark), plus scores from two Twitter lexicons ([NRC Hashtag and NRC Emoticon](https://saifmohammad.com/WebPages/lexicons.html)). Negated words count with the opposite sentiment.

### Models

- **Classic**: MaxEnt (logistic regression), LinearSVC, Multinomial Naive Bayes and an SGD classifier, each grid-searched on V1 and V2, with MaxEnt and LinearSVC also on V3 and V4. The search uses stratified 3-fold cross-validation and scores by macro-F1.
- **LSTM**: frozen 100-dimension GloVe embeddings (top 5,000 training words) feed a 2-layer bidirectional LSTM (256 hidden units). An attention layer weights each word's output, followed by dropout (0.5) and a linear classifier. The model has 2.6M trainable parameters. It is trained with class-weighted cross-entropy and Adam (learning rate 1e-4), with early stopping on dev loss (patience 3), keeping the best epoch's weights.
- **BERTweet**: [BERTweet-base](https://huggingface.co/vinai/bertweet-base), a RoBERTa transformer pre-trained on 850M English tweets, with a new 3-class classification layer. It reads the raw tweets, since its tokenizer does its own light normalisation (@mentions → `@USER`, links → `HTTPURL`, emojis → text). All 135M weights are fine-tuned for 3 epochs with class-weighted cross-entropy and AdamW (learning rate 2e-5, warmed up over the first 10% of steps, then decayed linearly), keeping the epoch with the best dev score.

The two best classic models on dev are carried forward: MaxEnt on V1 and LinearSVC on V3.

### Evaluation

The score is SemEval's official metric: macro-F1 over the **positive and negative** classes only. Neutral isn't scored directly, but mistakes involving it still lower the positive and negative scores.

## Results

| Model | Dev | Test 1 | Test 2 | Test 3 | Test avg |
|---|---|---|---|---|---|
| MaxEnt (V1) | 0.672 | 0.653 | 0.657 | 0.601 | 0.637 |
| LinearSVC (V3) | 0.672 | 0.652 | 0.669 | 0.608 | 0.643 |
| LSTM + attention | 0.643 | 0.653 | 0.627 | 0.586 | 0.622 |
| **BERTweet** | **0.765** | **0.771** | **0.780** | **0.752** | **0.768** |

- **BERTweet is the best model by a wide margin**: 0.125 above the best classic model on the test average. Its test scores match its dev score, so choosing its epoch on dev didn't inflate the result. It rarely mixes up positive and negative; most of its errors involve neutral ([confusion matrix](figures/BERTweet_ConfMat.png)).
- **The classic models and the LSTM are within about 0.02 of each other.** LinearSVC on character n-grams is the best of them on the test average.
- **The LSTM does no better than the classic models.** It confuses positive and neutral tweets most, despite the class weights ([confusion matrix](figures/LSTM_ConfMat.png)).
- **The earlier models drop about 0.05 on test 3, but BERTweet drops only about 0.02.** Test 3 seems to use vocabulary the training set doesn't cover well, which BERTweet's pre-training does.
- **Compared with the original competitions**, the classic models and LSTM score below the top systems, while BERTweet scores above them. BERTweet benefits from years of later research and from pre-training on hundreds of millions of tweets. Those tweets date from 2012–2019, so some of the test tweets' text (but not their labels) may have been part of it.

Full grid-search results on dev, from [`Model Stats.csv`](Model%20Stats.csv). **These are from before the scoring fix** (see [Changes since the original](#changes-since-the-original)), so every score is about 0.04 too low. They'll be updated, with V4 added, after the next full grid search.

| Model | V1 (counts) | V2 (TF-IDF) | V3 (word + char TF-IDF) |
|---|---|---|---|
| MaxEnt | 0.628 | 0.617 | 0.627 |
| LinearSVC | 0.623 | 0.612 | **0.629** |
| SGD | 0.621 | 0.610 | – |
| Multinomial NB | 0.563 | 0.611 | – |

Character n-grams helped LinearSVC (+0.006 over V1) but not MaxEnt. With 2,000 dev tweets, a gap that small could be noise.

Figures: [MaxEnt](figures/MaxEnt_ConfMat.png), [LinearSVC](figures/LSVC_ConfMat.png), [LSTM](figures/LSTM_ConfMat.png) and [BERTweet](figures/BERTweet_ConfMat.png) dev confusion matrices, and the [LSTM loss curve](figures/LSTM_ConvPlot.png).

## Changes since the original

- **About 10% of predictions were scored as neutral.** Tweet IDs were read as numbers, so IDs starting with 0 lost that 0 and no longer matched the answer file, and the scorer counts an unmatched tweet as neutral. This lowered every model's scores by about 0.04. IDs are now read as text.
- **75 training tweets were merged into 16 rows.** Quote marks in tweets were treated as quoted fields, joining the lines in between. All 45,101 training tweets now load.
- **The LSTM was validated on raw tweets.** A later cell reloaded the dev set without cleaning it, so the validation loss and early stopping ran on uncleaned text, unlike training.
- **The best epoch was never kept.** `state_dict().copy()` is a shallow copy, so the "saved" weights kept changing, and they were never loaded back. The model is now deep-copied and restored.
- **Training batches were the same every epoch.** The data is now shuffled each epoch, with a fixed random seed. GPU training isn't fully deterministic, so LSTM scores can still vary by about 0.005 between runs.
- **The LSTM's attention included padding.** Shorter tweets spread some attention over padding tokens. Padding is now masked out before the softmax.
- **Emojis became single unknown words.** Removing punctuation joined each emoji's name into one word ("smilingfacewithsmilingeyes") that isn't in GloVe. Emojis now become separate words ("smiling face with smiling eyes"). This changes the cleaning for every model.
- **The full grid search replaced the main models.** Its cells reused the main models' names, so with the search turned on, the test evaluation used whichever model the search trained last. The search now keeps its own copies.
- **Character n-gram features (V3) were added**, and LinearSVC now uses them. It used V2 before.
- **Negation marking and Twitter lexicons (V4) were added**, compared in the full grid search.
- **BERTweet was added.**
- **Smaller fixes**: the embeddings path now works on Mac and Linux, only the NLTK data that's needed is downloaded, and the full grid search is off by default.

Under the old scoring, the emoji change moved MaxEnt's scores by less than 0.005, and the LSTM fixes didn't change its scores beyond normal run-to-run variation (dev 0.613 → 0.611). What holds the LSTM back lies elsewhere (see [Future improvements](#future-improvements)).

## Future improvements

- **BERTweet**: try [BERTweet-large](https://huggingface.co/vinai/bertweet-large), and average several fine-tuning runs with different seeds, since single runs vary. Avoid checkpoints already fine-tuned for tweet sentiment, as they were likely trained on these test sets.
- **LSTM**:
  - pack padded sequences so the backward LSTM doesn't read padding first;
  - use Twitter GloVe without the 5,000-word cap;
  - use lighter cleaning than the classic models.
- **Classic models**: build character n-grams from less-cleaned text (they currently see the cleaned, lemmatised text), and match lexicon words before lemmatising.
- **Ensembling**: average the models' predictions, as half of the competition's top ten did. With BERTweet this far ahead, it would need to be weighted towards BERTweet to help.

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
│   ├── embeddings/                # not in git: GloVe vectors (see below)
│   └── lexicons/                  # not in git: NRC sentiment lexicons (see below)
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
2. **Get the NRC lexicons** (optional, only needed for V4): download the NRC Hashtag Sentiment Lexicon (v0.1) and the NRC Emoticon Lexicon (v1.0) from [Saif Mohammad's lexicon page](https://saifmohammad.com/WebPages/lexicons.html) and unzip both into `semeval-tweets/lexicons/`. Their terms allow non-commercial research use but not redistribution, so they aren't in git. Without them, the notebook skips V4.
3. **Run** `sentiment_analysis.ipynb` from the repo root, with the `nlp-sentiment-analysis` environment as the kernel.

A full run takes about 10–15 minutes with a GPU (RTX 4070 SUPER), most of it fine-tuning BERTweet. The first run downloads BERTweet (~540 MB); set `run_bertweet = False` to skip it, which is worth doing without a GPU. Set `full_model_search = True` to repeat the grid search over every model and feature set. That takes much longer, since the SGD search alone is 972 fits per feature set.

`environment.yml` installs the CUDA 12.8 build of PyTorch. For a CPU-only machine, remove the `--extra-index-url` line; the LSTM still trains, just more slowly.
