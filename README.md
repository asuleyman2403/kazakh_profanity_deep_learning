# Kazakh Profanity Detection in Product Reviews

DSSC5113 Bonus Project. Author: Assyl Suleiman.

The model reads one Kazakh review or comment and predicts:

- 0 = no Kazakh profanity
- 1 = Kazakh profanity present

It also gives the probability of class 1.

The model is a character-level CNN that I trained from scratch with PyTorch. The full walkthrough with outputs is in `Suleiman_Assyl_Bonus_Project.ipynb`.

## Results (held-out test set, used once)

- Test examples: 2,246 (251 profane)
- Threshold: 0.51 (chosen on validation)
- Precision: 0.952, recall: 0.948, F1: 0.950 (profanity class)
- Confusion matrix: TN 1983, FP 12, FN 13, TP 238

Files: `results/test_metrics.json`, `results/confusion_matrix.png`, `results/test_predictions.csv`, `results/train_history.csv`.

## Dataset

Source: KazSAnDRA (Kazakh Sentiment Analysis Dataset of Reviews and Attitudes), ISSAI, CC BY 4.0. Link: https://github.com/IS2AI/KazSAnDRA. It has about 168k real Kazakh reviews of apps, books, places and goods (app store, book store, maps, online marketplaces). The texts are short and informal, often mixed with Russian, and have many spelling mistakes, like real product comments. `src/prepare_data.py` downloads the polarity CSV files automatically (no login needed).

My word list started from the Kazakh Swear Words dataset (https://huggingface.co/datasets/Rustem-Kaimolla/kazakh-swear-words, MIT). I use it again at the end as an extra check.

KazSAnDRA has sentiment labels only, so I made new profanity labels:

1. Regex rules (`src/lexicon.py`) find obscene words and their spelling variants. A match gives label 1.
2. 1,050 unclear matches were checked by hand (`data/manual_labels.csv`). Examples: bare "ам", "көт", "боқ", "зб", masked words like "бл**", and words that only look like profanity, such as "педикюр", "епті", "сигнал" or the surname "Далбаев".
3. Everything else is label 0. I checked 120 random negatives by hand and added the few forms I had missed.

There are no synthetic examples.

### Labeling rules

- Label 1: obscene or vulgar words. These are sexual words and body parts (қотақ, ам, көт, сігу and their forms), mother insults (шешеңді ..., шшс, 667), жәлеп, қаншық, сайқал, and excrement words (боқ, говно).
- Mixed language: Russian mat inside a Kazakh comment (бля, нахуй, пиздец, заебал/зайбал, сука, далбаёб) is label 1. People mix the two languages a lot, and a moderator would remove these too.
- Label 0 for mild insults: мал, ешак, ақымақ, жынды, тупой, қоқыс. A negative review is not profane by itself.
- Talking about swearing without swearing is 0, for example "боқ сөз айтпаңдар".
- Masked words and changed spellings count when the word is clear: "қ***бас", "бл**", "к0тақ", "зб". The rules join letters split by * or ., map Latin look-alike letters to Cyrillic, and shorten repeated letters.
- Ambiguous cases ("ой шешең...", "нау") are 0. I mark 1 only when I am sure.

### Examples

- "Өте жақсы. Тез әрі уақытылы жеткізді, қатты ұнады" → 0
- "Мал ойын, тартылмайды" → 0 (rude, but no profanity)
- "Реклама көп, жабылып қала береді" → 0
- "Қотақбас ойын мал мал" → 1
- "Пиздец ойын екен" → 1 (Russian mat in a Kazakh comment)
- "Шшс чит тартпаңдар акк қа бан беред" → 1 (abbreviation of a mother insult)
- "жарнама жарнама қ****бас" → 1 (masked word)

### Size, splits and seed

- Raw reviews: 167,961. After removing exact duplicates: 166,049.
- Final dataset: 22,530 reviews, 2,534 profane (11.2%). I kept all profane reviews and sampled 20,000 clean ones. Half of those come from negative reviews and half from positive reviews, and all clean reviews with mild insults are included, so the model must learn that negative does not mean profane.
- Near-duplicates (same letters after removing punctuation, digits and repeated letters) are grouped. A group goes to one split only.
- Split 80/10/10, stratified by label, seed 42:
  - train: 18,038 (2,033 profane)
  - validation: 2,246 (250 profane)
  - test: 2,246 (251 profane)
- By domain: app store 18,690 (2,517 profane), book store 547 (10), maps 947 (7), marketplace 2,346 (0).

The processed splits are in `data/processed/` (columns: custom_id, text, label, label_source, sentiment, domain, group, split).

Privacy: KazSAnDRA has no reviewer names. I also mask links, e-mails, @usernames, phone numbers and surnames (-ов, -ова, -ұлы, -қызы ...) as [URL], [EMAIL], [USER], [PHONE], [NAME].

## Model

Character-level CNN (`src/model.py`), about 145k parameters.

- Why: profanity is written in many ways (қотақ, котак, қотағым, к0тақ, қ*тақ, зайбал, далбайоп). A word model would see most of these as unknown words. Every character is in the vocabulary, and CNN filters of width 2-5 find short pieces like "қот", "сіг" or "бля" anywhere in the text. The model is small, fast on CPU, and the saved file is under 1 MB.
- Text to input: lowercase, then letters repeated 3+ times are cut to 2. Each character becomes an id. The vocabulary is built on training data only, and characters seen fewer than 3 times become <unk>. Punctuation and emoji are kept as characters.
- Long comments: max 300 characters. Longer ones keep the first 150 and the last 150 characters.
- Layers: Embedding(64) → Conv1d with kernels 2, 3, 4, 5 (128 filters each) → ReLU → global max-pool → dropout 0.3 → Linear → 1 logit.
- Training: BCEWithLogits loss, Adam (lr 1e-3, weight decay 1e-5), batch 128, 15 epochs, seed 42, deterministic PyTorch on CPU. Batches group texts of similar length to reduce padding. The best epoch and the decision threshold are chosen by validation F1.
- All settings are saved in `models/config.json`.

Saved files needed for inference (in this repo, no download needed):

- `models/char_cnn.pt`: weights
- `models/vocab.json`: character vocabulary
- `models/config.json`: settings and threshold

## Setup

Python 3.10. Tested on macOS (Apple M1, CPU).

```
git clone <this repo>
cd kz_profanity
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Reproduce everything

```
python src/prepare_data.py   # download KazSAnDRA, label, dedup, split (seed 42)
python src/train.py          # train, choose epoch and threshold on validation (about 7 min on CPU)
python src/evaluate.py       # test metrics, confusion matrix, predictions -> results/
```

You can also run `Suleiman_Assyl_Bonus_Project.ipynb` from top to bottom. It runs the same code and shows all outputs.

## Predict on new text (no retraining)

```
python src/predict.py "Өте жақсы тауар, рахмет" "қотақбас ойын"
python src/predict.py      # interactive: type a comment, empty line to quit
```

In Python:

```python
import sys; sys.path.insert(0, "src")
from predict import ProfanityDetector
det = ProfanityDetector()
det.predict("Сапасы нашар, мал магазин")   # [{'text': ..., 'probability': 0.0, 'label': 0}]
```

## Error analysis (short)

The full discussion is in the notebook, section 3.

- Some false positives are actually my labeling mistakes, and the model is right: "сі#ейын", "За#бал", "нахууууу", "зай бал".
- Long comments: a bad word in the middle of a long text is cut off (only the first and last 150 characters are kept).
- Latin script ("Kotak", "Sgin ... dalbaeop suka") is missed, because the training data has very few such examples.
- Rare and short forms ("Щшс", "зб", "нах") are sometimes missed.
- Extra check on the 63 phrases of the Kazakh Swear Words dataset: accuracy 0.889.

A false positive hides an honest review (often a negative one). A false negative shows profanity on the product page. A practical setup is to auto-hide only very confident cases and send the middle zone to a human moderator.

Improvements: score long texts in 300-character windows, use the Latin-to-Cyrillic mapping in the model input as well, add augmented copies with random *, #, spaces and Latin letters, and have a second person label the test set.

## Limitations

- One person made the labels (rules plus manual review of unclear cases). The model partly copies these rules, and the test labels come from the same process.
- Not every text was checked by hand. Some label noise remains (about 2-3% of negatives in a random check before I fixed the patterns).
- Almost all profanity comes from app and game reviews. Marketplace reviews of goods have no profane examples, so results on goods reviews are less tested.
- First names can remain in some texts.

## Repository structure

```
src/lexicon.py        labeling rules (regex)
src/text_utils.py     masking of personal info, normalization, near-duplicate key
src/prepare_data.py   dataset building and splits
src/model.py          vocabulary and CharCNN
src/train.py          training, epoch and threshold selection on validation
src/evaluate.py       one-time test evaluation
src/predict.py        inference (function and command line)
data/manual_labels.csv  my manual decisions for unclear texts
data/processed/       train.csv, val.csv, test.csv
models/               trained model, vocabulary, config
results/              metrics, confusion matrix, predictions, training history
```

Data credit: Yeshpanov R., Varol H. A. KazSAnDRA: Kazakh Sentiment Analysis Dataset of Reviews and Attitudes. LREC-COLING 2024. CC BY 4.0.
