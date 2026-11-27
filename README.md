# BOQ Rate Predictor

I am a quantity surveyor learning Python, and I built this tool to answer a question I kept asking at work: given what we paid last time, what should this BOQ item cost?

It learns from past priced items and predicts a rate for a new item, from the words in its description, its unit and its quantity. It is a small machine learning project, written with the Python standard library only.

## What it does

You give it a history file of priced BOQ items. It trains a ridge regression model on them, then predicts rates for new items. The model reads patterns like concrete grade, bar size and element type from the description, so a Grade 40 column costs more than a Grade 25 slab, just like in real pricing.

## Sample data

The file `data/history.csv` holds 122 priced items across concrete, reinforcement steel, formwork, blockwork, finishes and M&E. These are sample rates I made up to follow realistic Singapore construction patterns. They are not prices from a real project or a real client. Treat them as practice data.

## How to run it

You need Python 3.10 or newer. There is nothing to install.

Train the model and see how it does on items it has not seen:

```bash
python -m boq_rate_predictor train data/history.csv
```

Predict a rate for a new item:

```bash
python -m boq_rate_predictor predict --description "Grade 30 concrete in foundations" --unit m3 --quantity 120
```

Options for training: `--test-size` sets the share of items held back for testing (default 0.2), and `--seed` sets the random seed (default 42).

Run the tests:

```bash
python -m unittest discover -s tests -v
```

## How it works

The model is ridge regression, solved with the normal equations and Gaussian elimination. I wrote the maths myself, because this repo uses the standard library only, with no numpy, pandas or scikit-learn.

Features come from the item in three parts. Words in the description are hashed into 128 buckets, so "grade", "30" and "columns" each push the prediction in their own direction. The unit is hashed the same way. The quantity goes in as its log, since doubling a quantity does not double a rate. The model predicts the log of the rate, then converts it back to dollars.

## Honest scope

This is a learning project, not a pricing service. It works best on items that look like the ones in its history file, and it will be wrong on anything unusual. The sample data is small and made up, so the predictions are a starting point for a human to check, not a number to put in a tender. If you use your own history file with real rates, it will do better, but you should still check every rate before it goes anywhere near a client.

## Project structure

- `boq_rate_predictor/model.py` - feature hashing, ridge regression, training and prediction
- `boq_rate_predictor/metrics.py` - MAE and R-squared
- `boq_rate_predictor/cli.py` - the command line interface
- `data/history.csv` - sample priced BOQ items
- `tests/test_model.py` - unit and end-to-end tests

## Licence

MIT. See `LICENSE`.
