## Intro

The original code comes from this [Volatility Vibes YouTube video](https://www.youtube.com/watch?v=oW6MHjzxHpU&t=797s).

Issue: Yahoo Finance has pretty strict API rate limits, so we need to find something else. Tradier has all the data we need to replicate the calculation.

## Prerequisites

- Open an account with [Tradier](https://tradier.com/).
- Retrieve your API Access Token from [here](https://dash.tradier.com/settings/api).

## Installation

Store your Tradier API access token as an environment variable:

`export TRADIER_API_KEY="your-real-key-here"`

## Launching the App

```
cd /path/to/script
pip install -r requirements.txt
python3 -m venv venv
source venv/bin/activate
```

Prior to running the app, enter the stock tickers you want to analyze in this snippet near the bottom of the file `tradier_impl.py`

```
#Enter one or more stock symbols here...
async def test():
       await compute_recommendation("NKE")
       await compute_recommendation("ANGO")
```

Then run the code with the command:

`python3 tradier_impl.py`

Note that there is no UI for this implementation. The output is summarized in the final three lines of the console:

```
RED.  Avg volume of 403131 is below the minimum of 1500000
GREEN. iv-to-rv of 2.317 exceeds the minimum of 1.25
GREEN.  Term structure slope of -0.00654 falls below the maximum of -0.00406
```

When done:

```
deactivate
rm -rf venv/
```
