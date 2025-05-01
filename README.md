# PowerCat

PowerCat is an LLM based transaction categorization tool for [Tiller](https://tiller.com/) spreadsheets.

It uses multiple algorithms, including existing autocat rules to categorize transactions
into one of your pre-defined categories.

## How it works

Powercat uses a few different methods to try to make categorizations, cheaply, quickly
and accurately.

First, it uses and prioritizes your existing autocat rules so that any "manual overrides"
you have will take effect. This also speeds up the process since autocat is _much_ simpler
and doesn't require the LLM.

A similar strategy is used to match based on previous transactions. It will find previous
transactions that look exactly the same as the one being categorized. It does some simple
string normalization to make matching more reliable.

Finally, if neither of the previous two methods were able to make a categorization,
then the LLM is used. First, it indexes all of your previous transactions into
a vector database. It does a similarity search to find any similar previous transactions.
It will then use those previous transactions as context to inform the LLM on
how to more accurately categorize future transactions.

## Disclaimer

This is obviously not meant to correctly categorize 100% of your transactions. It, as well
as all other tools like this make mistakes. LLMs make mistakes often. Which brings us to:

### Reviewing categorizations

Powercat will color the transactions it categorizes based on which algorithm was used
to make the transaction.

- Autocat categorizations are colored blue.
- Previous transaction matche categorizations are colored purple.
- LLM categorizations are colored yellow.

This makes it very simple to see how the decision was made and can inform how closely
it needs to be reviewed. For example, if you trust your autocat configuration, then
you can probably just ignore all the blue cells. Previous transaction matches can be
wrong, in instances like checks where no information is provided in the description
and you may have categorized them differently in the past. LLM categorizations have
been more accurate than I expected, but I would still assume that a decent chunk
of them will be wrong and should be reviewed.

Each of these algorithm gets better with time as more transactions are categorized.

### Performance

It runs entirely locally (your data doesn't leave your computer) using ollama models and implements a basic RAG pattern.

You will need a GPU with 8-10GiB of video ram to run it.

It indexes your historical transactions and uses those to inform its decision
on how to categorize your uncategorized transactions.

It is slow, currently it takes about 2 seconds per transaction (on my computer), but
no interaction is required.
