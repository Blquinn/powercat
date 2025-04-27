# PowerCat

PowerCat is an LLM based transaction categorization tool for [Tiller](https://tiller.com/) spreadsheets.

It runs entirely locally (your data doesn't leave your computer) using ollama models and implements a basic RAG pattern.

You will need a GPU with 8-10GiB of video ram to run it.

It indexes your historical transactions and uses those to inform its decision
on how to categorize your uncategorized transactions.

It is slow, currently it takes about 2 seconds per transaction, but no interaction is required.
