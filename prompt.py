import csv
import json
import time
import re
import logging

import tiktoken

import ollama
from ollama import chat
from ollama import ChatResponse

import chromadb


logging.basicConfig(level=logging.INFO)

logging.getLogger('httpx').setLevel(logging.WARN)

log = logging.getLogger(__name__)


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


embedding_model = 'nomic-embed-text'
# model = 'llama3.2'
# model = 'deepseek-r1'
model = 'mistral'
transaction_file = "./data/transactions-all.csv"
accounts_file = "./data/accounts.csv"
categories_file = "./data/categories.csv"


def read_accounts():
    with open(accounts_file, 'r') as f:
        r = csv.DictReader(f)
        return [
            {
                'Account': row['Account'],
                'Account #': row['Account #'],
                'Class': row['Class'],
                'Group': row['Group'],
            }
            for row in r
        ]


def read_categories():
    with open(categories_file, 'r') as cats_f:
        r = csv.DictReader(cats_f)
        return [row['Category'] for row in r]


def read_transactions():
    with open(transaction_file, 'r') as trans_f:

        r = csv.DictReader(trans_f)

        return [
            {
                k: v
                for k, v in row.items()
                if k and v
            }
            for row in r
        ]


def get_transaction_fields():
    with open(transaction_file, 'r') as trans_f:
        first = trans_f.readline()
        return first.strip().split(',')


cats = read_categories()
trans = read_transactions()
accts = read_accounts()


def get_example_trans():
    categorized = [
        t for t in trans
        if t.get('Category')
    ]

    examples = []
    seen_descriptions = set()
    for t in categorized:
        if t['Description'] in seen_descriptions:
            continue

        seen_descriptions.add(t['Description'])

        examples.append(t)

    return examples

def get_example_trans_docs():
    return [
        f"Transaction with description '{t['Description']}', Account '{t['Account']}' from institution '{t['Institution']}' " +
            f"was categorized as '{t['Category']}'"
        for t in example_trans
    ]

example_trans = get_example_trans()


def build_prompt(t, cat_response) -> str:
    accts_text = "\n".join(
        (f"Account Name: {a["Account"]}, Account Type: {a["Group"]}, Account Class: {a["Class"]}" for a in accts)
    )
    t_text = "\n".join((f"{k}: {v}" for k, v in t.items()))

    example_t_text = ""
    if cat_response:
        example_t_text = f"""Here are some examples of previous categorization(s):

```
{cat_response}
```
"""

    c_text = "\n".join(cats)
    
    return f"""Please categorize the following financial transaction.
{example_t_text}

For context, this transaction may be from one of the following accounts:

```
{accts_text}
```

The transaction will fall into one of the following categories:

```
{c_text}
```

Finally, here is the transaction information:

```
{t_text}
```

I will ask you to output a confidence level as a percentage. Don't be over-confident. 
If there is ambiguity in the data provided, you should provide a lower confidence level.

Output the response in the following json format, only output the json and nothing else:

```json
{{
    "transactionId": "(string) The transaction id from the supplied transaction",
    "accountId": "(string) The account id from the supplied transaction",
    "category": "(string) The category from the supplied list of categories",
    "confidenceLevel": "(number) The confidence level (in percent) in your determination of the category",
    "explanation": "(string) A brief explanation as to how you made the categorization determination"
}}
```
"""

tries = 5


def validate_response_data(dat, transaction):
    if dat['category'] not in cats:
        raise Exception(f"Got unexpected category {dat['category']}")

    lvl = dat['confidenceLevel']
    if lvl <= 0 or lvl > 100:
        raise Exception(f"Got unexpected confidence level {lvl} {lvl}")

    if dat['transactionId'] != transaction['Transaction ID']:
        raise Exception(f"Got unexpected transaction id")

    if dat['accountId'] != transaction['Account ID']:
        raise Exception(f"Got unexpected account id")
    

def massage_data(dat):

    lvl = dat['confidenceLevel']
    new_lvl = None
    if isinstance(lvl, int) and lvl in (1, 100):
        new_lvl = lvl
    elif isinstance(lvl, float) and lvl > 0 and lvl <= 1:
        new_lvl = int(lvl * 100)
    else:
        # Prob a string
        new_lvl = int(lvl)

    log.debug(f"Converted confidence from {type(lvl)} {lvl} -> {type(new_lvl)} {new_lvl}")

    return {
        **dat,
        'confidenceLevel': new_lvl,
    }
    

rex = re.compile(f'```(json)?(.+)```', re.DOTALL | re.IGNORECASE | re.MULTILINE)

def extract_data(response, transaction):
    try:
        if model == 'llama3.2':
            res = json.loads(response)
        else:
            match = rex.search(response)
            if not match:
                log.info(f"Failed to find json in response: {response}")
                return None

            res = json.loads(match.groups()[1])

        log.info(f"Parsed json {res}")
            
        res = massage_data(res)

        validate_response_data(res, transaction)

        return res
    except Exception as e:
        log.info(f"Got an exception while extracting data {e}")
        return None



confidence_cuttoff = 70

# Make sure we're over the cutoff on half of the attempts
total_confidence_cuttoff = confidence_cuttoff * (tries // 2)


def choose_category(datas):
    cat_sums = {}
    for d in datas:
        cat = d['category']
        confidence = d['confidenceLevel']
        if confidence < confidence_cuttoff:
            continue

        if cat in cat_sums:
            cat_sums[cat] += confidence
        else:
            cat_sums[cat] = confidence

    if not cat_sums:
        return None

    sums_sorted = sorted(cat_sums.items(), key=lambda x: x[1], reverse=True)

    confidence_sum = sums_sorted[0][1]

    if confidence_sum < total_confidence_cuttoff:
        return None

    return sums_sorted[0][0]


def get_multiple_tries(prompt, transaction):

    datas = []
    for _ in range(tries):
        response: ChatResponse = chat(model=model, messages=[
            {
                'role': 'user',
                'content': prompt,
            },
        ], keep_alive='10m')

        response_content = response['message']['content']

        print(response_content)

        data = extract_data(response_content, transaction)
        if data:
            datas.append(data)

    if not datas:
        log.error(f"Failed to get data after {tries} attempts.")

    return datas



def generate_categorization_query_prompt(t):
    return f"Which category was the transaction with the description '{t['Description']}', " + \
        f" account '{t['Account']}' and institution 'f{t["Institution"]}' " + \
        f"categorized as?"


client = chromadb.Client()
collection = client.create_collection(name="example_transactions")

# store each document in a vector embedding database
for i, d in enumerate(get_example_trans_docs()):
    response = ollama.embed(model=embedding_model, input=d)
    embeddings = response["embeddings"]
    collection.add(
        ids=[str(i)],
        embeddings=embeddings,
        documents=[d]
    )



with open('./out/transactions-out.csv', 'w', newline='', encoding='utf-8') as out_f:
    fields = get_transaction_fields()
    w = csv.DictWriter(out_f, fieldnames=fields)
    w.writeheader()

    start = time.time()
    for t in trans[:200]:
        # Already categorized
        if t.get('Category'):
            w.writerow(t)
            out_f.flush()
            log.info("Line is already categorized, skipping.")
            continue

        log.info("Categorizing line")

        # generate an embedding for the input and retrieve the most relevant doc
        cat_query_prompt = generate_categorization_query_prompt(t)

        log.info(f"Prompt for categorization query: {cat_query_prompt}")

        response = ollama.embed(
            model=embedding_model,
            input=cat_query_prompt
        )

        results = collection.query(
            query_embeddings=response["embeddings"],
            n_results=1
        )

        cat_response = results['documents'][0][0]

        log.info(f"Categorization query response: {cat_response}")

        prompt = build_prompt(t, cat_response)

        log.debug(f"Full prompt: {prompt}")

        num_tokens = num_tokens_from_string(prompt, "o200k_base")

        log.info(f"This prompt contains {num_tokens} tokens.")

        datas = get_multiple_tries(prompt, t)
        cat = choose_category(datas)

        log.info(f"Categorized {cat} from {t}")

        out_dict = {
            **t,
            'Category': cat,
        }

        w.writerow(out_dict)
        out_f.flush()

    print(f"Processed {len(trans)} in {time.time() - start} seconds")
