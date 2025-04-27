import csv
import logging
import time
from typing import List, Optional, TypedDict

from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

from langgraph.graph import START, StateGraph


logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARN)
log = logging.getLogger(__name__)


embedding_model = "nomic-embed-text"
# model = 'llama3.2'
# model = 'deepseek-r1'
model = "mistral"
transaction_file = "./data/transactions-all.csv"
accounts_file = "./data/accounts.csv"
categories_file = "./data/categories.csv"
output_file = "./out/transactions-out.csv"

# Minimum reported confidence to accept a categorization
confidence_cuttoff = 80
# High level of confidence, which will skip retries if reported
high_confidence_level = 97
# Maximum number of retries to categorize
max_attempts = 5


llm = ChatOllama(model=model, keep_alive='10m')


def read_categories() -> List[str]:
    with open(categories_file, "r") as cats_f:
        r = csv.DictReader(cats_f)
        return [row["Category"] for row in r]


cats = read_categories()


class Transaction(TypedDict):
    date: str
    description: str
    category: str
    amount: str
    account: str
    account_number: str
    institution: str
    month: str
    week: str
    transaction_id: str
    account_id: str
    check_number: str
    full_description: str
    date_added: str


class Account(TypedDict):
    account: str
    account_number: str
    account_class: str
    group: str


class Categorization(BaseModel):
    transaction_id: str = Field(
        description="The transaction id from the supplied transaction"
    )
    account_id: str = Field(description="The account id from the supplied transaction")
    description: str = Field(
        description="The description from the supplied transaction"
    )
    category: str = Field(
        description="The category of the transaction, from the supplied list of categories",
        enum=cats,
    )
    confidence_level: int = Field(
        description="The confidence level (in percent between 0 and 100) in your determination of the category",
        gt=0,
        le=100,
    )
    explanation: str = Field(
        description="""A brief explanation as to how you made the categorization determination.
 Also explain how you arrived at your chosen confidence level."""
    )


class State(TypedDict):
    transaction: Transaction
    previous_transactions: List[Document]
    categorization: Categorization


def read_accounts() -> List[Account]:
    with open(accounts_file, "r") as f:
        r = csv.DictReader(f)
        return [
            {
                "account": row["Account"],
                "account_number": row["Account #"],
                "account_class": row["Class"],
                "group": row["Group"],
            }
            for row in r
            if row["Account"]
        ]


def read_transactions() -> List[Transaction]:
    with open(transaction_file, "r") as f:

        r = csv.DictReader(f)

        return [
            {
                "date": row["Date"],
                "description": row["Description"],
                "category": row["Category"],
                "amount": row["Amount"],
                "account": row["Account"],
                "account_number": row["Account #"],
                "institution": row["Institution"],
                "month": row["Month"],
                "week": row["Week"],
                "transaction_id": row["Transaction ID"],
                "account_id": row["Account ID"],
                "check_number": row["Check Number"],
                "full_description": row["Full Description"],
                "date_added": row["Date Added"],
            }
            for row in r
        ]


def transaction_to_row(t: Transaction) -> dict:
    return {
        "Date": t["date"],
        "Description": t["description"],
        "Category": t["category"],
        "Amount": t["amount"],
        "Account": t["account"],
        "Account #": t["account_number"],
        "Institution": t["institution"],
        "Month": t["month"],
        "Week": t["week"],
        "Transaction ID": t["transaction_id"],
        "Account ID": t["account_id"],
        "Check Number": t["check_number"],
        "Full Description": t["full_description"],
        "Date Added": t["date_added"],
    }


def get_transaction_fields():
    with open(transaction_file, "r") as f:
        first = f.readline()
        return first.strip().split(",")


all_transactions = read_transactions()
accounts = read_accounts()


def get_example_transactions() -> List[Transaction]:
    categorized = [t for t in all_transactions if t.get("category")]

    examples = []
    seen_descriptions = set()
    for t in categorized:
        if t["description"] in seen_descriptions:
            continue

        seen_descriptions.add(t["description"])

        examples.append(t)

    return examples


def format_transaction(t: Transaction) -> str:
    return "\n".join((f"{k}: '{v}'" for k, v in t.items() if k and v))


def get_multiple_tries(tx: Transaction) -> Optional[Categorization]:
    max_cat: Categorization = None
    for i in range(max_attempts):
        try:
            graph = (
                StateGraph(State)
                .add_sequence([retrieve_previous_txs, categorize])
                .add_edge(START, "retrieve_previous_txs")
                .compile()
            )

            response: State = graph.invoke({"transaction": tx})

            cat = massage_categorization(response["categorization"])

            if cat.confidence_level >= high_confidence_level:
                log.info(f"Categorized with high confidence after {i+1} attempts")
                return cat
            elif cat.confidence_level >= confidence_cuttoff:
                if not max_cat:
                    max_cat = cat
                elif cat.confidence_level > max_cat.confidence_level:
                    max_cat = cat
        except Exception as e:
            log.info(f"Graph failed: {e}")
            continue

    if not max_cat:
        log.error(f"Failed to get data after {max_attempts} attempts.")

    return max_cat


start_embedding = time.time()

log.info(f"Begin indexing previous transactions")

embeddings = OllamaEmbeddings(model=embedding_model, keep_alive=10 * 60)

vector_store = InMemoryVectorStore(embeddings)

example_tx_docs = [
    Document(
        id=t["transaction_id"],
        page_content=format_transaction(t),
        metadata={"source": "transaction_file"},
    )
    for t in get_example_transactions()
]

docs = vector_store.add_documents(
    documents=example_tx_docs,
)

log.info(f"Completed indexing transactions in {round(time.time() - start_embedding, 2)}s")


def retrieve_previous_txs(state: State):
    # TODO I don't think embedding models are capable of returning nothing
    # They always provide some context
    prev_tx_tmpl = ChatPromptTemplate.from_template(
        """Which category was the transaction with the description '{description}',
account '{account}', and institution '{institution}' categorized as?
Say there were no previous categorizations if there were none."""
    )

    similarity_prompt = prev_tx_tmpl.invoke(
        {
            "description": state["transaction"]["description"],
            "account": state["transaction"]["account"],
            "institution": state["transaction"]["institution"],
        }
    )

    prompt = similarity_prompt.to_messages()[0].content

    log.debug(f"Retrieve previous categorizations prompt {prompt}")

    context_res = vector_store.similarity_search(prompt)

    log.debug(f"Retreived {len(context_res)} similar transactions for context")

    return {"previous_transactions": context_res}


def massage_categorization(c: Categorization):
    if c.confidence_level == 1:
        c.confidence_level = 100

    if c.confidence_level < 1 and c.confidence_level > 0:
        c.confidence_level = int(c.confidence_level * 100)

    return c


def categorize(state: State):
    template = ChatPromptTemplate.from_template(
        """Categorize the following transaction.

Here's the list of accounts that the transaction may have come from for context:

```
{accounts}
```

Here is a possibly related previous categorizations, if this categorization matches the
transaction that is provided, you should prefer this previous categorization over 
inferring them from the transaction's fields:

```
{previous_transactions}
```

And here is the transaction that needs to be categorized:

```
{transaction_text}
```
"""
    )

    accts_text = "\n\n".join(
        (
            f"Account Name: '{a["account"]}', Account Type: '{a["group"]}', Account Class: '{a["account_class"]}'"
            for a in accounts
        )
    )

    transaction_text = format_transaction(state["transaction"])
    previous_tx_text = "\n\n".join(
        (", ".join(c.page_content.split("\n")) for c in state["previous_transactions"])
    )

    structured_llm = llm.with_structured_output(Categorization)

    prompt = template.invoke(
        {
            "accounts": accts_text,
            "previous_transactions": previous_tx_text,
            "transaction_text": transaction_text,
        }
    )

    log.debug(f"The full prompt: {prompt.to_messages()[0].content}")

    response = structured_llm.invoke(prompt)

    return {"categorization": response}


with open(output_file, "w", newline="", encoding="utf-8") as out_f:
    fields = get_transaction_fields()
    w = csv.DictWriter(out_f, fieldnames=fields)
    w.writeheader()

    start = time.time()
    for tx in all_transactions[:100]:
        # Already categorized
        if tx["category"].strip():
            w.writerow(transaction_to_row(tx))
            out_f.flush()
            log.info("Line is already categorized, skipping.")
            continue

        line_start = time.time()
        log.info("Categorizing line")

        cat = get_multiple_tries(tx)
        category = tx["category"]

        if cat:
            line_end = time.time()
            log.info(
                f"Categorized ({round(line_end-line_start, 3)} s) {tx["description"]} -- {cat.category}, {cat.confidence_level} -- {cat.explanation}"
            )
            category = cat.category
        else:
            log.info(f"Failed to categorize transaction {tx}")

        out_dict = transaction_to_row({**tx, "category": category})

        w.writerow(out_dict)
        out_f.flush()

    log.info(f"Processed {len(all_transactions)} in {time.time() - start} seconds")
