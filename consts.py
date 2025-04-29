from datetime import datetime
from typing import TypedDict


AutoCatRule = TypedDict("AutoCatRule", {
    "Category": str,
    "Description Contains": str,
    "Account Contains": str,
    "Institution Contains": str,
    "Amount Min": float,
    "Amount Max": float,
    'Description Contains Slug': str,
    'Account Contains Slug': str,
    'Institution Contains Slug': str,
})

Transaction = TypedDict(
    "Transaction",
    {
        "Date": datetime,
        "Description": str,
        "Category": str,
        "Amount": str,
        "Account": str,
        "Account #": str,
        "Institution": str,
        "Month": datetime,
        "Week": datetime,
        "Transaction ID": str,
        "Account ID": str,
        "Check Number": int,
        "Full Description": str,
        "Date Added": datetime,
    },
)

Account = TypedDict(
    "Account",
    {
        "Account": str,
        "Account Number": str,
        "Class Override": str,
        "Group": str,
    },
)


transaction_cols = [
    "Date",
    "Description",
    "Category",
    "Amount",
    "Account",
    "Account #",
    "Institution",
    "Month",
    "Week",
    "Transaction ID",
    "Account ID",
    "Check Number",
    "Full Description",
    "Date Added",
]
