from datetime import date, datetime
import logging
import re
from typing import Optional

from openpyxl.worksheet.worksheet import Worksheet
from slugify import slugify

from consts import Transaction, transaction_cols


log = logging.getLogger(__name__)


def excel_fmt_date(d: datetime) -> str:
    return f"{d.month}/{d.day}/{d.year}"


class AutoCat:
    """
    The autocat sheet will contain a "Category" column and a
    column for each condition that is used in any rule.
    """
    def __init__(self, sheet: Worksheet):

        # Ops take the value from the AutoCat col on the left and the value from
        # the transaction on the right.
        _text_ops = {
            "Equals": lambda a, b: a == b,
            "Contains" : lambda a, b: a in b,
            "Starts With": lambda a, b: a.startswith(b),
            "Ends With": lambda a, b: a.endswith(b),
            "Regex": lambda a, b: bool(a.match(b)),
        }

        def _polarity_eq(a, b):
            if a not in ("positive", "negative"):
                return False

            if a == "positive" and b > 0:
                return True

            if a == "negative" and b < 0:
                return True
            
            return False

        _number_ops = {
            "Equals": lambda a, b: a == b,
            "Min": lambda a, b: a <= b,
            "Max": lambda a, b: a >= b,
            "Polarity": _polarity_eq,
        }

        _ops = {}
        # Maps the Operation e.g. 'Amount Min' to the transaction field e.g. 'Amount'
        _ops_tx_val_map = {}

        for name, op in _number_ops.items():
            col_name = f"Amount {name}"
            _ops[col_name] = op
            _ops_tx_val_map[col_name] = "Amount"

        for col in transaction_cols:
            if col in ("Category", "Amount"):
                continue

            for name, op in _text_ops.items():
                col_name = f"{col} {name}"
                _ops[col_name] = op
                _ops_tx_val_map[col_name] = col

        self._rules = []

        for idx, row in enumerate(sheet.rows):
            if idx == 0:
                continue

            category = None
            row_ops = []

            for cell in row:
                if cell.value is None:
                    continue

                header = sheet.cell(1, cell.column).value
                if header == "Category":
                    category = cell.value
                    continue
                
                if header not in _ops:
                    log.warning(f"AutoCat header {header} is not a valid operation.")
                    continue

                tx_field = _ops_tx_val_map[header]
                op_name = header.lstrip(tx_field).strip()

                value = None
                if header.endswith("Regex"):
                    value = re.compile(cell.value, re.IGNORECASE)
                elif isinstance(cell.value, str):
                    # Normalize all strings
                    value = slugify(cell.value)
                else:
                    value = cell.value

                row_ops.append({
                    "field": tx_field,
                    "op": _ops[header],
                    "op_name": op_name,
                    "value": value,
                    "orig_value": cell.value,
                })

            if row_ops:
                self._rules.append({
                    "category": category,
                    "ops": row_ops,
                })

    # Optionally returns an autocat categorization
    def try_categorize(self, tx: Transaction) -> Optional[str]:
        for rule in self._rules:
            matches = True
            for op in rule["ops"]:
                field = tx[op["field"]]

                # Don't normalize regex text
                if op["op_name"] == "Regex":
                    pass
                # Normalize text
                elif isinstance(field, str):
                    field = slugify(field)
                elif isinstance(field, datetime) or isinstance(field, date):
                    field = excel_fmt_date(field)

                if not op["op"](op["value"], field):
                    matches = False
                    break

            if matches:
                return rule["category"]

        return None
