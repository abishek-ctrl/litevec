"""Metadata query compiler and evaluator."""

from typing import Dict, Any, List
from vector_db.exceptions import InvalidQueryError

def evaluate_filter(metadata: Dict[str, Any], query: Dict[str, Any]) -> bool:
    """Evaluate a query filter against a metadata dictionary.

    Supports comparison operators ($eq, $ne, $gt, $gte, $lt, $lte, $in, $nin)
    and logical operators ($and, $or, $not).

    Args:
        metadata: The document metadata dictionary.
        query: The query specification dictionary.

    Returns:
        True if the metadata matches the query, False otherwise.

    Raises:
        InvalidQueryError: If the query format or operator is invalid.
    """
    if not query:
        return True

    for key, value in query.items():
        # Logical operators
        if key == "$and":
            if not isinstance(value, list):
                raise InvalidQueryError("$and operator must contain a list of filters")
            if not all(evaluate_filter(metadata, q) for q in value):
                return False
        elif key == "$or":
            if not isinstance(value, list):
                raise InvalidQueryError("$or operator must contain a list of filters")
            if not any(evaluate_filter(metadata, q) for q in value):
                return False
        elif key == "$not":
            if not isinstance(value, dict):
                raise InvalidQueryError("$not operator must contain a filter dict")
            if evaluate_filter(metadata, value):
                return False
        elif key.startswith("$"):
            raise InvalidQueryError(f"Unsupported logical operator: {key}")
        else:
            # Field comparison or operator
            if isinstance(value, dict) and any(k.startswith("$") for k in value.keys()):
                field_val = metadata.get(key)
                for op, op_val in value.items():
                    if op == "$eq":
                        if field_val != op_val:
                            return False
                    elif op == "$ne":
                        if field_val == op_val:
                            return False
                    elif op == "$gt":
                        try:
                            if field_val is None or not (field_val > op_val):
                                return False
                        except TypeError:
                            return False
                    elif op == "$gte":
                        try:
                            if field_val is None or not (field_val >= op_val):
                                return False
                        except TypeError:
                            return False
                    elif op == "$lt":
                        try:
                            if field_val is None or not (field_val < op_val):
                                return False
                        except TypeError:
                            return False
                    elif op == "$lte":
                        try:
                            if field_val is None or not (field_val <= op_val):
                                return False
                        except TypeError:
                            return False
                    elif op == "$in":
                        if not isinstance(op_val, (list, tuple, set)):
                            raise InvalidQueryError("$in operator value must be a list, tuple, or set")
                        if field_val not in op_val:
                            return False
                    elif op == "$nin":
                        if not isinstance(op_val, (list, tuple, set)):
                            raise InvalidQueryError("$nin operator value must be a list, tuple, or set")
                        if field_val in op_val:
                            return False
                    else:
                        raise InvalidQueryError(f"Unsupported comparison operator: {op}")
            else:
                # Implicit equality
                if metadata.get(key) != value:
                    return False
    return True
