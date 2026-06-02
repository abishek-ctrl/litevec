"""Unit tests for the MongoDB-like metadata query compiler."""

import pytest
from vector_db.filter import evaluate_filter
from vector_db.exceptions import InvalidQueryError

def test_implicit_equality_matching():
    """Verify standard key-value matches work correctly (implicit $eq)."""
    meta = {"category": "sports", "views": 100}
    assert evaluate_filter(meta, {"category": "sports"}) is True
    assert evaluate_filter(meta, {"category": "news"}) is False
    assert evaluate_filter(meta, {"views": 100}) is True
    assert evaluate_filter(meta, {"views": 101}) is False

def test_comparison_operators():
    """Verify operators ($eq, $ne, $gt, $gte, $lt, $lte, $in, $nin)."""
    meta = {"views": 100, "status": "active"}

    # $eq / $ne
    assert evaluate_filter(meta, {"views": {"$eq": 100}}) is True
    assert evaluate_filter(meta, {"views": {"$ne": 100}}) is False
    assert evaluate_filter(meta, {"status": {"$ne": "archived"}}) is True

    # $gt / $gte / $lt / $lte
    assert evaluate_filter(meta, {"views": {"$gt": 50}}) is True
    assert evaluate_filter(meta, {"views": {"$gt": 100}}) is False
    assert evaluate_filter(meta, {"views": {"$gte": 100}}) is True
    assert evaluate_filter(meta, {"views": {"$lt": 150}}) is True
    assert evaluate_filter(meta, {"views": {"$lte": 100}}) is True

    # $in / $nin
    assert evaluate_filter(meta, {"status": {"$in": ["active", "pending"]}}) is True
    assert evaluate_filter(meta, {"status": {"$nin": ["archived", "deleted"]}}) is True
    assert evaluate_filter(meta, {"status": {"$in": ["archived"]}}) is False

def test_logical_operators():
    """Verify Logical operators ($and, $or, $not)."""
    meta = {"category": "tech", "views": 250}

    # $and
    query_and = {
        "$and": [
            {"category": "tech"},
            {"views": {"$gt": 200}}
        ]
    }
    assert evaluate_filter(meta, query_and) is True

    # $or
    query_or = {
        "$or": [
            {"category": "news"},
            {"views": {"$gt": 200}}
        ]
    }
    assert evaluate_filter(meta, query_or) is True

    # $not
    query_not = {
        "$not": {"category": "news"}
    }
    assert evaluate_filter(meta, query_not) is True

def test_invalid_query_format_raises_exception():
    """Verify InvalidQueryError is raised when malformed queries are submitted."""
    meta = {"category": "tech"}
    with pytest.raises(InvalidQueryError):
        evaluate_filter(meta, {"$and": "not a list"})

    with pytest.raises(InvalidQueryError):
        evaluate_filter(meta, {"category": {"$in": "not a list"}})

    with pytest.raises(InvalidQueryError):
        evaluate_filter(meta, {"$unknown": True})
