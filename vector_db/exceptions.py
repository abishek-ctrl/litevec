"""Custom exceptions for the LiteVec package."""

class LiteVecError(Exception):
    """Base exception class for all LiteVec errors."""
    pass

class DimensionMismatchError(LiteVecError):
    """Exception raised when vector dimensions do not match the index configuration."""
    pass

class DuplicateIDError(LiteVecError):
    """Exception raised when attempting to insert a vector with an existing ID."""
    pass

class IDNotFoundError(LiteVecError):
    """Exception raised when operations target an ID that does not exist."""
    pass

class InvalidQueryError(LiteVecError):
    """Exception raised when metadata query filters are malformed."""
    pass

class EmbeddingError(LiteVecError):
    """Exception raised on remote API connection or parsing failures."""
    pass
