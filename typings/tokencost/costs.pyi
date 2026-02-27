from decimal import Decimal
from typing import Literal

TokenType = Literal["input", "output", "cached"]

def calculate_cost_by_tokens(
    num_tokens: int, model: str, token_type: TokenType
) -> Decimal:
    """
    Calculate the cost based on the number of tokens and the model.

    Args:
        num_tokens (int): The number of tokens.
        model (str): The model name.
        token_type (str): Type of token ('input' or 'output').

    Returns:
        Decimal: The calculated cost in USD.
    """
