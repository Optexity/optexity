from pydantic import BaseModel


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    tool_use_tokens: int = 0
    thoughts_tokens: int = 0
    total_tokens: int = 0
    calculated_total_tokens: int = 0

    input_cost: float = 0
    output_cost: float = 0
    tool_use_cost: float = 0
    thoughts_cost: float = 0
    total_cost: float = 0

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            tool_use_tokens=self.tool_use_tokens + other.tool_use_tokens,
            thoughts_tokens=self.thoughts_tokens + other.thoughts_tokens,
            calculated_total_tokens=self.calculated_total_tokens
            + other.calculated_total_tokens,
            input_cost=self.input_cost + other.input_cost,
            output_cost=self.output_cost + other.output_cost,
            tool_use_cost=self.tool_use_cost + other.tool_use_cost,
            thoughts_cost=self.thoughts_cost + other.thoughts_cost,
            total_cost=self.total_cost + other.total_cost,
        )

    def __sub__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            input_tokens=self.input_tokens - other.input_tokens,
            output_tokens=self.output_tokens - other.output_tokens,
            total_tokens=self.total_tokens - other.total_tokens,
            tool_use_tokens=self.tool_use_tokens - other.tool_use_tokens,
            thoughts_tokens=self.thoughts_tokens - other.thoughts_tokens,
            calculated_total_tokens=self.calculated_total_tokens
            - other.calculated_total_tokens,
            input_cost=self.input_cost - other.input_cost,
            output_cost=self.output_cost - other.output_cost,
            tool_use_cost=self.tool_use_cost - other.tool_use_cost,
            thoughts_cost=self.thoughts_cost - other.thoughts_cost,
            total_cost=self.total_cost - other.total_cost,
        )
