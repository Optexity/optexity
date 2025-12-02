from typing import Literal, Optional

from pydantic import BaseModel, field_validator, model_validator

from optexity.schema.actions.extraction_action import LLMExtraction


class LLMAssertion(LLMExtraction):
    source: list[Literal["axtree", "screenshot"]] = ["screenshot"]
    extraction_format: dict = {"assertion_result": "bool", "assertion_reason": "str"}

    @model_validator(mode="after")
    def validate_output_var_in_format(self):
        instruction = """You are a helpful assistant that verifies if the condition is met. 
        Use the info supplied below to verify the condition.
        The assertion_reason should be a short explanation of why the condition was met or not met.
        The assertion_result should be True if the condition is met, False otherwise.
        """
        self.extraction_instructions = instruction + "\n" + self.extraction_instructions
        if "screenshot" not in self.source:
            self.source.append("screenshot")

        return self


class NetworkCallAssertion(BaseModel):
    url_pattern: Optional[str] = None
    header_filter: Optional[dict[str, str]] = None


class PythonScriptAssertion(BaseModel):
    script: str
    ## TODO: add output to memory variables

    @field_validator("script")
    @classmethod
    def validate_script(cls, v: str):
        if not v.strip():
            raise ValueError("Script cannot be empty")
        return v


class AssertionAction(BaseModel):
    network_call: Optional[NetworkCallAssertion] = None
    llm: Optional[LLMAssertion] = None
    python_script: Optional[PythonScriptAssertion] = None

    @model_validator(mode="after")
    def validate_one_assertion(cls, model: "AssertionAction"):
        """Ensure exactly one of the extraction types is set and matches the type."""
        provided = {
            "llm": model.llm,
            "network_call": model.network_call,
            "python_script": model.python_script,
        }
        non_null = [k for k, v in provided.items() if v is not None]

        if len(non_null) != 1:
            raise ValueError(
                "Exactly one of llm, networkcall, or python must be provided"
            )

        return model

    def replace(self, pattern: str, replacement: str):
        if self.network_call:
            pass
        if self.llm:
            self.llm.replace(pattern, replacement)
        if self.python_script:
            pass
        return self
