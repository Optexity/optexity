from typing import List, Literal, Optional

from pydantic import BaseModel, field_validator, model_validator

from optexity.utils.utils import build_model


class LLMExtraction(BaseModel):
    source: list[Literal["axtree", "screenshot"]] = ["axtree"]
    extraction_format: dict
    extraction_instructions: str
    output_variable_names: list[str] | None = None

    def build_model(self):
        return build_model(self.extraction_format)

    @field_validator("extraction_format")
    def validate_extraction_format(cls, v):
        if isinstance(v, dict):
            try:
                build_model(v)
            except Exception as e:
                raise ValueError(f"Invalid extraction_format dict: {e}")
            return v
        raise ValueError("extraction_format must be either a string or a dict")

    @model_validator(mode="after")
    def validate_output_var_in_format(self):

        if self.output_variable_names is not None:
            for key in self.output_variable_names:
                if key not in self.extraction_format:
                    raise ValueError(
                        f"Output variable {key} not found in extraction_format"
                    )
                if eval(self.extraction_format[key]) not in [str, list[str], List[str]]:
                    raise ValueError(
                        f"Output variable {key} must be a string or a list of strings"
                    )

        return self

    def replace(self, pattern: str, replacement: str):
        return self


class NetworkCallExtraction(BaseModel):
    url_pattern: Optional[str] = None
    header_filter: Optional[dict[str, str]] = None

    def replace(self, pattern: str, replacement: str):
        return self


class PythonScriptExtraction(BaseModel):
    script: str
    ## TODO: add output to memory variables

    @field_validator("script")
    @classmethod
    def validate_script(cls, v: str):
        if not v.strip():
            raise ValueError("Script cannot be empty")
        return v

    def replace(self, pattern: str, replacement: str):
        self.script = self.script.replace(pattern, replacement)
        return self


class ExtractionAction(BaseModel):
    network_call: Optional[NetworkCallExtraction] = None
    llm: Optional[LLMExtraction] = None
    python_script: Optional[PythonScriptExtraction] = None

    @model_validator(mode="after")
    def validate_one_extraction(cls, model: "ExtractionAction"):
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
            self.network_call.replace(pattern, replacement)
        if self.llm:
            self.llm.replace(pattern, replacement)
        if self.python_script:
            self.python_script.replace(pattern, replacement)
        return self
