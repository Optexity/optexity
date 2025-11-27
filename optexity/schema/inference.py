from pydantic import BaseModel, model_validator


class InferenceRequest(BaseModel):
    endpoint_name: str
    input_parameters: dict[str, list[str]]
    unique_parameter_names: list[str]

    @model_validator(mode="after")
    def validate_unique_parameter_names(self):
        for unique_parameter_name in self.unique_parameter_names:
            if unique_parameter_name not in self.input_parameters:
                raise ValueError(
                    f"unique_parameter_name {unique_parameter_name} not found in input_parameters"
                )
        return self
