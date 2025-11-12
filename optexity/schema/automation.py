from pydantic import BaseModel, model_validator

from optexity.schema.actions.assertion_action import AssertionAction
from optexity.schema.actions.extraction_action import ExtractionAction
from optexity.schema.actions.interaction_action import InteractionAction
from optexity.schema.actions.misc_action import PythonScriptAction
from optexity.schema.actions.two_factor_auth_action import Fetch2faAction


class ActionNode(BaseModel):
    interaction_action: InteractionAction | None = None
    assertion_action: AssertionAction | None = None
    extraction_action: ExtractionAction | None = None
    python_script_action: PythonScriptAction | None = None
    fetch_2fa_action: Fetch2faAction | None = None
    before_sleep_time: float = 0.0
    end_sleep_time: float = 1.0
    expect_new_tab: bool = False
    max_new_tab_wait_time: float = 0.0

    @model_validator(mode="after")
    def validate_one_node(cls, model: "ActionNode"):
        """Ensure exactly one of the node types is set and matches the type."""
        provided = {
            "interaction_action": model.interaction_action,
            "assertion_action": model.assertion_action,
            "extraction_action": model.extraction_action,
            "python_script_action": model.python_script_action,
            "fetch_2fa_action": model.fetch_2fa_action,
        }
        non_null = [k for k, v in provided.items() if v is not None]

        if len(non_null) != 1:
            raise ValueError(
                "Exactly one of interaction_action, assertion_action, extraction_action, python_script_action, or fetch_2fa_action must be provided"
            )

        assert (
            model.end_sleep_time >= 0 and model.end_sleep_time <= 10
        ), "end_sleep_time must be greater than 0 and less than 10"
        assert (
            model.max_new_tab_wait_time >= 0 and model.max_new_tab_wait_time <= 10
        ), "max_new_tab_wait_time must be greater than 0 and less than 10"

        # --- Adjust defaults only if user didn't override them ---
        # We detect user-provided fields using model.__pydantic_fields_set__
        user_set = model.__pydantic_fields_set__

        if "end_sleep_time" not in user_set:
            model.end_sleep_time = (
                0.0
                if model.assertion_action
                or model.extraction_action
                or model.fetch_2fa_action
                else 1.0
            )

        if "before_sleep_time" not in user_set:
            model.before_sleep_time = 3.0 if model.extraction_action else 0.0

        if model.expect_new_tab:
            assert (
                model.interaction_action is not None
            ), "expect_new_tab is only allowed for interaction actions"
            model.max_new_tab_wait_time = 10.0
        else:
            model.max_new_tab_wait_time = 0.0

        return model

    def replace(self, pattern: str, replacement: str):
        if self.interaction_action:
            self.interaction_action.replace(pattern, replacement)
        if self.assertion_action:
            raise NotImplementedError(
                "Assertion replacement function is not implemented"
            )
        if self.extraction_action:
            self.extraction_action.replace(pattern, replacement)
        if self.python_script_action:
            pass
        if self.fetch_2fa_action:
            pass

        return self

    def replace_variables(self, variables: dict[str, list[str]]):
        for key, values in variables.items():
            for index, value in enumerate(values):
                pattern = f"{{{key}[{index}]}}"
                self.replace(pattern, value)

        return self


class ForLoopNode(BaseModel):
    # Loops through range of values of {variable_name[index]}
    variable_name: str
    nodes: list[ActionNode]


class IfElseNode(BaseModel):
    condition: str
    if_nodes: list[ActionNode]
    else_nodes: list[ActionNode] = []


class Parameters(BaseModel):
    input_parameters: dict[str, list[str]]
    generated_parameters: dict[str, list[str]]


class Automation(BaseModel):
    url: str
    parameters: Parameters
    nodes: list[ActionNode | ForLoopNode | IfElseNode]

    @model_validator(mode="after")
    def validate_parameters_with_examples(cls, model: "Automation"):
        ## TODO: static check that all parameters with examples are used in the nodes
        return model
