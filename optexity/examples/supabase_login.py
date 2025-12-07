from optexity.schema.actions.assertion_action import AssertionAction, LLMAssertion
from optexity.schema.actions.interaction_action import (
    ClickElementAction,
    InputTextAction,
    InteractionAction,
)
from optexity.schema.automation import (
    ActionNode,
    Automation,
    OnePasswordParameter,
    Parameters,
    SecureParameter,
)

description = "Supabase Login Example"
endpoint_name = "supabase_login"
automation = Automation(
    url="https://supabase.com",
    parameters=Parameters(
        input_parameters={},
        generated_parameters={},
        secure_parameters={
            "username": [
                SecureParameter(
                    onepassword=OnePasswordParameter(
                        vault_name="optexity_automation",
                        item_name="supabase",
                        field_name="username",
                    )
                )
            ],
            "password": [
                SecureParameter(
                    onepassword=OnePasswordParameter(
                        vault_name="optexity_automation",
                        item_name="supabase",
                        field_name="password",
                    )
                )
            ],
        },
    ),
    nodes=[
        ActionNode(
            interaction_action=InteractionAction(
                click_element=ClickElementAction(
                    command="""get_by_role("link", name="Sign in")""",
                    prompt_instructions="Click the Sign in link",
                )
            )
        ),
        ActionNode(
            interaction_action=InteractionAction(
                input_text=InputTextAction(
                    command="""get_by_role("textbox", name="Email")""",
                    input_text="{username[0]}",
                    prompt_instructions="Enter the email",
                )
            )
        ),
        ActionNode(
            interaction_action=InteractionAction(
                input_text=InputTextAction(
                    command="""get_by_role("textbox", name="Password")""",
                    input_text="{password[0]}",
                    prompt_instructions="Enter the password",
                )
            )
        ),
        ActionNode(
            interaction_action=InteractionAction(
                click_element=ClickElementAction(
                    command="""get_by_role("button", name="Sign In")""",
                    prompt_instructions="Click the Sign In button",
                )
            )
        ),
        ActionNode(
            assertion_action=AssertionAction(
                llm=LLMAssertion(
                    extraction_instructions="Check if the login was successful",
                )
            )
        ),
    ],
)
