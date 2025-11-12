from optexity.schema.actions.interaction_action import (
    ClickElementAction,
    InputTextAction,
    InteractionAction,
)
from optexity.schema.automation import ActionNode, Automation, Parameters

supabase_login_test = Automation(
    url="https://supabase.com",
    parameters=Parameters(
        input_parameters={
            "username": ["test@test.com"],
            "password": ["password"],
        },
        generated_parameters={},
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
    ],
)
