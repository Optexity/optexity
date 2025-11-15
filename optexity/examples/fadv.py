from optexity.schema.actions.interaction_action import (
    ClickElementAction,
    InputTextAction,
    InteractionAction,
)
from optexity.schema.automation import ActionNode, Automation, Parameters

# test incorrect password
# test correct login
# test login somewhere else before
# test login somewhere else after
# secret question not city


fadv_test = Automation(
    url="https://enterprise.fadv.com/pub/l/login/userLogin.do",
    parameters=Parameters(
        input_parameters={
            "client_id": ["********"],
            "user_id": ["********"],
            "password": ["********"],
            "secret_answer": ["********"],
        },
        generated_parameters={},
    ),
    nodes=[
        ActionNode(
            interaction_action=InteractionAction(
                input_text=InputTextAction(
                    command="""locator('#new-login-iframe').content_frame.get_by_role('textbox', name='Client ID *')""",
                    input_text="{client_id[0]}",
                    prompt_instructions="Enter the client id",
                )
            )
        ),
        ActionNode(
            interaction_action=InteractionAction(
                input_text=InputTextAction(
                    command="""locator('#new-login-iframe').content_frame.get_by_role('textbox', name='User ID *')""",
                    input_text="{user_id[0]}",
                    prompt_instructions="Enter the user id",
                )
            )
        ),
        ActionNode(
            interaction_action=InteractionAction(
                input_text=InputTextAction(
                    command="""locator('#new-login-iframe').content_frame.get_by_role('textbox', name='Password *')""",
                    input_text="{password[0]}",
                    prompt_instructions="Enter the password",
                )
            )
        ),
        ActionNode(
            interaction_action=InteractionAction(
                click_element=ClickElementAction(
                    command="""locator('#new-login-iframe').content_frame.get_by_role('button', name='Login')""",
                    prompt_instructions="Click the Login button",
                )
            )
        ),
        ActionNode(
            interaction_action=InteractionAction(
                input_text=InputTextAction(
                    command="""locator('#new-login-iframe').content_frame.get_by_role('textbox', name='In what city or town did your')""",
                    input_text="{secret_answer[0]}",
                    prompt_instructions="Fill {secret_answer[0]} into the textbox which asks for secret answer. The questions can be anything. But the value should be {secret_answer[0]}.",
                )
            )
        ),
        ActionNode(
            interaction_action=InteractionAction(
                click_element=ClickElementAction(
                    command="""locator('#new-login-iframe').content_frame.get_by_role('button', name='Submit')""",
                    prompt_instructions="Click the Submit button",
                    assert_locator_presence=True,
                )
            )
        ),
        ActionNode(
            interaction_action=InteractionAction(
                click_element=ClickElementAction(
                    command="""locator('#new-login-iframe').content_frame.get_by_role('button', name='Proceed')""",
                    prompt_instructions="Click the Proceed button",
                    skip_prompt=True,
                )
            )
        ),
        ActionNode(
            interaction_action=InteractionAction(
                click_element=ClickElementAction(
                    command="""locator('#new-login-iframe').content_frame.get_by_role('button', name='I Agree')""",
                    prompt_instructions="Click the I Agree button",
                    skip_prompt=True,
                )
            )
        ),
        ActionNode(
            interaction_action=InteractionAction(
                click_element=ClickElementAction(
                    command="""locator('#new-login-iframe').content_frame.get_by_role('button', name='Continue')""",
                    prompt_instructions="Click the Continue button",
                    skip_prompt=True,
                )
            )
        ),
        ActionNode(
            interaction_action=InteractionAction(
                input_text=InputTextAction(
                    command="""get_by_role('textbox', name='Search')""",
                    input_text="1234567890",
                    prompt_instructions="Enter the search term",
                    assert_locator_presence=True,
                )
            )
        ),
    ],
)
