from optexity.schema.actions.extraction_action import (
    ExtractionAction,
    NetworkCallExtraction,
)
from optexity.schema.actions.interaction_action import (
    ClickElementAction,
    InputTextAction,
    InteractionAction,
)
from optexity.schema.actions.misc_action import PythonScriptAction
from optexity.schema.automation import ActionNode, Automation, Parameters

description = "I94 Example"
endpoint_name = "i94"
automation = Automation(
    browser_channel="chrome",
    url="https://i94.cbp.dhs.gov/search/recent-search",
    parameters=Parameters(
        input_parameters={
            "first_name": ["First Name"],
            "last_name": ["Last Name"],
            "nationality": ["IND"],
            "date_of_birth": ["MM/DD/YYYY"],
            "document_number": ["Document Number"],
        },
        generated_parameters={},
    ),
    nodes=[
        ActionNode(
            python_script_action=PythonScriptAction(
                execution_code="""async def code_fn(page):\n    print(\"entering code_fn\")\n    await page.evaluate(\n        \"\"\"  const el = document.querySelector('mat-dialog-content');  if (el) el.scrollTop = el.scrollHeight;\"\"\"\n    )\n    print(\"exiting code_fn\")\n"""
            ),
            before_sleep_time=2.0,
        ),
        ActionNode(
            interaction_action=InteractionAction(
                click_element=ClickElementAction(
                    command="""get_by_role("button", name="I ACKNOWLEDGE AND AGREE")""",
                    prompt_instructions="Click the I ACKNOWLEDGE AND AGREE button",
                )
            )
        ),
        ActionNode(
            interaction_action=InteractionAction(
                input_text=InputTextAction(
                    command="""get_by_role("textbox", name="Please enter your first name")""",
                    input_text="{first_name[0]}",
                    prompt_instructions="Enter the First Name",
                )
            )
        ),
        ActionNode(
            interaction_action=InteractionAction(
                input_text=InputTextAction(
                    command="""get_by_role("textbox", name="Please enter your last name")""",
                    input_text="{last_name[0]}",
                    prompt_instructions="Enter the Last Name",
                )
            )
        ),
        ActionNode(
            interaction_action=InteractionAction(
                input_text=InputTextAction(
                    command="""get_by_role("textbox", name="Date of Birth")""",
                    input_text="{date_of_birth[0]}",
                    prompt_instructions="Enter the Date of Birth",
                )
            )
        ),
        ActionNode(
            interaction_action=InteractionAction(
                input_text=InputTextAction(
                    command="""get_by_role("textbox", name="Please enter your document")""",
                    input_text="{document_number[0]}",
                    prompt_instructions="Enter the Document Number",
                )
            )
        ),
        ActionNode(
            interaction_action=InteractionAction(
                input_text=InputTextAction(
                    command="""get_by_role("combobox", name="Please enter your document")""",
                    input_text="{nationality[0]}",
                    prompt_instructions="Enter the Nationality",
                )
            )
        ),
        ActionNode(
            interaction_action=InteractionAction(
                click_element=ClickElementAction(
                    prompt_instructions="Select {nationality[0]} from the options. Be careful to select the correct option. which will be of the format `nationality (code)`",
                )
            )
        ),
        ActionNode(
            interaction_action=InteractionAction(
                click_element=ClickElementAction(
                    command="""get_by_role("button", name="Click to submit the form")""",
                    prompt_instructions="Click the Submit button",
                )
            )
        ),
        ActionNode(
            extraction_action=ExtractionAction(
                network_call=NetworkCallExtraction(
                    url_pattern="https://i94.cbp.dhs.gov/api/services/i94/recent"
                )
            ),
        ),
    ],
)
