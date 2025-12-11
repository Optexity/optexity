from optexity.schema.automation import Automation

description = "I94 Example"
endpoint_name = "i94"
automation_json = {
    "url": "https://i94.cbp.dhs.gov/search/recent-search",
    "nodes": [
        {
            "end_sleep_time": 1,
            "before_sleep_time": 2,
            "python_script_action": {
                "execution_code": 'async def code_fn(page):\n    print("entering code_fn")\n    await page.evaluate(\n        """  const el = document.querySelector(\'mat-dialog-content\');  if (el) el.scrollTop = el.scrollHeight;"""\n    )\n    print("exiting code_fn")\n'
            },
        },
        {
            "end_sleep_time": 1,
            "interaction_action": {
                "click_element": {
                    "command": 'get_by_role("button", name="I ACKNOWLEDGE AND AGREE")',
                    "prompt_instructions": "Click the I ACKNOWLEDGE AND AGREE button",
                }
            },
        },
        {
            "end_sleep_time": 1,
            "interaction_action": {
                "input_text": {
                    "command": 'get_by_role("textbox", name="Please enter your first name")',
                    "input_text": "{first_name[0]}",
                    "prompt_instructions": "Enter the First Name",
                }
            },
        },
        {
            "end_sleep_time": 1,
            "interaction_action": {
                "input_text": {
                    "command": 'get_by_role("textbox", name="Please enter your last name")',
                    "input_text": "{last_name[0]}",
                    "prompt_instructions": "Enter the Last Name",
                }
            },
        },
        {
            "end_sleep_time": 1,
            "interaction_action": {
                "input_text": {
                    "command": 'get_by_role("textbox", name="Date of Birth")',
                    "input_text": "{date_of_birth[0]}",
                    "prompt_instructions": "Enter the Date of Birth",
                }
            },
        },
        {
            "end_sleep_time": 1,
            "interaction_action": {
                "input_text": {
                    "command": 'get_by_role("textbox", name="Please enter your document")',
                    "input_text": "{document_number[0]}",
                    "prompt_instructions": "Enter the Document Number",
                }
            },
        },
        {
            "end_sleep_time": 1,
            "interaction_action": {
                "input_text": {
                    "command": 'get_by_role("combobox", name="Please enter your document")',
                    "input_text": "{nationality[0]}",
                    "prompt_instructions": "Enter the Nationality",
                }
            },
        },
        {
            "end_sleep_time": 1,
            "interaction_action": {
                "click_element": {
                    "prompt_instructions": "Select {nationality[0]} from the options. Be careful to select the correct option. which will be of the format `nationality (code)`"
                }
            },
        },
        {
            "end_sleep_time": 1,
            "interaction_action": {
                "click_element": {
                    "command": 'get_by_role("button", name="Click to submit the form")',
                    "prompt_instructions": "Click the Submit button",
                }
            },
        },
        {
            "end_sleep_time": 0,
            "before_sleep_time": 3,
            "extraction_action": {
                "network_call": {
                    "url_pattern": "https://i94.cbp.dhs.gov/api/services/i94/recent"
                }
            },
        },
    ],
    "parameters": {
        "input_parameters": {
            "last_name": ["Last Name"],
            "first_name": ["First Name"],
            "nationality": ["IND"],
            "date_of_birth": ["MM/DD/YYYY"],
            "document_number": ["Document Number"],
        },
        "generated_parameters": {},
    },
    "browser_channel": "chrome",
}


automation = Automation.model_validate(automation_json)
