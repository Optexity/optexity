from optexity.schema.actions.extraction_action import ExtractionAction, LLMExtraction
from optexity.schema.actions.interaction_action import (
    ClickElementAction,
    GoBackAction,
    InputTextAction,
    InteractionAction,
    SelectOptionAction,
)
from optexity.schema.automation import (
    ActionNode,
    Automation,
    ForLoopNode,
    ParametersWithExamples,
)

pshpgeorgia_login_test = Automation(
    name="PSHP Georgia Login Test",
    url="https://sso.entrykeyid.com/as/authorization.oauth2?response_type=code&client_id=f6a6219c-be42-421b-b86c-e4fc509e2e87&scope=openid%20profile&state=_igWklSsnrkO5DQfjBMMuN41ksMJePZQ_SM_61wTJlA%3D&redirect_uri=https://provider.pshpgeorgia.com/careconnect/login/oauth2/code/pingcloud&code_challenge_method=S256&nonce=xG41TJjco_x7Vs_MQgcS3bw5njLiJsXCqvO-V8THmY0&code_challenge=ZTaVHaZCNFTejXNJo51RlJ3Kv9dH0tMODPTqO7hiP3A&app_origin=https://provider.pshpgeorgia.com/careconnect/login/oauth2/code/pingcloud&brand=pshpgeorgia",
    parameters_with_examples=ParametersWithExamples(
        input_parameters={
            "username": [],
            "password": [],
            "plan_type": [],
            "member_id": [],
            "dob": [],
        },
        generated_parameter={},
    ),
    description="Login to PSHP Georgia",
    nodes=[
        ActionNode(
            interaction_action=InteractionAction(
                input_text=InputTextAction(
                    command="""get_by_test_id("text-field")""",
                    input_text="{username[0]}",
                    prompt_instructions="Enter the email in the text field",
                )
            )
        ),
        ActionNode(
            interaction_action=InteractionAction(
                click_element=ClickElementAction(
                    command="""get_by_role("button", name="Continue")""",
                    prompt_instructions="Click the Continue button",
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
                    command="""get_by_role("button", name="Login")""",
                    prompt_instructions="Click the Login button",
                )
            )
        ),
    ],
)


pshpgeorgia_medicaid_test = Automation(
    name="PSHP Georgia Medicaid Test",
    description="Get the PSHP Georgia Medicaid application",
    url="https://provider.pshpgeorgia.com/careconnect/login",
    parameters_with_examples=ParametersWithExamples(
        input_parameters={},
        generated_parameter={},
    ),
    nodes=[
        ActionNode(
            interaction_action=InteractionAction(
                select_option=SelectOptionAction(
                    command="""get_by_label("Plan Type")""",
                    select_values=["{plan_type[0]}"],
                    prompt_instructions="Select the Plan Type 8774789",
                )
            )
        ),
        ActionNode(
            interaction_action=InteractionAction(
                click_element=ClickElementAction(
                    command="""get_by_role("button", name="GO")""",
                    prompt_instructions="Click the GO button",
                )
            )
        ),
        ActionNode(
            interaction_action=InteractionAction(
                input_text=InputTextAction(
                    command="""get_by_test_id("MemberIDOrLastName")""",
                    input_text="{member_id[0]}",
                    prompt_instructions="Enter the Member ID or Last Name",
                )
            ),
        ),
        ActionNode(
            interaction_action=InteractionAction(
                input_text=InputTextAction(
                    command="""locator("#tDatePicker")""",
                    input_text="{dob[0]}",
                    prompt_instructions="Enter the Date of Birth",
                )
            ),
        ),
        ActionNode(
            interaction_action=InteractionAction(
                click_element=ClickElementAction(
                    command="""get_by_role("combobox", name="Select Action Type Select")""",
                    prompt_instructions="Click the Select Action Type Select combobox",
                )
            )
        ),
        ActionNode(
            interaction_action=InteractionAction(
                click_element=ClickElementAction(
                    command="""get_by_test_id("ActionType-option-0")""",
                    prompt_instructions="Click the View eligibility & patient info option",
                )
            )
        ),
        ActionNode(
            interaction_action=InteractionAction(
                click_element=ClickElementAction(
                    command="""get_by_test_id("submitBtn")""",
                    prompt_instructions="Click the Submit button",
                )
            ),
            expect_new_tab=True,
        ),
        ActionNode(
            interaction_action=InteractionAction(
                click_element=ClickElementAction(
                    command="""get_by_label("Eligibility", exact=True).get_by_role("link", name="Authorizations")""",
                    prompt_instructions="Click the Authorizations link",
                )
            ),
        ),
        ActionNode(
            extraction_action=ExtractionAction(
                llm=LLMExtraction(
                    source=["axtree"],
                    extraction_format={
                        "authorization_numbers": "List[str]",
                    },
                    extraction_instructions="I am giving you an axtree of a webpage that shows the information about authorizations in a tabular format. Status, Auth Nbr, From Date, To Date, Diagnosis, Auth Type, Service. You need to output me a list of all Auth Nbr. Do not output any other information.",
                    output_variable_names=["authorization_numbers"],
                )
            ),
        ),
        ForLoopNode(
            variable_name="authorization_numbers",
            nodes=[
                ActionNode(
                    interaction_action=InteractionAction(
                        click_element=ClickElementAction(
                            command="""get_by_role("link", name="{authorization_numbers[index]}")""",
                            prompt_instructions="Click the Authorizations link for the authorization number {authorization_numbers[index]}",
                        )
                    ),
                ),
                ActionNode(
                    extraction_action=ExtractionAction(
                        llm=LLMExtraction(
                            source=["axtree"],
                            extraction_format={
                                "Auth Nbr": "str",
                                "End Date": "str",
                                "Auth Type": "str",
                                "Start Date": "str",
                                "Auth Status": "str",
                                "Service Type": "str",
                                "Units Approved": "str",
                                "Units Required": "str",
                            },
                            extraction_instructions="I am giving you an axtree of a webpage that shows information about authorizations, and I want the 8 following fields. 'Auth Status', 'Auth Nbr', 'Auth Type', 'Service Type', 'Start Date', 'End Date', 'Units Required', 'Units Approved'. Fields 'Auth Status', 'Auth Nbr', 'Auth Type' can be found in the top and rest of the information can be found in the tabular format. You need to output me key-value pairs for all 8 fields.",
                        )
                    ),
                ),
                ActionNode(
                    interaction_action=InteractionAction(go_back=GoBackAction())
                ),
            ],
        ),
    ],
)
