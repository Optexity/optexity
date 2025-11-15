from optexity.schema.actions.interaction_action import (
    ClickElementAction,
    InputTextAction,
    InteractionAction,
    SelectOptionAction,
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
            "start_date": ["*****"],
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
        # ActionNode(
        #     interaction_action=InteractionAction(
        #         click_element=ClickElementAction(
        #             command="""locator('#new-login-iframe').content_frame.get_by_role('button', name='Continue')""",
        #             prompt_instructions="Click the Continue button",
        #             skip_prompt=True,
        #         )
        #     )
        # ),
        ActionNode(
            interaction_action=InteractionAction(
                click_element=ClickElementAction(
                    command="""get_by_role('link', name='Employment Screening')""",
                    prompt_instructions="Click the Employment Screening link",
                    assert_locator_presence=True,
                )
            )
        ),
        ActionNode(
            interaction_action=InteractionAction(
                click_element=ClickElementAction(
                    command="""get_by_text('Search Orders')""",
                    prompt_instructions="Click the Search Orders link",
                )
            )
        ),
        ActionNode(
            interaction_action=InteractionAction(
                input_text=InputTextAction(
                    command="""locator('#ORDER_VIEWING_SEARCH_FROM_LBL')""",
                    input_text="{start_date[0]}",
                    prompt_instructions="Fill {start_date[0]} into the order viewing search date input field.",
                    assert_locator_presence=True,
                )
            )
        ),
        ActionNode(
            interaction_action=InteractionAction(
                click_element=ClickElementAction(
                    # command="""locator('#ext-gen93')""",
                    prompt_instructions="Click the search button at the bottom near the reset button.",
                ),
            ),
            before_sleep_time=5.0,
        ),
        ActionNode(
            interaction_action=InteractionAction(
                select_option=SelectOptionAction(
                    # command="""locator('#ext-gen124')""",
                    select_values=["ETC"],
                    prompt_instructions="Click on the <select >Select button below the <td >Actions /> and above the Order Details. ",
                    expect_download=True,
                    download_filename="fadv_orders.csv",
                )
            ),
            before_sleep_time=5.0,
        ),
        ActionNode(
            interaction_action=InteractionAction(
                click_element=ClickElementAction(
                    command="""get_by_text('Logout')""",
                    prompt_instructions="Click the Logout link",
                )
            )
        ),
    ],
)
#     {
#     "title": "Navigate to the Search Orders page",
#     "use_vision": false,
#     "subgoal_index": "subgoal0",
#     "subgoal_actions": [
#       {
#         "action": {
#           "type": "ClickElementAction",
#           "command": "page.getByRole('link', { name: 'Employment Screening' }).click()",
#           "download": false,
#           "filename": "1757535166244_fbyacob7sr6",
#           "locators": [
#             {
#               "options": {
#                 "name": "Employment Screening"
#               },
#               "first_arg": "link",
#               "locator_class": "getByRole"
#             }
#           ],
#           "double_click": false,
#           "optexity_bid": "c5f2382b-ac44-47ee-9a1c-1fe8007a01a6"
#         },
#         "sleep_time": 0,
#         "use_vision": false,
#         "action_index": "action0",
#         "locator_type": "fixed",
#         "can_skip_step": false,
#         "fixed_checkpoint": true,
#         "action_description": "Click on the 'Employment Screening' link."
#       },
#       {
#         "action": {
#           "type": "ClickElementAction",
#           "command": "page.getByText('Search Orders').click()",
#           "download": false,
#           "filename": "1757535169135_ie3rkz0sxhi",
#           "locators": [
#             {
#               "first_arg": "Search Orders",
#               "locator_class": "getByText"
#             }
#           ],
#           "double_click": false,
#           "optexity_bid": "455773c2-f225-4ede-9e3f-72d9b732cb54"
#         },
#         "sleep_time": 0,
#         "use_vision": false,
#         "action_index": "action1",
#         "locator_type": "fixed",
#         "can_skip_step": false,
#         "fixed_checkpoint": false,
#         "action_description": "Click on 'Search Orders'."
#       },
#       {
#         "action": {
#           "text": "15/Aug/2025",
#           "type": "InputTextAction",
#           "command": "page.locator('#ORDER_VIEWING_SEARCH_FROM_LBL').fill('15/Aug/2025')",
#           "do_type": false,
#           "filename": "1757535184697_57mewnpyiwj",
#           "locators": [
#             {
#               "first_arg": "#ORDER_VIEWING_SEARCH_FROM_LBL",
#               "locator_class": "locator"
#             }
#           ],
#           "is_slider": false,
#           "optexity_bid": "undefined"
#         },
#         "fill_type": "dynamic",
#         "sleep_time": 0,
#         "use_vision": false,
#         "action_index": "action3",
#         "locator_type": "fixed",
#         "can_skip_step": false,
#         "fixed_checkpoint": false,
#         "action_description": "Fill {start_date} in the order viewing search date input field.",
#         "fill_value_parameter_name": "start_date"
#       },
#       {
#         "action": {
#           "type": "ClickElementAction",
#           "command": "page.locator('#ext-gen93').click()",
#           "download": false,
#           "filename": "1757535250065_9wb6nexd5zh",
#           "locators": [
#             {
#               "first_arg": "#ext-gen93",
#               "locator_class": "locator"
#             }
#           ],
#           "double_click": false,
#           "optexity_bid": "92a802c0-2b0b-4588-93e5-2c43869f1618"
#         },
#         "sleep_time": 0,
#         "use_vision": false,
#         "action_index": "action4",
#         "locator_type": "dynamic",
#         "can_skip_step": false,
#         "fixed_checkpoint": false,
#         "action_description": "Click on search button at the buttom near the reset button."
#       },
#       {
#         "action": {
#           "type": "SelectOptionAction",
#           "values": [
#             "ETC"
#           ],
#           "command": "page.locator('#ext-gen124').selectOption('ETC')",
#           "download": true,
#           "filename": "1757535266703_l7fuadpfuf",
#           "locators": [
#             {
#               "first_arg": "#ext-gen124",
#               "locator_class": "locator"
#             }
#           ],
#           "optexity_bid": "80a1ff30-9d0f-4828-85f1-a9f4e3ac5294",
#           "download_type": "csv"
#         },
#         "sleep_time": 0,
#         "use_vision": false,
#         "action_index": "action5",
#         "locator_type": "dynamic",
#         "can_skip_step": false,
#         "fixed_checkpoint": false,
#         "action_description": "Click on the <select >Select button below the <td >Actions /> and above the Order Details. "
#       },
#       {
#         "action": {
#           "type": "ClickElementAction",
#           "command": "page.getByText('Logout').click();",
#           "download": false,
#           "filename": "1756333833270_fzc7ows7nzm",
#           "locators": [
#             {
#               "first_arg": "Logout",
#               "locator_class": "getByText"
#             }
#           ],
#           "double_click": false,
#           "optexity_bid": "160d132e-d9d9-4606-b23c-148c7fa01b83"
#         },
#         "sleep_time": 1,
#         "use_vision": false,
#         "action_index": "action16",
#         "locator_type": "fixed",
#         "fixed_checkpoint": false,
#         "action_description": "Click 'Logout' to logout"
#       }
#     ],
#     "loop_start_value": 0,
#     "end_value_addition": 0,
#     "can_number_of_steps_change": false
#   }
# )
