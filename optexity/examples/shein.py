from optexity.schema.actions.extraction_action import (
    ExtractionAction,
    ScreenshotExtraction,
)
from optexity.schema.actions.interaction_action import (
    ClickElementAction,
    InteractionAction,
)
from optexity.schema.automation import ActionNode, Automation, Parameters

shein_test = Automation(
    url="https://us.shein.com/",
    parameters=Parameters(
        input_parameters={},
        generated_parameters={},
    ),
    nodes=[
        # ActionNode(
        #     before_sleep_time=5,
        #     interaction_action=InteractionAction(
        #         max_tries=3,
        #         click_element=ClickElementAction(
        #             command="""get_by_role("link", name="Random Click")""",
        #             prompt_instructions="Click the Random Click link",
        #             assert_locator_presence=True,
        #         ),
        #     ),
        # ),
        ActionNode(
            before_sleep_time=5,
            extraction_action=ExtractionAction(
                screenshot=ScreenshotExtraction(filename="shein_homepage.png"),
            ),
        ),
        # ActionNode(
        #     before_sleep_time=5,
        #     interaction_action=InteractionAction(
        #         close_overlay_popup=CloseOverlayPopupAction(max_steps=20),
        #     ),
        # ),
    ],
)
