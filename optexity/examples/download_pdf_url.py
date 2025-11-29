from optexity.schema.actions.interaction_action import (
    DownloadUrlAsPdfAction,
    GoToUrlAction,
    InteractionAction,
)
from optexity.schema.automation import ActionNode, Automation, Parameters

download_pdf_url_test = Automation(
    url="about:blank",
    parameters=Parameters(
        input_parameters={
            "pdf_url": ["https://s24.q4cdn.com/216390268/files/doc_downloads/test.pdf"],
        },
        generated_parameters={},
    ),
    nodes=[
        ActionNode(
            interaction_action=InteractionAction(
                go_to_url=GoToUrlAction(
                    url="{pdf_url[0]}",
                ),
            ),
        ),
        ActionNode(
            interaction_action=InteractionAction(
                download_url_as_pdf=DownloadUrlAsPdfAction(
                    download_filename="example.pdf",
                ),
            ),
        ),
    ],
)
