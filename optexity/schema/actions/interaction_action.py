from enum import Enum, unique
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator

from optexity.schema.actions.prompts import overlay_popup_prompt


class Locator(BaseModel):
    regex_options: list[str] | None = None
    locator_class: str
    first_arg: str | int | None = None
    options: dict | None = None


class DialogAction(BaseModel):
    action: Literal["accept", "reject"]
    prompt_instructions: str


class BaseAction(BaseModel):
    xpath: str | None = None
    command: str | None = None
    prompt_instructions: str
    skip_command: bool = False
    skip_prompt: bool = False
    assert_locator_presence: bool = False

    @model_validator(mode="after")
    def validate_one_extraction(cls, model: "BaseAction"):
        """Ensure exactly one of the extraction types is set and matches the type."""

        provided = {
            "xpath": model.xpath,
            "command": model.command,
        }
        non_null = [k for k, v in provided.items() if v is not None]

        if len(non_null) > 1:
            raise ValueError("Exactly one of xpath, command must be provided")

        if model.assert_locator_presence:
            assert (
                model.command is not None
            ), "command is required when assert_locator_presence is True"

        if model.command is not None and model.command.strip() == "":
            model.command = None

        return model

    def replace(self, pattern: str, replacement: str):
        if self.prompt_instructions:
            self.prompt_instructions = self.prompt_instructions.replace(
                pattern, replacement
            )
        if self.xpath:
            self.xpath = self.xpath.replace(pattern, replacement)
        if self.command:
            self.command = self.command.replace(pattern, replacement).strip('"')


class CheckAction(BaseAction):
    pass


class UncheckAction(BaseAction):
    pass


class SelectOptionAction(BaseAction):
    select_values: list[str]
    expect_download: bool = False
    download_filename: str | None = None

    @model_validator(mode="after")
    def set_download_filename(cls, model: "SelectOptionAction"):

        if model.expect_download and model.download_filename is None:
            model.download_filename = str(uuid4())

        return model

    def replace(self, pattern: str, replacement: str):
        super().replace(pattern, replacement)
        if self.select_values:
            self.select_values = [
                value.replace(pattern, replacement).strip('"')
                for value in self.select_values
            ]
        if self.download_filename:
            self.download_filename = self.download_filename.replace(
                pattern, replacement
            ).strip('"')
        return self


class ClickElementAction(BaseAction):
    double_click: bool = False
    expect_download: bool = False
    download_filename: str | None = None

    @model_validator(mode="after")
    def set_download_filename(cls, model: "ClickElementAction"):

        if model.expect_download and model.download_filename is None:
            model.download_filename = str(uuid4())

        return model

    def replace(self, pattern: str, replacement: str):
        super().replace(pattern, replacement)
        if self.download_filename:
            self.download_filename = self.download_filename.replace(
                pattern, replacement
            ).strip('"')
        return self


class InputTextAction(BaseAction):
    input_text: str | None = None
    is_slider: bool = False
    fill_or_type: Literal["fill", "type"] = "fill"
    press_enter: bool = False

    @model_validator(mode="after")
    def validate_press_enter(self):
        if self.press_enter and self.command is None:
            raise ValueError("command is required when press_enter is True")
        return self

    def replace(self, pattern: str, replacement: str):
        super().replace(pattern, replacement)
        if self.input_text:
            self.input_text = self.input_text.replace(pattern, replacement).strip('"')
        return self


class DownloadUrlAsPdfAction(BaseModel):
    # Used when the current page is a PDF and we want to download it
    download_filename: str = Field(default_factory=lambda: str(uuid4()))
    url: str | None = None

    def replace(self, pattern: str, replacement: str):
        if self.download_filename:
            self.download_filename = self.download_filename.replace(
                pattern, replacement
            ).strip('"')
        return self


class ScrollAction(BaseModel):
    down: bool  # True to scroll down, False to scroll up


class UploadFileAction(BaseAction):
    file_path: str

    def replace(self, pattern: str, replacement: str):
        if self.file_path:
            self.file_path = self.file_path.replace(pattern, replacement).strip('"')
        return self


class GoToUrlAction(BaseModel):
    url: str
    new_tab: bool = False  # True to open in new tab, False to navigate in current tab

    def replace(self, pattern: str, replacement: str):
        if self.url:
            self.url = self.url.replace(pattern, replacement).strip('"')
        return self


class GoBackAction(BaseModel):
    pass


class SwitchTabAction(BaseModel):
    tab_index: int


class CloseCurrentTabAction(BaseModel):
    pass


class CloseAllButLastTabAction(BaseModel):
    pass


class CloseTabsUntil(BaseModel):
    matching_url: str | None = None
    tab_index: int | None = None

    @model_validator(mode="after")
    def validate_one_of_matching_url_or_tab_index(self):
        non_null = [k for k, v in self.model_dump().items() if v is not None]
        if len(non_null) != 1:
            raise ValueError(
                "Exactly one of matching_url or tab_index must be provided"
            )
        return self

    def replace(self, pattern: str, replacement: str):
        if self.matching_url:
            self.matching_url = self.matching_url.replace(pattern, replacement).strip(
                '"'
            )
        return self


@unique
class KeyPressType(str, Enum):
    ENTER = "Enter"
    TAB = "Tab"
    DELETE = "Delete"
    BACKSPACE = "Backspace"
    ESCAPE = "Escape"


class KeyPressAction(BaseModel):
    type: KeyPressType


class AgenticTask(BaseModel):
    task: str
    max_steps: int
    backend: Literal["browser_use", "browserbase"]
    use_vision: bool = False
    keep_alive: bool = True

    def replace(self, pattern: str, replacement: str):
        if self.task:
            self.task = self.task.replace(pattern, replacement).strip('"')
        return self


class CloseOverlayPopupAction(AgenticTask):
    task: str = Field(default=overlay_popup_prompt)
    max_steps: int = Field(default=5)
    backend: Literal["browser_use", "browserbase"] = Field(default="browser_use")
    use_vision: bool = Field(default=True)
    keep_alive: bool = Field(default=True)


class InteractionAction(BaseModel):
    start_2fa_timer: bool = False
    max_tries: int = 10
    max_timeout_seconds_per_try: float = 1.0
    click_element: ClickElementAction | None = None
    input_text: InputTextAction | None = None
    select_option: SelectOptionAction | None = None
    check: CheckAction | None = None
    uncheck: UncheckAction | None = None
    download_url_as_pdf: DownloadUrlAsPdfAction | None = None
    scroll: ScrollAction | None = None
    upload_file: UploadFileAction | None = None
    go_to_url: GoToUrlAction | None = None
    go_back: GoBackAction | None = None
    switch_tab: SwitchTabAction | None = None
    close_current_tab: CloseCurrentTabAction | None = None
    close_all_but_last_tab: CloseAllButLastTabAction | None = None
    close_tabs_until: CloseTabsUntil | None = None
    agentic_task: AgenticTask | None = None
    close_overlay_popup: CloseOverlayPopupAction | None = None
    key_press: KeyPressAction | None = None

    @model_validator(mode="after")
    def validate_one_interaction(cls, model: "InteractionAction"):
        """Ensure exactly one of the interaction types is set and matches the type."""
        provided = {
            "click_element": model.click_element,
            "input_text": model.input_text,
            "select_option": model.select_option,
            "check": model.check,
            "uncheck": model.uncheck,
            "download_url_as_pdf": model.download_url_as_pdf,
            "scroll": model.scroll,
            "upload_file": model.upload_file,
            "go_to_url": model.go_to_url,
            "go_back": model.go_back,
            "switch_tab": model.switch_tab,
            "close_current_tab": model.close_current_tab,
            "close_all_but_last_tab": model.close_all_but_last_tab,
            "close_tabs_until": model.close_tabs_until,
            "agentic_task": model.agentic_task,
            "close_overlay_popup": model.close_overlay_popup,
            "key_press": model.key_press,
        }
        non_null = [k for k, v in provided.items() if v is not None]

        if len(non_null) != 1:
            raise ValueError(
                "Exactly one of click_element, input_text, select_option, check, uncheck, download_url_as_pdf, scroll, upload_file, go_to_url, go_back, switch_tab, close_current_tab, close_all_but_last_tab, close_tabs_until, key_press, or agentic_task must be provided"
            )

        if model.start_2fa_timer:
            assert (
                model.click_element is not None
            ), "2fa timer can only be started when clicking on an element"

        if (
            (model.click_element and model.click_element.skip_prompt)
            or (model.input_text and model.input_text.skip_prompt)
            or (model.select_option and model.select_option.skip_prompt)
        ):
            model.max_tries = 5

        return model

    def replace(self, pattern: str, replacement: str):
        if self.click_element:
            self.click_element.replace(pattern, replacement)
        if self.input_text:
            self.input_text.replace(pattern, replacement)
        if self.select_option:
            self.select_option.replace(pattern, replacement)
        if self.check:
            self.check.replace(pattern, replacement)
        if self.uncheck:
            self.uncheck.replace(pattern, replacement)
        if self.download_url_as_pdf:
            self.download_url_as_pdf.replace(pattern, replacement)
        if self.close_tabs_until:
            self.close_tabs_until.replace(pattern, replacement)
        if self.agentic_task:
            self.agentic_task.replace(pattern, replacement)
        if self.close_overlay_popup:
            self.close_overlay_popup.replace(pattern, replacement)
        if self.go_to_url:
            self.go_to_url.replace(pattern, replacement)
        if self.upload_file:
            self.upload_file.replace(pattern, replacement)

        return self
