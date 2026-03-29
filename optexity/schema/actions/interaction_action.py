import re
from enum import Enum, unique
from typing import Any, Literal
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
    coordinates: tuple[int, int] | None = None
    keyword: str | None = None
    command: str | None = None
    prompt_instructions: str
    skip_command: bool = False
    skip_prompt: bool = False
    assert_locator_presence: bool = False

    @model_validator(mode="after")
    def validate_one_extraction(self):
        """Ensure exactly one of the extraction types is set and matches the type."""

        provided = {"xpath": self.xpath, "command": self.command}
        non_null = [k for k, v in provided.items() if v is not None]

        if len(non_null) > 1:
            raise ValueError("Exactly one of xpath, command must be provided")

        if self.assert_locator_presence:
            assert (
                self.command is not None
            ), "command is required when assert_locator_presence is True"

        if self.command is not None and self.command.strip() == "":
            self.command = None

        return self

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


class HoverAction(BaseAction):
    pass


class SelectOptionAction(BaseAction):
    select_values: list[str]
    expect_download: bool = False
    download_filename: str | None = None

    @model_validator(mode="after")
    def set_download_filename(self):

        if self.expect_download and self.download_filename is None:
            self.download_filename = str(uuid4())

        return self

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


class ClickElementAction(BaseAction):
    double_click: bool = False
    expect_download: bool = False
    download_filename: str | None = None
    button: Literal["left", "right", "middle"] = "left"

    @model_validator(mode="after")
    def set_download_filename(self):

        if self.expect_download and self.download_filename is None:
            self.download_filename = str(uuid4())

        return self

    def replace(self, pattern: str, replacement: str):
        super().replace(pattern, replacement)
        if self.download_filename:
            self.download_filename = self.download_filename.replace(
                pattern, replacement
            ).strip('"')


class InputTextAction(BaseAction):
    input_text: str | None = None
    is_slider: bool = False
    fill_or_type: Literal["fill", "type", "key_press"] = "fill"
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


class DownloadUrlAsPdfAction(BaseModel):
    # Used when the current page is a PDF and we want to download it
    download_filename: str = Field(default_factory=lambda: str(uuid4()))
    url: str | None = None

    def replace(self, pattern: str, replacement: str):
        if self.download_filename:
            self.download_filename = self.download_filename.replace(
                pattern, replacement
            ).strip('"')


class ScrollAction(BaseModel):
    down: bool = True  # True to scroll down, False to scroll up
    amount: int = -1  ## -1 means scroll max amount

    @model_validator(mode="after")
    def validate_amount(self):
        if self.amount is None or (self.amount < 0 and self.amount != -1):
            raise ValueError("amount must be -1 or positive")
        return self


class UploadFileAction(BaseAction):
    file_path: str

    def replace(self, pattern: str, replacement: str):
        if self.file_path:
            self.file_path = self.file_path.replace(pattern, replacement).strip('"')


class GoToUrlAction(BaseModel):
    url: str
    new_tab: bool = False  # True to open in new tab, False to navigate in current tab

    def replace(self, pattern: str, replacement: str):
        if self.url:
            self.url = self.url.replace(pattern, replacement).strip('"')


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


@unique
class KeyPressType(str, Enum):
    ENTER = "Enter"
    TAB = "Tab"
    DELETE = "Delete"
    BACKSPACE = "Backspace"
    ESCAPE = "Escape"
    ZERO = "0"
    ONE = "1"
    TWO = "2"
    THREE = "3"
    FOUR = "4"
    FIVE = "5"
    SIX = "6"
    SEVEN = "7"
    EIGHT = "8"
    NINE = "9"
    SLASH = "/"
    SPACE = "Space"
    CTRL = "Ctrl"
    ALT = "Alt"
    SHIFT = "Shift"
    META = "Meta"
    COMMAND = "Command"
    OPTION = "Option"
    CMD = "Cmd"


class KeyPressAction(BaseModel):
    type: KeyPressType | Any

    @model_validator(mode="after")
    def validate_type(self):
        if self.type is None:
            raise ValueError("type is required")
        return self

    def replace(self, pattern: str, replacement: str):
        if self.type:
            self.type = self.type.replace(pattern, replacement).strip('"')
        return self


class KeyCombinationAction(BaseModel):
    key_combination: list[KeyPressType | str]

    @model_validator(mode="after")
    def validate_key_combination(self):
        key_combination = []
        for key in self.key_combination:
            if key is None:
                raise ValueError("key_combination is required")
            key = key.strip()
            if isinstance(key, str):
                try:
                    key = KeyPressType(key)
                except ValueError:
                    if not re.fullmatch(r"[a-zA-Z]", key) is not None:
                        raise ValueError(f"Invalid key: {key}")
                    key_combination.append(key)
        self.key_combination = key_combination
        return self


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
    max_tries: int = 10
    max_timeout_seconds_per_try: float = 1.0
    click_element: ClickElementAction | None = None
    input_text: InputTextAction | None = None
    select_option: SelectOptionAction | None = None
    check: CheckAction | None = None
    uncheck: UncheckAction | None = None
    hover: HoverAction | None = None
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
    key_combination: KeyCombinationAction | None = None

    @model_validator(mode="after")
    def validate_one_interaction(self):
        """Ensure exactly one of the interaction types is set and matches the type."""
        provided = {
            "click_element": self.click_element,
            "input_text": self.input_text,
            "select_option": self.select_option,
            "check": self.check,
            "uncheck": self.uncheck,
            "hover": self.hover,
            "download_url_as_pdf": self.download_url_as_pdf,
            "scroll": self.scroll,
            "upload_file": self.upload_file,
            "go_to_url": self.go_to_url,
            "go_back": self.go_back,
            "switch_tab": self.switch_tab,
            "close_current_tab": self.close_current_tab,
            "close_all_but_last_tab": self.close_all_but_last_tab,
            "close_tabs_until": self.close_tabs_until,
            "agentic_task": self.agentic_task,
            "close_overlay_popup": self.close_overlay_popup,
            "key_press": self.key_press,
            "key_combination": self.key_combination,
        }
        non_null = [k for k, v in provided.items() if v is not None]

        if len(non_null) != 1:
            raise ValueError(
                "Exactly one of click_element, input_text, select_option, check, uncheck, hover, download_url_as_pdf, scroll, upload_file, go_to_url, go_back, switch_tab, close_current_tab, close_all_but_last_tab, close_tabs_until, key_press, or agentic_task must be provided"
            )

        if not self.max_tries and (
            (self.click_element and self.click_element.skip_prompt)
            or (self.input_text and self.input_text.skip_prompt)
            or (self.select_option and self.select_option.skip_prompt)
        ):
            self.max_tries = 5

        return self

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
        if self.hover:
            self.hover.replace(pattern, replacement)
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
        if self.key_press:
            self.key_press.replace(pattern, replacement)

        return self
