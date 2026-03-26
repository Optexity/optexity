from typing import Literal

from pydantic import BaseModel, Field, model_validator


class CaptchaAction(BaseModel):
    locator: str
    secondary_locator: str | None = None
    wait_time: float = 2.0  # Seconds to wait after trigger click for captcha to appear
    llm_provider: Literal["gemini", "openai", "anthropic"] = "gemini"
    llm_model_name: str = "gemini-2.5-pro"

    config: dict = Field(
        default_factory=lambda: {
            "primary_click_x_offset": 0,
            "primary_click_y_offset": 0,
            "grid_top_offset": 100,
            "grid_bottom_trim": 100,
            "max_captcha_retries": 3,
            "thinking_budget_tokens": None,  # Set to int (e.g. 8000) to enable extended thinking (Anthropic only)
        }
    )

    @model_validator(mode="after")
    def merge_config_with_defaults(self):
        defaults = {
            "primary_click_x_offset": 0,
            "primary_click_y_offset": 0,
            "grid_top_offset": 100,
            "grid_bottom_trim": 100,
            "max_captcha_retries": 3,
            "thinking_budget_tokens": None,
        }
        # Merge: defaults first, then override with any values provided in JSON
        self.config = {**defaults, **self.config}
        return self

    def replace(self, pattern: str, replacement: str):
        self.locator = self.locator.replace(pattern, replacement)
        self.secondary_locator = self.secondary_locator.replace(pattern, replacement)
        return self
