from typing import Literal

from pydantic import BaseModel, model_validator


class FetchEmail2FaAction(BaseModel):
    email_address: str
    subject: str
    service: Literal["gmail"]


class FetchTotp2FaAction(BaseModel):
    totp_secret: str


class Fetch2faApiCallAction(BaseModel):
    api_call_url: str
    api_call_method: Literal["GET", "POST"]
    api_call_headers: dict
    api_call_body: dict


class Fetch2faAction(BaseModel):
    fetch_email_2fa_action: FetchEmail2FaAction | None = None
    fetch_totp_2fa_action: FetchTotp2FaAction | None = None
    fetch_2fa_api_call_action: Fetch2faApiCallAction | None = None
    output_variable_name: str

    @model_validator(mode="after")
    def validate_one_fetch_2fa_action(cls, model: "Fetch2faAction"):
        """Ensure exactly one of the extraction types is set and matches the type."""
        provided = {
            "fetch_email_2fa_action": model.fetch_email_2fa_action,
            "fetch_totp_2fa_action": model.fetch_totp_2fa_action,
            "fetch_2fa_api_call_action": model.fetch_2fa_api_call_action,
        }
        non_null = [k for k, v in provided.items() if v is not None]

        if len(non_null) != 1:
            raise ValueError(
                "Exactly one of fetch_email_2fa_action, fetch_totp_2fa_action, or fetch_2fa_api_call_action must be provided"
            )

        return model
