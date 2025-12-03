import argparse
import logging
from urllib.parse import urljoin

import httpx

from optexity.examples import download_pdf_url, i94, peachstate_medicaid, supabase_login
from optexity.utils.settings import Settings

logger = logging.getLogger(__name__)
settings = Settings()

logger.setLevel(logging.INFO)


def main(args):
    if args.example == "i94":
        example = i94
    elif args.example == "peachstate_medicaid":
        example = peachstate_medicaid
    elif args.example == "supabase_login":
        example = supabase_login
    elif args.example == "download_pdf_url":
        example = download_pdf_url
    else:
        raise ValueError(f"Invalid example: {args.example}")
    try:
        logger.info(f"➕ Adding example: {args.example}")
        headers = {"x-api-key": settings.API_KEY}
        with httpx.Client() as client:
            response = client.post(
                urljoin(
                    settings.SERVER_URL,
                    (
                        settings.ADD_EXAMPLE_ENDPOINT
                        if not args.update
                        else settings.UPDATE_EXAMPLE_ENDPOINT
                    ),
                ),
                headers=headers,
                json={
                    "automation": example.automation.model_dump(
                        exclude_unset=True, exclude_none=True, exclude_defaults=True
                    ),
                    "description": example.description,
                    "endpoint_name": example.endpoint_name,
                },
            )
            response.raise_for_status()
            logger.info(f"✓ Example added successfully: {response.json()}")
    except Exception as e:
        logger.error(f"❌ Error adding example: {response.json()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--example",
        type=str,
        choices=["i94", "peachstate_medicaid", "supabase_login", "download_pdf_url"],
        required=True,
    )
    parser.add_argument(
        "--update",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()

    main(args)
