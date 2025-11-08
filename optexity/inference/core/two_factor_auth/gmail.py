import base64
import logging
import os.path
import re

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

logger = logging.getLogger(__name__)


SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


def get_gmail_service():
    creds = None
    # Load token if it exists
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)

    # If no valid token, do OAuth flow
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())  # auto-refresh if possible
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)
        # Save token for next time
        with open("token.json", "w") as token_file:
            token_file.write(creds.to_json())
    return build("gmail", "v1", credentials=creds)


def extract_body_content(payload):
    """Extract body content from email payload, handling various structures"""

    # Check if the message has a direct body (simple structure)
    if "body" in payload and payload["body"].get("data"):
        return payload["body"]["data"], "direct"

    # Check parts for multipart messages
    parts = payload.get("parts", [])

    # First, try to find plain text
    for part in parts:
        if part["mimeType"] == "text/plain" and part["body"].get("data"):
            return part["body"]["data"], "text/plain"

    # If no plain text, try HTML
    for part in parts:
        if part["mimeType"] == "text/html" and part["body"].get("data"):
            return part["body"]["data"], "text/html"

    # Handle nested multipart structures
    for part in parts:
        if part["mimeType"].startswith("multipart/"):
            nested_data, nested_type = extract_body_content(part)
            if nested_data:
                return nested_data, nested_type

    return None, None


def decode_body_data(body_data):
    """Decode base64 encoded body data"""
    try:
        return base64.urlsafe_b64decode(body_data).decode("utf-8")
    except Exception as e:
        logger.error(f"Error decoding body data: {e}")
        return None


def clean_html_content(html_content):
    """Basic HTML tag removal for better readability"""

    # Remove HTML tags
    clean_text = re.sub("<.*?>", "", html_content)
    # Replace HTML entities
    clean_text = clean_text.replace("&nbsp;", " ")
    clean_text = clean_text.replace("&amp;", "&")
    clean_text = clean_text.replace("&lt;", "<")
    clean_text = clean_text.replace("&gt;", ">")
    # Clean up whitespace
    clean_text = re.sub("\s+", " ", clean_text).strip()
    return clean_text


def extract_verification_code(text):
    """Extract verification code from email text"""

    # Look for 6-digit numbers (common verification code format)
    pattern = r"\b\d{6}\b"
    matches = re.findall(pattern, text)
    return matches[0] if matches else None


def get_code_from_gmail(sender_email: str, subject: str, start_2fa_time: float):
    ## TODO: build it so that we fetch emails only after the start_2fa_time
    raise NotImplementedError("Fetching emails from Gmail is not implemented")
    service = get_gmail_service()
    results = (
        service.users()
        .messages()
        .list(userId="me", maxResults=1, q=f"from: {sender_email} subject: {subject}")
        .execute()
    )
    messages = results.get("messages", [])

    for i, msg in enumerate(messages, 1):
        msg_data = (
            service.users()
            .messages()
            .get(userId="me", id=msg["id"], format="full")
            .execute()
        )

        # Get subject
        subject = next(
            h["value"] for h in msg_data["payload"]["headers"] if h["name"] == "Subject"
        )

        # Get date
        date = next(
            (h["value"] for h in msg_data["payload"]["headers"] if h["name"] == "Date"),
            "Unknown date",
        )

        # print(f"\n=== Message {i} ===")
        # print(f"Subject: {subject}")
        # print(f"Date: {date}")

        # Extract body content
        payload = msg_data["payload"]
        body_data, body_type = extract_body_content(payload)
        verification_code = None
        if body_data:
            decoded_text = decode_body_data(body_data)
            if decoded_text:
                if body_type == "text/html":
                    # Clean HTML for better readability
                    clean_text = clean_html_content(decoded_text)
                    # print(f"Body (HTML converted to text):\n{clean_text}")

                    # Extract verification code
                    verification_code = extract_verification_code(clean_text)
                    if verification_code:
                        # print(f"\nVERIFICATION CODE: {verification_code}")
                        return verification_code
                    else:
                        logger.error("Body: (No verification code found in HTML)")
                else:
                    # print(f"Body ({body_type}):\n{decoded_text}")

                    # Extract verification code from plain text too
                    verification_code = extract_verification_code(decoded_text)
                    if verification_code:
                        # print(f"\nVERIFICATION CODE: {verification_code}")
                        return verification_code
                    else:
                        logger.error("Body: (No verification code found in plain text)")
            else:
                logger.error("Body: (Failed to decode)")
        else:
            logger.error("Body: (No body content found)")

        return verification_code


if __name__ == "__main__":
    get_code_from_gmail("support@optimantra.com", "verification code")
