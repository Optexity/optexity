import asyncio
import logging
import os

from dotenv import load_dotenv

from optexity.examples.i94 import i94_test
from optexity.examples.pshpgeorgia_medicaid import (
    pshpgeorgia_login_test,
    pshpgeorgia_medicaid_test,
)
from optexity.examples.supabase_login import supabase_login_test
from optexity.inference.core.run_automation import run_automation
from optexity.inference.infra.browser import Browser
from optexity.schema.memory import Memory, Variables

load_dotenv()


logger = logging.getLogger(__name__)


async def run_supabase_login_test():
    logger.debug("Starting Supabase login test")
    browser = Browser()
    memory = Memory(
        variables=Variables(
            input_variables={
                "username": ["test@test.com"],
                "password": ["password"],
            }
        )
    )

    await browser.start()
    logger.info("Browser started")
    logger.info("Navigating to Supabase")
    await browser.go_to_url("https://supabase.com")
    logger.info("Navigated to Supabase")
    logger.info("Sleeping for 5 seconds")
    await asyncio.sleep(2)

    logger.info("Running automation")
    await run_automation(supabase_login_test, memory, browser)
    logger.info("Automation finished")
    await asyncio.sleep(5)

    await browser.stop()


async def run_pshpgeorgia_test():
    try:
        logger.debug("Starting PSHP Georgia test")
        browser = Browser()
        memory = Memory(
            variables=Variables(
                input_variables={
                    "username": [os.environ.get("USERNAME")],
                    "password": [os.environ.get("PASSWORD")],
                    "plan_type": [os.environ.get("PLAN_TYPE")],
                    "member_id": [os.environ.get("MEMBER_ID")],
                    "dob": [os.environ.get("DOB")],
                }
            )
        )

        await browser.start()
        logger.debug("Browser started")
        logger.debug("Navigating to PSHP Georgia")
        await browser.go_to_url(
            "https://sso.entrykeyid.com/as/authorization.oauth2?response_type=code&client_id=f6a6219c-be42-421b-b86c-e4fc509e2e87&scope=openid%20profile&state=_igWklSsnrkO5DQfjBMMuN41ksMJePZQ_SM_61wTJlA%3D&redirect_uri=https://provider.pshpgeorgia.com/careconnect/login/oauth2/code/pingcloud&code_challenge_method=S256&nonce=xG41TJjco_x7Vs_MQgcS3bw5njLiJsXCqvO-V8THmY0&code_challenge=ZTaVHaZCNFTejXNJo51RlJ3Kv9dH0tMODPTqO7hiP3A&app_origin=https://provider.pshpgeorgia.com/careconnect/login/oauth2/code/pingcloud&brand=pshpgeorgia"
        )
        logger.debug("Navigated to PSHP Georgia")

        logger.debug("Running login test")
        await run_automation(pshpgeorgia_login_test, memory, browser)
        logger.debug("Login test finished")

        logger.debug("Running Medicaid test")
        await run_automation(pshpgeorgia_medicaid_test, memory, browser)
        logger.debug("Medicaid test finished")

        await asyncio.sleep(5)
        await browser.stop()
    except Exception as e:
        logger.error(f"Error running PSHP Georgia test: {e}")
        raise e
    finally:
        await browser.stop()


async def run_i94_test():
    try:
        logger.debug("Starting I-94 test")
        browser = Browser()
        memory = Memory(
            variables=Variables(
                input_variables={
                    "last_name": [os.environ.get("LAST_NAME")],
                    "first_name": [os.environ.get("FIRST_NAME")],
                    "nationality": [os.environ.get("NATIONALITY")],
                    "date_of_birth": [os.environ.get("DATE_OF_BIRTH")],
                    "document_number": [os.environ.get("DOCUMENT_NUMBER")],
                }
            )
        )

        await browser.start()
        logger.debug("Browser started")
        logger.debug("Navigating to I-94")
        await browser.go_to_url(i94_test.url)
        logger.debug("Navigated to I-94")

        logger.debug("Running I-94 test")
        await asyncio.sleep(5)
        await run_automation(i94_test, memory, browser)
        logger.debug("I-94 test finished")

        await asyncio.sleep(5)
        await browser.stop()
    except Exception as e:
        logger.error(f"Error running I-94 test: {e}")
        raise e
    finally:
        await browser.stop()


if __name__ == "__main__":
    # asyncio.run(run_supabase_login_test())
    # asyncio.run(run_pshpgeorgia_test())
    asyncio.run(run_i94_test())
