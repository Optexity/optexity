from optexity.schema.automation import Automation

description = "Extract stock price from StockAnalysis"
endpoint_name = "extract_price_stockanalysis"
automation_json = {
    "url": "https://stockanalysis.com/",
    "parameters": {
        "input_parameters": {"stock_ticker": ["AAPL"]},
        "generated_parameters": {},
    },
    "nodes": [
        {
            "interaction_action": {
                "input_text": {
                    "command": 'locator("#search-header")',
                    "prompt_instructions": "Fill the input field with ID 'search-header' with the value of the 'stock_ticker' variable.",
                    "input_text": "{stock_ticker[0]}",
                }
            }
        },
        {
            "interaction_action": {
                "click_element": {
                    "prompt_instructions": "Click on the link with the name of the stock equivalent for {stock_ticker[0]}."
                }
            }
        },
        {
            "extraction_action": {
                "llm": {
                    "source": ["screenshot"],
                    "extraction_format": {
                        "stock_name": "str",
                        "stock_price": "str",
                        "stock_symbol": "str",
                    },
                    "extraction_instructions": "Extract the stock price, stock name, and stock symbol from the webpage.",
                }
            }
        },
    ],
}

automation = Automation.model_validate(automation_json)
