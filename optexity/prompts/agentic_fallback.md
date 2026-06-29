# Agentic Fallback

A step in an automated browser workflow failed and has been handed to you to
complete on the live page. Look at the page, act, then stop.

## Information you have

- Error and recent run log: <<ERROR_LOGS>>
- Input parameters for this run (use these values as-is; don't invent any):
  <<INPUT_PARAMETERS>>
- Current page: <<CURRENT_URL>>
- Surrounding steps — `[already ran]` came before you (with their values),
  `>> CURRENT <<` is yours, later steps are context only, don't do them:
  <<WORKFLOW_WINDOW>>

## The step to perform

<<GOAL>>

## How to handle it

- If this step is failing because an earlier step didn't do what it should have,
  fix that first, then do this step.
- Using all of the above, perform the step.
- If it's a genuine failure that cannot be fixed from here, don't force it — leave
  it and report the failure with the reason.
