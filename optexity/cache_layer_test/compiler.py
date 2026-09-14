import argparse
import json
import re
import time
import subprocess
import requests
import os
from datetime import datetime
import shutil

import anthropic
from optexity.schema.automation import Automation

# Tokens this script spends compiling. Accumulated because a single compile
# can make two API calls when a check fails and the retry fires.
COMPILE_TOKENS = {"input": 0, "output": 0}


def _log_paths(endpoint: str) -> tuple[str, str]:
    #actions.jsonl and server.log paths for one endpoint.
    
    run_dir = os.path.join("run_logs", endpoint)
    return os.path.join(run_dir, "actions.jsonl"), os.path.join(run_dir, "server.log")


def run_optexity_local_task(endpoint_identifier: str = "roboform") -> dict | None:
    """Start the server, run the task, and record timings.
    Returns None on failure.
    """
    log_file, server_log = _log_paths(endpoint_identifier)
    os.makedirs(os.path.dirname(server_log) or ".", exist_ok=True)
    server_out = open(server_log, "w")

    t_spawn = time.time()
    server_process = subprocess.Popen(
        ["optexity", "inference", "--port", "9000", "--child_process_id", "0"],
        stdout=server_out, stderr=subprocess.STDOUT,
        env={**os.environ, "ACTION_LOG": log_file},
    )

    print("Starting local inference server...")

    server_ready = False
    for attempt in range(15):  # Try for up to 30 seconds
        if server_process.poll() is not None:
            print("Fatal: The Optexity server process crashed during startup.")
            print(f"  see {server_log} for the traceback")
            return None

        try:
            requests.get("http://localhost:9000/", timeout=1)
            server_ready = True
            break
        except requests.exceptions.ConnectionError:
            time.sleep(0.5)

    if not server_ready:
        print("Timeout: Server failed to bind to port 9000.")
        server_process.terminate()
        return None

    print("Server is up! Sending payload...")

    with open("test_automation.json", "r") as f:
        automation_data = json.load(f)

    dynamic_input_parameters = automation_data.get("parameters", {}).get("input_parameters", {})

    payload = {
        "endpoint_name": endpoint_identifier,
        "input_parameters": dynamic_input_parameters,
        "unique_parameter_names": []
    }

    try:
        response = requests.post("http://localhost:9000/inference", json=payload)

        if response.status_code == 202:
            print("Task allocated successfully. Executing...")

            is_running = True
            while is_running:
                try:
                    status_response = requests.get("http://localhost:9000/is_task_running")
                    if status_response.status_code == 200:
                        is_running = status_response.json()
                        if is_running:
                            time.sleep(0.25)
                    else:
                        print("Server health check failed.")
                        return None
                except requests.exceptions.RequestException:
                    print("Connection to server lost.")
                    return None

            print("Task execution completed.")
            t_done = time.time()
            server_out.flush()
            phases = parse_server_phases(server_log)
            phases["total"] = t_done - t_spawn
            return phases
        else:
            print(f"Failed to allocate task. Status: {response.status_code} - {response.text}")
            return None

    finally:
        server_process.terminate()
        server_process.wait()
        server_out.close()


_TS = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3})")

# regex for token use
_TOKENS = re.compile(r"[Tt]otal[^\n]{0,30}?[Tt]okens?[^\d]{0,10}([\d,]+)")


def parse_server_phases(path: str) -> dict:
 
    nodes_start = nodes_end = None
    tokens = 0
    try:
        with open(path) as fh:
            for line in fh:
                hit = _TOKENS.search(line)
                if hit:
                    tokens += int(hit.group(1).replace(",", ""))
                stamp = _TS.match(line)
                if not stamp:
                    continue
                t = datetime.strptime(stamp.group(1), "%Y-%m-%d %H:%M:%S,%f")
                if "Running node new" in line and nodes_start is None:
                    nodes_start = t
                if "Finished node" in line:
                    nodes_end = t
    except OSError:
        return {"tokens": 0}

    out = {"tokens": tokens}
    if nodes_start and nodes_end:
        out["automation"] = (nodes_end - nodes_start).total_seconds()
    return out


def compile_with_claude_to_nodes(logs_data: list, objective: str, params: dict,
                                 _retry: bool = True, correction: str = "") -> list:
    """Passes execution logs to Claude to remove redundancies and return deterministic ActionNodes."""
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    system_prompt = """
    You are a compiler for browser automation. You receive the objective, the
    declared input_parameters, and an execution log of the actions a browser
    agent actually took.

    OPTIMIZATION RULES:
    1. NEVER invent a locator. Every 'command' must be built from the
       attributes, xpath or label of a logged action. If you cannot build one
       for a step, omit the node rather than guessing.
    2. Drop redundant steps: exploration unrelated to the objective, clicks
       the agent immediately undid, and verification attempts (extract,
       evaluate, screenshot) - those read the page, they do not change it.
       A click that navigates (cart, checkout, confirmation) is not a
       verification attempt even if nothing happens after it - keep it if
       the objective asks to reach or check that page.
    3. Keep only the last write to any given field; delete the earlier
       retries. Delete actions that clear or reset form data.
    4. Prefer durable locators: data-testid > id > name > placeholder >
       visible label > xpath. Avoid ids that look auto-generated (long hex
       runs, framework counters like mat-input-17).
    5. When the same action repeats for different items (e.g. 'Add to
       cart' on several products), name the item in prompt_instructions -
       "Click 'Add to cart' for Claw Hammer", not "Click the Add to cart
       button". A repair round is told what's already done via this text;
       a generic description can't be told apart from a different item
       using the same button, and the step gets redone.

    SCHEMA REQUIREMENTS:
    Output strictly valid JSON matching this structure. Map each logged action
    to the right node type: click -> click_element, input -> input_text,
    select_dropdown -> select_option, navigate/search -> go_to_url.
    Skip scroll, switch, close, extract, wait and done entirely.

    {
      "nodes": [
        {
          "type": "action_node",
          "end_sleep_time": 0.0,
          "interaction_action": {
            "max_tries": 3,
            "input_text": {
              "command": "locator(\"input[name=\\\"example_name\\\"]\")",
              "prompt_instructions": "Enter the value into the example field.",
              "input_text": "{example_param[0]}",
              "skip_prompt": true,
              "fill_or_type": "fill"
            }
          }
        },
        {
          "type": "action_node",
          "end_sleep_time": 0.0,
          "interaction_action": {
            "max_tries": 3,
            "click_element": {
              "command": "get_by_role(\"button\", name=\"Submit\")",
              "prompt_instructions": "Click on the Submit button.",
              "skip_prompt": true
            }
          }
        },
        {
          "type": "action_node",
          "end_sleep_time": 0.0,
          "interaction_action": {
            "max_tries": 3,
            "select_option": {
              "command": "locator(\"select[name=\\\"example\\\"]\")",
              "prompt_instructions": "Select the option from the dropdown.",
              "select_values": ["{example_param[0]}"],
              "skip_prompt": true
            }
          }
        }
      ]
    }

    LOCATOR RULES:
    - The 'command' field MUST be a valid Playwright string.
    - Use locator("...") for form inputs and complex CSS; use
      get_by_role("button", name="...") for semantic UI elements.
    - Escape nested quotes properly.
    - 'command' and 'xpath' are MUTUALLY EXCLUSIVE - set one, never both.
    - Exactly ONE action key per interaction_action.

    PARAMETERS - THIS IS NOT OPTIONAL:
    - You are given input_parameters as {"name": ["value"], ...}.
    - For EVERY input_text node you emit, check whether the logged text
      matches or closely matches one of those values. If it does, the
      "input_text" field MUST be the reference "{name[0]}" - the literal
      string with braces - and NEVER the value itself.
      Correct:   "input_text": "{user_title[0]}"
      WRONG:     "input_text": "Mr"
      WRONG:     "input_text": "Mr."
    - "Closely matches" means ignore trailing punctuation, case and
      whitespace: a logged "Mr." matches the parameter value "Mr". Emit the
      reference, do not try to reconcile the difference yourself.
    - Only emit a literal when the logged value corresponds to no parameter
      at all.
    - A record with "sensitive": true is a credential: emit the locator and
      reference the parameter, never invent a value.
    - select_option REQUIRES "select_values": a list containing the option
      to select. A select_option node without it is silently skipped at
      replay - the dropdown is never touched and nothing reports a failure.

    DOWNLOADS:
    - If a logged click triggered a file download, emit the click_element
      node with "expect_download": true. Without it the automation moves on
      before the file lands and the download is lost.
    - Signals that a click was a download: the element is an <a> whose href
      or visible text ends in a file extension (.txt, .pdf, .csv, .zip,
      .xlsx, .png), or it sits on a page whose purpose is downloading.
    - Set "download_filename" to the file's name when the log makes it
      obvious; otherwise omit it and let Optexity generate one.

    NAVIGATION:
    - A logged "navigate" or "search" action becomes a go_to_url node:
      {"type": "action_node", "end_sleep_time": 0.0,
       "interaction_action": {"go_to_url": {"url": "..."}}}
      go_to_url takes only a url - it has no command, no locator, and rule 1
      does not apply to it.

    LATENCY:
    - Set "end_sleep_time": 0.0 on every node. The schema default is 5.0, so
      ten nodes would idle for fifty seconds doing nothing.
    - Set "max_tries": 3. The default is 10 at one second each, so a stale
      locator burns ten seconds before failing.
    - Set "skip_prompt": true on every action. Without it a failed locator
      falls back to the LLM and the replay is not actually free.

    Output ONLY JSON. No markdown, no explanations.
    """

    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=8000,
        system=system_prompt,
        messages=[{"role": "user", "content":
            f"Objective:\n{objective}\n\n"
            f"input_parameters: {json.dumps(params)}\n\n"
            f"Execution log:\n{json.dumps(logs_data, indent=2)}"
            + (f"\n\n{correction}" if correction else "")}]
    )

    usage = getattr(response, "usage", None)
    if usage is not None:
        COMPILE_TOKENS["input"] += getattr(usage, "input_tokens", 0) or 0
        COMPILE_TOKENS["output"] += getattr(usage, "output_tokens", 0) or 0

    try:
        text = "".join(b.text for b in response.content if b.type == "text")
        claude_output = json.loads(text.strip().strip("`").removeprefix("json").strip())
    except json.JSONDecodeError as e:
        print(f"Claude returned invalid JSON ({e}); {len(text)} chars, tail: {text[-200:]!r}")
        if _retry:
            print("  retrying compile...")
            return compile_with_claude_to_nodes(
                logs_data, objective, params, _retry=False,
                correction=f"Your previous response was not valid JSON ({e}). "
                           "Output ONLY the JSON object, no markdown fences, "
                           "no prose before or after it.",
            )
        return []

    nodes = claude_output.get("nodes", [])

    problems = []
    leaked = check_parameterized(nodes, params)
    if leaked:
        problems.append(
            f"You emitted literal values where parameter references were "
            f"required: {leaked}. Emit the {{name[0]}} reference for each."
        )
    """
    Checking for duplicated steps manually can fail
    dupes = check_deduped(nodes)
    if dupes:
        problems.append(
            f"You emitted duplicate nodes targeting the same element: "
            f"{dupes}. Keep only the last interaction with each element and "
            f"delete the earlier retries."
        )
    """
    if problems and _retry:
        for line in leaked:
            print(f"  {line}")
        print("  retrying compile with corrections...")
        return compile_with_claude_to_nodes(
            logs_data, objective, params, _retry=False,
            correction="Your previous attempt had these problems:\n- "
                       + "\n- ".join(problems),
        )
    if problems:
        print(f"  WARNING: uncorrected after retry: {leaked}")
    return nodes


def check_parameterized(nodes: list, params: dict) -> list[str]:
    # Return the literals that should have been parameter references. 
    
    wanted = {str(v).strip().lower(): name
              for name, values in params.items() for v in values}
    leaked = []
    for node in nodes:
        action = node.get("interaction_action", {}).get("input_text")
        if not isinstance(action, dict):
            continue
        text = str(action.get("input_text", ""))
        if text.startswith("{") and text.endswith("}"):
            continue  # already a reference
        match = wanted.get(text.strip().lower())
        if match:
            leaked.append(f'"{text}" should be {{{match}[0]}}')
    return leaked


def check_deduped(nodes: list) -> list[str]:
    #Return nodes that repeat an earlier node's target.

    seen: dict[tuple, int] = {}
    dupes = []
    for i, node in enumerate(nodes):
        body = (node.get("interaction_action") or {}).get("input_text")
        if not isinstance(body, dict):
            continue
        target = body.get("command") or body.get("xpath")
        if not target:
            continue
        key = (target, str(body.get("input_text", "")))
        if key in seen:
            dupes.append(f"node {i} repeats node {seen[key]}: "
                         f"writes {body.get('input_text')!r} to {target}")
        seen[key] = i
    return dupes


def _agent_finished(logs: list) -> bool:
    #True if the trace's last 'done' action reported success.

    done = [l for l in logs if l.get("action") == "done"]
    return bool(done) and done[-1].get("params", {}).get("success", True)


def _describe_nodes(nodes: list) -> str:
    """One line per compiled node, for the remainder task's 'already done' clause.

    go_to_url nodes have no prompt_instructions field at all, and the model
    can leave it blank on the rest - falling back to url/command keeps those
    nodes in the summary instead of silently vanishing from it.
    """
    parts = []
    for node in nodes:
        for kind, body in (node.get("interaction_action") or {}).items():
            if not isinstance(body, dict):
                continue
            parts.append(
                body.get("prompt_instructions")
                or body.get("url")
                or f"{kind} {body.get('command') or body.get('xpath') or ''}".strip()
            )
    return "; ".join(p for p in parts if p)


def append_remainder(nodes: list, objective: str, covered: str) -> list:
    #If the trace didn't finish the job, leave an agentic node for the rest.

    return nodes + [{
        "type": "action_node",
        "end_sleep_time": 0.0,
        "interaction_action": {"agentic_task": {
            "task": f"The following steps are already done: {covered}. "
                    f"Continue from there and complete: {objective}",
            "max_steps": 15, "backend": "browser_use", "keep_alive": True,
        }},
    }]


def replace_agentic_node(original_automation: dict, compiled_cached_nodes: list) -> dict:
    """Replaces the slow 'agentic_task' node with the fast deterministic nodes."""
    # A cached automation still has the "agentic_task" key - serialized as
    # null - so test the value, not the key. Likewise interaction_action is
    # None on for_loop_node, if_else_node and any action_node using a
    # different action type.
    had_agentic = any(
        (node.get("interaction_action") or {}).get("agentic_task")
        for node in original_automation.get("nodes", [])
    )

    new_nodes = []
    for node in original_automation.get("nodes", []):
        if (node.get("interaction_action") or {}).get("agentic_task"):
            new_nodes.extend(compiled_cached_nodes)
        else:
            new_nodes.append(node)

    # Once iteration 0 has removed the agentic node there is nothing left to
    # replace, so without this the compiled nodes would be silently discarded
    # and every repair round would be a no-op.
    original_automation["nodes"] = new_nodes if had_agentic else compiled_cached_nodes
    return original_automation


def read_objective_and_params(automation_file: str) -> tuple[str, dict]:
    """The objective is already sitting in the agentic node's task field."""
    with open(automation_file) as f:
        source = json.load(f)
    objective = next(
        (node["interaction_action"]["agentic_task"]["task"]
         for node in source.get("nodes", [])
         if (node.get("interaction_action") or {}).get("agentic_task")),
        "",
    )
    params = source.get("parameters", {}).get("input_parameters", {})
    return objective, params


def iterative_optimization_loop(automation_file: str, endpoint: str, max_iterations: int = 5):
    """Executes the JIT-compilation loop: Run AI -> Cache -> Optimize -> Re-run Deterministically."""
    print("Starting Iteration 0: Baseline Agentic Run...")

    # Read these before the first run: after iteration 0 the agentic node is
    # gone and the objective would no longer be recoverable from the file.
    objective, params = read_objective_and_params(automation_file)
    log_file, _ = _log_paths(endpoint)
    timings = []

    for iteration in range(max_iterations):
        # One clean trace per iteration, so "did this run log anything?" is
        # a meaningful question.
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        open(log_file, "w").close()

        iteration_log = log_file.replace(".jsonl", f".{iteration}.jsonl")

        # 1. Execute the Optexity runner and check for success
        phases = run_optexity_local_task(endpoint)
        if phases is None:
            print("Execution failed. Halting optimization loop.")
            break

        shutil.copyfile(log_file, iteration_log)
        
        with open(automation_file) as f:
            source = json.load(f)
        nodes_list = source.get("nodes", [])
        # Count agentic nodes as an iteration might have both deterministic and agentic nodes
        agentic_count = sum(
            1 for n in nodes_list
            if (n.get("interaction_action") or {}).get("agentic_task")
        )
        phases.update(iteration=iteration, nodes=len(nodes_list),
                      agentic=agentic_count)
        timings.append(phases)
        summary = f"Iteration {iteration} completed in {phases['total']:.2f}s"
        if phases.get("nodes_time") is not None:
            summary += f" (automation: {phases['nodes_time']:.2f}s)"
        print(summary)

        # 2. Read the trace. JSONL - one object per line, not a JSON array.
        with open(log_file) as f:
            logs = [json.loads(line) for line in f if line.strip()]

        if not logs:
            print("No new AI logs generated. The workflow is fully cached and deterministic!")
            break

        # 3. Call the Claude Compiler
        print(f"Compiling {len(logs)} logged actions with Claude...")
        optimized_nodes = compile_with_claude_to_nodes(logs, objective, params)
        if not optimized_nodes:
            print("Optimization failed. Halting.")
            break

        if not _agent_finished(logs):
            print("  agent did not finish the objective - appending remainder node")
            optimized_nodes = append_remainder(
                optimized_nodes, objective, _describe_nodes(optimized_nodes)
            )

        # 4. Inject the optimized nodes into the main automation file
        with open(automation_file, "r") as f:
            current_automation = json.load(f)

        updated_automation_dict = replace_agentic_node(current_automation, optimized_nodes)

        # 5. Pydantic Validation Gate
        try:
            validated_automation = Automation.model_validate(updated_automation_dict)
        except Exception as e:
            print(f"Validation failed. Schema mismatch: {e}")
            break

        # 6. Overwrite the test file for the NEXT iteration
        with open(automation_file, "w") as f:
            f.write(validated_automation.model_dump_json(indent=2))

        print(f"Cache updated: {len(optimized_nodes)} deterministic nodes. "
              "Triggering next iteration to verify speedup...\n")

    print_timing_table(timings)


def print_timing_table(timings: list[dict]) -> None:
    #Print nodes,automation time, other overhead (starting the server, uploading trajectory)
    if not timings:
        return

    print(f"\n{'iter':>4} {'nodes':>6} {'agentic':>8} {'automation':>11} "
          f"{'overhead':>9} {'total':>7} {'tokens':>9}")
    for row in timings:
        auto = row.get("automation")
        overhead = row["total"] - auto if auto is not None else None
        print(f"{row['iteration']:>4} {row['nodes']:>6} {row['agentic']:>8} "
              f"{f'{auto:.1f}s' if auto else '-':>11} "
              f"{f'{overhead:.1f}s' if overhead else '-':>9} "
              f"{row['total']:>6.1f}s {row.get('tokens', 0):>9,}")

    first, last = timings[0], timings[-1]
    if len(timings) > 1 and first.get("automation") and last.get("automation"):
        a, c = first["automation"], last["automation"]
        print(f"\nautomation: {a:.1f}s -> {c:.1f}s  ({a / c:.1f}x faster)")
    if last.get("agentic"):
        print(f"note: {last['agentic']} agentic node(s) remain - not fully cached")

    compiled = COMPILE_TOKENS["input"] + COMPILE_TOKENS["output"]
    print(f"agent tokens: {sum(t.get('tokens', 0) for t in timings):,} total, "
          f"{last.get('tokens', 0):,} on the final run")
    print(f"compiler tokens: {compiled:,} (paid once per workflow)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("endpoint")
    args = parser.parse_args()
    iterative_optimization_loop(
        automation_file="test_automation.json",
        endpoint=args.endpoint,
        max_iterations=5
    )
