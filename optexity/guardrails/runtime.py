import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse, urlunparse

from optexity.guardrails.audit import GuardrailAuditLogger
from optexity.guardrails.exceptions import GuardrailViolation
from optexity.guardrails.models import GuardrailPolicy
from optexity.guardrails.prompt_injection import detect_prompt_injection


def _normalize_domain(value: str) -> str:
    parsed = urlparse(value if "://" in value else f"https://{value}")
    return (parsed.hostname or "").lower().strip(".")


class GuardrailRuntime:
    """Stateful policy evaluator scoped to exactly one task execution."""

    def __init__(
        self,
        *,
        task_id: str,
        policy: GuardrailPolicy,
        start_url: str,
        input_parameters: dict[str, list[Any]] | None = None,
        task_directory: Path | None = None,
    ):
        self.task_id = task_id
        self.policy = policy
        self.start_url = start_url
        start_domain = _normalize_domain(start_url)
        configured = {_normalize_domain(d) for d in policy.allowed_domains}
        self.allowed_domains = {d for d in configured if d} or {start_domain}
        self.sensitive_domains = {
            _normalize_domain(d)
            for d in policy.data_protection.sensitive_data_domains
            if _normalize_domain(d)
        } or set(self.allowed_domains)
        self.temporary_domains: set[str] = set()
        self.counters: Counter[str] = Counter()
        self.audit = GuardrailAuditLogger(
            task_id,
            (
                task_directory / "logs" / "guardrails.jsonl"
                if task_directory is not None
                else None
            ),
        )
        self.sensitive_values = self._collect_sensitive_values(input_parameters or {})
        roots = [Path(p).expanduser().resolve() for p in policy.allowed_upload_roots]
        if task_directory is not None:
            roots.append(task_directory.resolve())
        self.allowed_upload_roots = roots

    @classmethod
    def from_task(cls, task) -> "GuardrailRuntime":
        policy = task.automation.guardrails
        runtime = cls(
            task_id=str(task.task_id),
            policy=policy,
            start_url=task.automation.url,
            input_parameters=task.input_parameters,
            task_directory=task.task_directory,
        )
        for value in (
            getattr(task, "api_key", None),
            getattr(task, "task_callback_api_key", None),
        ):
            if value:
                runtime.register_sensitive_value(value)
        callback = getattr(task, "callback_url", None)
        if callback is not None:
            for field in ("api_key", "password"):
                value = getattr(callback, field, None)
                if value:
                    runtime.register_sensitive_value(value)
        for values in getattr(task, "secure_parameters", {}).values():
            for secure_parameter in values:
                runtime._register_sensitive_model_fields(secure_parameter)
        return runtime

    def _register_sensitive_model_fields(self, value: Any) -> None:
        if hasattr(value, "model_dump"):
            value = value.model_dump(exclude_none=True)
        if isinstance(value, dict):
            for key, item in value.items():
                if re.search(r"(?:secret|password|token|api[_-]?key)", key, re.I):
                    if not isinstance(item, (dict, list)):
                        self.register_sensitive_value(item)
                self._register_sensitive_model_fields(item)
        elif isinstance(value, list):
            for item in value:
                self._register_sensitive_model_fields(item)

    def _collect_sensitive_values(self, parameters: dict[str, list[Any]]) -> set[str]:
        patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.policy.data_protection.sensitive_parameter_patterns
        ]
        values: set[str] = set()
        for key, items in parameters.items():
            if not any(pattern.search(key) for pattern in patterns):
                continue
            for item in items if isinstance(items, list) else []:
                value = str(item)
                if value:
                    values.add(value)
        return values

    def register_sensitive_value(self, value: Any) -> None:
        text = str(value)
        if text:
            self.sensitive_values.add(text)

    def temporarily_allow_domain(self, domain: str) -> None:
        normalized = _normalize_domain(domain)
        if normalized:
            self.temporary_domains.add(normalized)

    def remove_temporary_domain(self, domain: str) -> None:
        self.temporary_domains.discard(_normalize_domain(domain))

    def _matches_domain(self, hostname: str, allowed: Iterable[str]) -> bool:
        hostname = hostname.lower().strip(".")
        for domain in allowed:
            if hostname == domain:
                return True
            if self.policy.allow_subdomains and hostname.endswith(f".{domain}"):
                return True
        return False

    def is_url_allowed(self, url: str, *, sensitive: bool = False) -> bool:
        if url in {"about:blank", ""}:
            return True
        parsed = urlparse(url)
        if parsed.scheme not in self.policy.allowed_schemes or not parsed.hostname:
            return False
        domains = self.sensitive_domains if sensitive else self.allowed_domains
        return self._matches_domain(
            parsed.hostname, set(domains) | self.temporary_domains
        )

    def _decision(self, denied: bool) -> str:
        if denied and self.policy.mode == "enforce":
            return "deny"
        if denied:
            return "audit"
        return "allow"

    def _enforce(self, *, denied: bool, code: str, message: str, **event) -> None:
        decision = self._decision(denied)
        for url_key in ("current_url", "target_url"):
            if event.get(url_key):
                event[url_key] = self._safe_audit_url(event[url_key])
        if event.get("details"):
            event["details"] = self.redact_value(event["details"])
        self.audit.emit(decision=decision, code=code, **event)
        if decision == "deny":
            raise GuardrailViolation(code, message)

    def authorize_action(
        self,
        action: str,
        *,
        source: str,
        current_url: str | None = None,
        target_url: str | None = None,
        data: Any = None,
    ) -> None:
        if not self.policy.enabled:
            return
        action = action.lower()
        blocked = action in self.policy.blocked_actions
        not_allowed = (
            bool(self.policy.allowed_actions)
            and action not in self.policy.allowed_actions
        )
        if blocked or not_allowed:
            self._enforce(
                denied=True,
                code="action_not_allowed",
                message=f"Action {action!r} is not allowed by this workflow",
                event_type="action",
                action=action,
                source=source,
                current_url=current_url,
                target_url=target_url,
            )
            return

        if (
            current_url
            and current_url != "about:blank"
            and not self.is_url_allowed(current_url)
        ):
            self._enforce(
                denied=True,
                code="current_domain_not_allowed",
                message=f"Current URL is outside the workflow domain policy: {current_url}",
                event_type="domain",
                action=action,
                source=source,
                current_url=current_url,
                target_url=target_url,
            )
            return
        if target_url and not self.is_url_allowed(target_url):
            self._enforce(
                denied=True,
                code="target_domain_not_allowed",
                message=f"Target URL is outside the workflow domain policy: {target_url}",
                event_type="domain",
                action=action,
                source=source,
                current_url=current_url,
                target_url=target_url,
            )
            return

        if (
            data is not None
            and target_url
            and self.contains_sensitive_data(data)
            and self.policy.data_protection.block_sensitive_data_cross_domain
            and not self.is_url_allowed(target_url, sensitive=True)
        ):
            self._enforce(
                denied=True,
                code="sensitive_data_destination_not_allowed",
                message="Sensitive data cannot be sent to this destination",
                event_type="data_flow",
                action=action,
                source=source,
                current_url=current_url,
                target_url=target_url,
            )
            return

        counter = {
            "api_call": "api_calls",
            "upload": "uploads",
            "download": "downloads",
        }.get(action)
        if counter:
            self.consume(counter)
        self._enforce(
            denied=False,
            code="policy_allow",
            message="allowed",
            event_type="action",
            action=action,
            source=source,
            current_url=current_url,
            target_url=target_url,
        )

    def authorize_file_path(
        self,
        path: str | Path,
        *,
        action: str = "read_file",
        source: str = "workflow",
    ) -> None:
        resolved = Path(path).expanduser().resolve()
        allowed = any(
            resolved == root or root in resolved.parents
            for root in self.allowed_upload_roots
        )
        self._enforce(
            denied=not allowed,
            code="file_path_allow" if allowed else "file_path_not_allowed",
            message=f"File path is outside allowed roots: {resolved}",
            event_type="filesystem",
            action=action,
            source=source,
            details={"path": str(resolved)},
        )

    def authorize_upload_path(self, path: str | Path) -> None:
        self.authorize_file_path(path, action="upload", source="workflow")

    def authorize_navigation_request(self, url: str) -> None:
        """Network-layer same-origin check used before a document is fetched.

        This is the accounting point for navigation so redirects and
        click-triggered navigations are counted as well as explicit goto calls.
        """
        if not self.policy.enabled:
            return
        allowed = self.is_url_allowed(url)
        if allowed:
            self.consume("navigations")
        self._enforce(
            denied=not allowed,
            code=(
                "navigation_request_allow" if allowed else "navigation_request_blocked"
            ),
            message=f"Navigation request is outside the workflow domain policy: {url}",
            event_type="network_navigation",
            action="navigate",
            source="browser_network",
            target_url=url,
        )

    def consume(self, resource: str, amount: int = 1) -> None:
        if not self.policy.enabled:
            return
        if amount < 0:
            raise ValueError("Guardrail resource consumption cannot be negative")
        limits = {
            "llm_calls": self.policy.limits.max_llm_calls,
            "api_calls": self.policy.limits.max_api_calls,
            "agentic_steps": self.policy.limits.max_agentic_steps,
            "navigations": self.policy.limits.max_navigations,
            "tabs": self.policy.limits.max_tabs,
            "uploads": self.policy.limits.max_uploads,
            "downloads": self.policy.limits.max_downloads,
        }
        limit = limits[resource]
        attempted = self.counters[resource] + amount
        denied = attempted > limit
        self._enforce(
            denied=denied,
            code="resource_limit_exceeded" if denied else "resource_consumed",
            message=f"Guardrail resource limit exceeded: {resource} ({attempted}/{limit})",
            event_type="resource",
            source="runtime",
            details={"resource": resource, "value": attempted, "limit": limit},
        )
        if not denied or self.policy.mode == "audit":
            self.counters[resource] = attempted

    def contains_sensitive_data(self, value: Any) -> bool:
        text = str(value)
        return any(secret and secret in text for secret in self.sensitive_values)

    def _safe_audit_url(self, url: str) -> str:
        """Remove credentials, query strings, fragments, and known secrets."""
        parsed = urlparse(str(url))
        hostname = parsed.hostname or ""
        netloc = hostname
        if parsed.port is not None:
            netloc = f"{hostname}:{parsed.port}"
        safe = urlunparse((parsed.scheme, netloc, parsed.path, "", "", ""))
        return self.redact_text(safe)

    def redact_text(self, text: str) -> str:
        if not self.policy.data_protection.redact_secrets_from_llm:
            return text
        redacted = text
        for secret in sorted(self.sensitive_values, key=len, reverse=True):
            redacted = redacted.replace(secret, "[REDACTED_SECRET]")
        return redacted

    def redact_value(self, value: Any) -> Any:
        """Recursively redact known secrets without changing container shape."""
        if isinstance(value, str):
            return self.redact_text(value)
        if isinstance(value, dict):
            redacted = {}
            patterns = self.policy.data_protection.sensitive_parameter_patterns
            for key, item in value.items():
                if any(re.search(pattern, str(key), re.I) for pattern in patterns):
                    redacted[key] = "[REDACTED_SENSITIVE_FIELD]"
                else:
                    redacted[key] = self.redact_value(item)
            return redacted
        if isinstance(value, list):
            return [self.redact_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self.redact_value(item) for item in value)
        return value

    def inspect_untrusted_text(self, text: str, *, source: str) -> str:
        policy = self.policy.prompt_injection
        if not self.policy.enabled or not policy.enabled:
            return self.redact_text(text)
        bounded = text[: policy.max_untrusted_text_chars]
        matches = detect_prompt_injection(bounded, policy.additional_patterns)
        if matches:
            details = {"patterns": [match.pattern_name for match in matches]}
            if policy.action == "block":
                self._enforce(
                    denied=True,
                    code="prompt_injection_detected",
                    message="Potential prompt injection detected in untrusted browser content",
                    event_type="prompt_injection",
                    source=source,
                    details=details,
                )
            else:
                self.audit.emit(
                    decision="audit",
                    code="prompt_injection_detected",
                    event_type="prompt_injection",
                    source=source,
                    details=details,
                )
            if policy.action == "redact":
                for match in matches:
                    bounded = bounded.replace(
                        match.excerpt, "[REDACTED_UNTRUSTED_INSTRUCTION]"
                    )
        return self.redact_text(bounded)

    def browser_use_excluded_actions(self) -> list[str]:
        mapping = {
            "search": "navigate",
            "navigate": "navigate",
            "go_back": "go_back",
            "wait": "sleep",
            "click": "click",
            "input": "input",
            "upload_file": "upload",
            "switch": "switch_tab",
            "close": "close_tab",
            "extract": "extract",
            "scroll": "scroll",
            "send_keys": "key_press",
            "find_text": "extract",
            "dropdown_options": "select",
            "select_dropdown": "select",
            "evaluate": "evaluate",
            "write_file": "write_file",
            "read_file": "read_file",
            "replace_file": "replace_file",
        }
        excluded = {
            tool
            for tool, action in mapping.items()
            if action in self.policy.blocked_actions
            or (
                self.policy.allowed_actions
                and action not in self.policy.allowed_actions
            )
        }
        # Opaque browser-use tool calls do not expose their path before
        # execution. Typed UploadFileAction nodes remain available because the
        # runtime can authorize their path deterministically.
        excluded.add("upload_file")
        # A model-triggered screenshot is another binary disclosure surface.
        # The serialized-message gateway also strips image data, but removing
        # the tool prevents needless capture in the first place.
        if not self.policy.data_protection.allow_screenshots_to_llm:
            excluded.add("screenshot")
        return sorted(excluded)
