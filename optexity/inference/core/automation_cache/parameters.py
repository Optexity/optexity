"""Runtime-value to Automation-parameter rebinding for learned workflows."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum

from optexity.schema.automation import Parameters, SecureParameter

ParameterScalar = str | int | float | bool
GeneratedParameterScalar = ParameterScalar | None

_PARAMETER_REFERENCE_PATTERN = re.compile(
    r"\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
    r"(?P<path>\[[0-9]+\]|\.[A-Za-z_][A-Za-z0-9_]*"
    r"(?:\.[A-Za-z_][A-Za-z0-9_]*|\[[0-9]+\])*)\}"
)


class ParameterKind(str, Enum):
    INPUT = "input"
    SECURE = "secure"
    GENERATED = "generated"


@dataclass(frozen=True, slots=True)
class RuntimeParameterBinding:
    """Ephemeral concrete value mapped back to its runtime placeholder.

    ``value`` is intentionally excluded from the representation. Bindings may
    contain resolved secrets and must never be serialized into cache artifacts.
    """

    reference: str
    value: ParameterScalar = field(repr=False)
    kind: ParameterKind

    def __post_init__(self) -> None:
        if _PARAMETER_REFERENCE_PATTERN.fullmatch(self.reference) is None:
            raise ValueError(f"Invalid runtime parameter reference: {self.reference!r}")


class ParameterAllocator:
    """Choose unambiguous references and build a value-free parameter contract."""

    def __init__(
        self,
        *,
        source_input_parameters: Mapping[str, list[ParameterScalar]] | None = None,
        runtime_bindings: Iterable[RuntimeParameterBinding] | None = None,
        source_secure_parameters: Mapping[str, list[SecureParameter]] | None = None,
        source_generated_parameters: (
            Mapping[str, list[GeneratedParameterScalar]] | None
        ) = None,
        preserve_unmatched_literals: bool = False,
    ) -> None:
        self._source_input = dict(source_input_parameters or {})
        self._source_secure = dict(source_secure_parameters or {})
        self._source_generated = dict(source_generated_parameters or {})
        self._bindings = [
            RuntimeParameterBinding(
                reference=f"{{{name}[{index}]}}",
                value=value,
                kind=ParameterKind.INPUT,
            )
            for name, values in self._source_input.items()
            for index, value in enumerate(values)
        ]
        self._bindings.extend(runtime_bindings or ())
        self._preserve_unmatched_literals = preserve_unmatched_literals
        self._validate_parameter_namespaces()
        self._input_parameters: dict[str, list[ParameterScalar]] = {}
        self._secure_parameters: dict[str, list[SecureParameter]] = {}
        self._generated_parameters: dict[str, list[GeneratedParameterScalar]] = {}

    def bind(self, value: str, *, source_step_number: int, suffix: str) -> str:
        """Return the sole matching reference or a new required input parameter."""

        if _PARAMETER_REFERENCE_PATTERN.fullmatch(value) is not None:
            self._register_reference(value)
            return value

        references = {
            binding.reference
            for binding in self._bindings
            if str(binding.value) == value
        }
        if len(references) == 1:
            reference = references.pop()
            self._register_reference(reference)
            return reference
        if len(references) > 1 and self._preserve_unmatched_literals:
            raise ValueError(
                "A recorded value matches multiple runtime parameters and cannot "
                "be rebound safely"
            )
        if self._preserve_unmatched_literals:
            # A value that did not originate from any runtime placeholder is a
            # static part of the workflow (for example, selecting a fixed menu
            # option). Keeping it literal does not make memory input-specific.
            return value

        base_name = f"step_{source_step_number}_{suffix}"
        name = base_name
        collision_index = 2
        known_names = {
            *self._source_input,
            *self._source_secure,
            *self._source_generated,
            *self._input_parameters,
            *self._secure_parameters,
            *self._generated_parameters,
        }
        while name in known_names:
            name = f"{base_name}_{collision_index}"
            collision_index += 1
        self._input_parameters[name] = []
        return f"{{{name}[0]}}"

    @property
    def parameters(self) -> Parameters:
        """Return only parameter declarations used by generated nodes."""

        return Parameters(
            input_parameters=self._input_parameters,
            secure_parameters=self._secure_parameters,
            generated_parameters=self._generated_parameters,
        )

    def _register_reference(self, reference: str) -> None:
        name = parameter_name(reference)
        kind = self._kind_for_reference(reference)
        if kind == ParameterKind.INPUT:
            self._input_parameters.setdefault(name, [])
        elif kind == ParameterKind.SECURE:
            definitions = self._source_secure.get(name)
            if definitions is None:
                raise ValueError(f"Secure parameter {name!r} has no source definition")
            self._secure_parameters.setdefault(name, list(definitions))
        else:
            self._generated_parameters.setdefault(
                name,
                list(self._source_generated.get(name, [])),
            )

    def _kind_for_reference(self, reference: str) -> ParameterKind:
        explicit = {
            binding.kind for binding in self._bindings if binding.reference == reference
        }
        if len(explicit) == 1:
            return explicit.pop()
        name = parameter_name(reference)
        if name in self._source_secure:
            return ParameterKind.SECURE
        if name in self._source_generated:
            return ParameterKind.GENERATED
        return ParameterKind.INPUT

    def _validate_parameter_namespaces(self) -> None:
        namespaces = (
            set(self._source_input),
            set(self._source_secure),
            set(self._source_generated),
        )
        duplicated = (
            (namespaces[0] & namespaces[1])
            | (namespaces[0] & namespaces[2])
            | (namespaces[1] & namespaces[2])
        )
        if duplicated:
            raise ValueError(
                "Parameter names cannot be shared across input, secure and "
                f"generated namespaces: {sorted(duplicated)}"
            )


def parameter_name(reference: str) -> str:
    match = _PARAMETER_REFERENCE_PATTERN.fullmatch(reference)
    if match is None:
        raise ValueError(f"Invalid runtime parameter reference: {reference!r}")
    return match.group("name")


def is_parameter_reference(value: str) -> bool:
    return _PARAMETER_REFERENCE_PATTERN.fullmatch(value) is not None


def find_parameter_references(value: str) -> set[str]:
    """Return every indexed or generated dot-path reference in text."""

    return {match.group(0) for match in _PARAMETER_REFERENCE_PATTERN.finditer(value)}
