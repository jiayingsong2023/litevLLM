# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
    A decorator that ensures all fields in a dataclass have default values
    and that each field has a docstring.

    If a `ConfigT` is used as a CLI argument itself, the `type` keyword argument
    provided by `get_kwargs` will be
    `pydantic.TypeAdapter(ConfigT).validate_json(cli_arg)` which treats the
    `cli_arg` as a JSON string which gets validated by `pydantic`.

    Config validation is performed by the tools/pre_commit/validate_config.py
    script, which is invoked during the pre-commit checks.
    A helper function that retrieves an attribute from an object which may
    have multiple possible names. This is useful when fetching attributes from
    arbitrary `transformers.PretrainedConfig` instances.

    In the case where the first name in `names` is the preferred name, and
    any other names are deprecated aliases, setting `warn=True` will log a
    warning when a deprecated name is used.
    Check if the text looks like a printed Python object, e.g.
    contains any substring matching the pattern: "at 0xFFFFFFF>"
    We match against 0x followed by 2-16 hex chars (there's
    a max of 16 on a 64-bit system).

    Args:
        text (str): The text to check

    Returns:
        result (bool): `True` if a match is found, `False` otherwise.
    Get any docstrings placed after attribute assignments in a class body.

    https://davidism.com/mit-license/
    Order: primitives, special types (Enum, callable, torch.dtype, Path), then
    generic containers (Mapping/Set/Sequence) with recursion.
    - Includes all dataclass fields not in `ignored_factors`.
    - Errors on non-normalizable values.
    return hashlib.sha256(json.dumps(items, sort_keys=True).encode()).hexdigest()

def handle_deprecated(
    config: ConfigT,
    old_name: str,
    new_name_or_names: str | list[str],
    removal_version: str,
) -> None:
    old_val = getattr(config, old_name)
    if old_val is None:
        return

    if isinstance(new_name_or_names, str):
        new_names = [new_name_or_names]
    else:
        new_names = new_name_or_names

    msg = (
        f"{old_name} is deprecated and will be removed in {removal_version}. "
        f"Use {', '.join(new_names)} instead."
    )
    logger.warning(msg)

    for new_name in new_names:
        setattr(config, new_name, old_val)

@dataclass
class Range:

    start: int
    end: int

    def is_single_size(self) -> bool:
        return self.start == self.end

    def __contains__(self, size: int) -> bool:
        # Inclusive of start, inclusive of end
        return self.start <= size <= self.end

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Range):
            return False
        return self.start == other.start and self.end == other.end

    def __hash__(self) -> int:
        return hash((self.start, self.end))

    def __str__(self) -> str:
        return f"({self.start}, {self.end})"

    def __repr__(self) -> str:
        return self.__str__()
