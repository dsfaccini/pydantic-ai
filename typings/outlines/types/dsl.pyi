from typing import Any

class JsonSchema:
    def __init__(self, schema: dict[str, Any]) -> None: ...

class Regex:
    def __init__(self, pattern: str) -> None: ...

class CFG:
    def __init__(self, definition: str) -> None: ...
