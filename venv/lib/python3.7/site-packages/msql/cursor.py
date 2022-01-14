from typing import Any, List, Optional, Dict, Union
from typing_extensions import Protocol
"""
This is not entirely true. What we're actually returning is dict-like object
which supports accessing both by index and by column name.
"""
QueryResult = Dict[Union[int, str], Any]


class Cursor(Protocol):

    def execute(self, statement: str, *args: Any, **kwargs: Any) -> None:
        ...

    def executemany(self, statement: str, *args: Any, **kwargs: Any) -> None:
        ...

    def fetchone(self) -> Optional[QueryResult]:
        ...

    def fetchall(self) -> List[QueryResult]:
        ...
