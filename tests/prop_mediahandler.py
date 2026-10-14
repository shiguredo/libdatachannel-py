"""MediaHandler のチェーン操作の property-based test

ランダムな `add_to_chain` / `set_next` の列に対して、 次の性質を検証する。

- cycle を作る操作は必ず `ValueError` になり、 チェーンは変わらない
- cycle を作らない操作は成功し、 実際のチェーンがモデルと一致する
- どの時点でも `next()` / `last()` が暴走しない (cycle が作られていない)
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from libdatachannel import MediaHandler

# 同時に扱う handler の数。 組み合わせを網羅しやすい小さな値にする
_HANDLER_COUNT: int = 4
# 1 つの例で実行する操作数
_MAX_OPERATIONS: int = 8


def _chain_indices(model: dict[int, int | None], start: int) -> list[int]:
    """start から next() をたどったノード列 (cycle があれば打ち切る)"""
    nodes: list[int] = []
    seen: set[int] = set()
    current: int | None = start
    while current is not None and current not in seen:
        seen.add(current)
        nodes.append(current)
        current = model[current]
    return nodes


def _tail_index(model: dict[int, int | None], start: int) -> int:
    return _chain_indices(model, start)[-1]


def _apply(model: dict[int, int | None], kind: str, index: int, handler_index: int) -> None:
    if kind == "add":
        # add_to_chain は chain の末尾に繋ぐ
        model[_tail_index(model, index)] = handler_index
    else:
        model[index] = handler_index


def _has_cycle_from(model: dict[int, int | None], start: int) -> bool:
    """start から next() をたどって cycle に到達するか"""
    seen: set[int] = set()
    current: int | None = start
    while current is not None:
        if current in seen:
            return True
        seen.add(current)
        current = model[current]
    return False


def _actual_chain(head: MediaHandler) -> list[MediaHandler]:
    """実際の handler から next() をたどったノード列"""
    nodes: list[MediaHandler] = []
    seen: set[int] = set()
    current: MediaHandler | None = head
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        nodes.append(current)
        current = current.next()
    return nodes


# CI の速度ゆらぎで deadline 超過の偽陽性を出さないようにする
@settings(max_examples=200, deadline=None)
@given(
    st.lists(
        st.tuples(
            st.sampled_from(["add", "set"]),
            st.integers(min_value=0, max_value=_HANDLER_COUNT - 1),
            st.integers(min_value=0, max_value=_HANDLER_COUNT - 1),
        ),
        max_size=_MAX_OPERATIONS,
    )
)
def test_mediahandler_chain_operations(
    operations: list[tuple[str, int, int]],
) -> None:
    """cycle になる操作だけが拒否され、 チェーンがモデルと一致し続けること"""
    handlers = [MediaHandler() for _ in range(_HANDLER_COUNT)]
    # handler の index -> 次の handler の index (None は終端)
    model: dict[int, int | None] = dict.fromkeys(range(_HANDLER_COUNT))

    for kind, index, handler_index in operations:
        candidate = dict(model)
        _apply(candidate, kind, index, handler_index)
        would_cycle = _has_cycle_from(candidate, index)

        if would_cycle:
            with pytest.raises(ValueError):
                if kind == "add":
                    handlers[index].add_to_chain(handlers[handler_index])
                else:
                    handlers[index].set_next(handlers[handler_index])
            # 拒否された操作はモデルにも反映しない
            assert not _has_cycle_from(model, index)
        else:
            if kind == "add":
                handlers[index].add_to_chain(handlers[handler_index])
            else:
                handlers[index].set_next(handlers[handler_index])
            model = candidate

        # 実際のチェーンがモデルと一致すること
        expected = [handlers[i] for i in _chain_indices(model, index)]
        assert _actual_chain(handlers[index]) == expected
        # 末尾が取得できること (cycle があればここで暴走する)
        assert handlers[index].last() is expected[-1]
