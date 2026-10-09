# MediaHandler chain に cycle を作ると即 SEGV する

- Priority: High
- Created: 2026-08-30
- Completed: 2026-10-09
- Branch: feature/fix-mediahandler-chain-cycle-segv
- Polished: 2026-10-09

## 目的

add_to_chain / set_next は cycle を阻止しないため、 `h.add_to_chain(h)` や `h2.add_to_chain(h1)` という 1 行の誤用で cycle ができ、 その後の `MediaHandler::last()` が next() を無限に再帰してプロセスが stack overflow (SIGSEGV) で落ちる。 さらに shared_ptr の cycle としてリークも残る。 binding 側で cycle を検出して例外にする。

## 優先度根拠

- 実測: cycle を作る操作自体は成功し、 その後の `last()` で SIGSEGV する (exit 139)。 自己参照・相互参照のどちらも再現する
- tests/test_mediahandler.py は一方向の chain しか試しておらず、この誤用が未検証
- MediaHandler chain は映像送信の標準的な構成要素であり、誤用時に落ちるのは debug 困難

## 現状

再現手順:

```python
from libdatachannel import MediaHandler

h1 = MediaHandler()
h2 = MediaHandler()
h1.add_to_chain(h2)
h2.add_to_chain(h1)  # ここでは落ちず、 cycle ができる
h2.last()  # 無限再帰 → SIGSEGV (exit 139)
```

自己参照 (`h.add_to_chain(h)` / `h.set_next(h)`) も同様に、 作った後の `last()` で落ちる。 `addToChain` は `last()->setNext(handler)` の順で動くため、 cycle 生成時点の走査は終端に届いてしまう。

- `bind_mediahandler` の add_to_chain / set_next は `shared_ptr<MediaHandler>` を制約なしで受け取る
- libdatachannel の `MediaHandler::last()` (`src/mediahandler.cpp`) は `next()` の再帰で終端を探すため、cycle があると戻らない
- `MediaHandler::mNext` は shared_ptr 保持のため、cycle はメモリリークとしても残る

## 設計方針

- binding 側の add_to_chain / set_next で cycle を検出し、 例外 (`std::invalid_argument` → Python の `ValueError`) を投げる
  - `add_to_chain` が張る辺は `last(self) -> handler` なので、 条件は「self から到達可能なノードの集合」と「handler から到達可能なノードの集合」が交わること。 同じ handler を 2 回追加する場合もこれで検出できる
  - `set_next` は置換なので、 条件は「handler から self に到達可能なこと」
  - 検査は `addToChain` を呼ぶ前に行う (内部で先に `last()` が走るため)
- 走査には上限 (1024 ノード) を設ける。 入力チェーンの長さが上限を超えた場合は「長すぎる」として例外にする (既に cycle がある場合に検査自体が落ちないようにするため)。 上限を超えた場合は cycle とは別のメッセージにする
- 上限は入力チェーンの走査に対する制限であり、 連結後のチェーン長を制限するものではない (1024 ノード同士を連結すると 2048 ノードになり得る)
- `Track.chain_media_handler` も `Track::chainMediaHandler` 経由で `addToChain` を呼ぶため、 同じ検査を入れる (media handler 未設定時は置換のみなので検査しない)
- cycle 検出の性質は property-based test (`tests/prop_mediahandler.py`) で検証する (ランダムな連結操作列に対して「cycle になる操作だけが拒否され、 チェーンがモデルと一致する」ことを確かめる)
- cycle 検出のテストを追加する (自己参照 / 相互参照 / 自分のチェーン途中の handler / 同じ handler の 2 回追加 / 例外後にチェーンが壊れていないこと)

## 完了条件

- cycle を作ろうとすると `ValueError` になり、 その後に `last()` を呼んでも SEGV しないこと
- 上限 (1024 ノード) ちょうどのチェーンへは接続でき、 上限を超えるチェーンへの接続は「長すぎる」旨の `ValueError` になること
- 例外になった後もチェーンが壊れていないこと (`next()` / `last()` が元の値を返す)
- cycle 検出のテストが追加されていること
- `prek run --all-files pytest` (prek.toml の pytest フック = 既知の恒停テストを `--deselect` で除外) が PASS する
- CI (wheel.yml の leg / prek.yml の `ty` ジョブ) が PASS する
- `CHANGES.md` の `## develop` に `[FIX]` エントリが追加されている
- `/review-diff-code` の致命的 / 重要指摘が 0 件であること

## 解決方法

- `src/bind_libdatachannel.cpp` に MediaHandler のチェーン検査を追加した
  - `collect_media_handler_chain` で `next()` を最大 1024 ノードたどり、 終端に到達しない場合は長さ超過として扱う (検査自体が無限再帰しないようにするため)
  - `add_to_chain` は `last(self) -> handler` の辺を張るため、 両チェーンのノードが交われば cycle
  - `set_next` は置換のため、 handler のチェーンに self が含まれれば cycle
  - `throw_media_handler_chain_error` で cycle と長さ超過を区別した `std::invalid_argument` (Python の `ValueError`) を投げる
  - `add_to_chain` / `set_next` / `Track.chain_media_handler` の binding で、 連結する前に検査する
- テスト
  - `tests/test_mediahandler.py` に issue の再現手順 (相互参照)、 Track 経路、 上限ちょうどの成功と上限超過の例外を追加した
  - `tests/prop_mediahandler.py` を追加し、 ランダムな連結操作列に対して「cycle になる操作だけが拒否され、 チェーンがモデルと一致する」ことを検証する (hypothesis)
  - `pyproject.toml` の `python_files` に `prop_*.py` を追加した
- `CHANGES.md` の `## develop` に `[FIX]` エントリを追加した
- 実測: 修正前は cycle を作った後の `last()` で exit 139 (SIGSEGV)。 修正後は mediahandler の 7 テストが PASS、 全体で 89 passed / 12 skipped / 1 deselected

## 参考

- 対象シンボル: `bind_mediahandler` 内の add_to_chain / set_next (src/bind_libdatachannel.cpp)
- libdatachannel v0.24.0: `src/mediahandler.cpp` (`MediaHandler::addToChain` は `last()->setNext(handler)`、 `last()` は `next()` の再帰)、 `src/track.cpp` (`Track::chainMediaHandler` は先頭の media handler へ `addToChain`)
- 対象外: `PeerConnection.set_media_handler` は置換のみ、 `reset_callbacks` は `mNext` に触れないため cycle 経路ではない
- 対象外: `set_next` を繰り返して線形チェーンを極端に長くする (10 万ノード) と、 `last()` の再帰の深さで SEGV し得る。 cycle ではないため本 issue では扱わない (現実的な構成ではない)
