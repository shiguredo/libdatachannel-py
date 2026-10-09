"""Trickle ICE の SDP fragment を組み立てるヘルパー

WHIP (RFC 9725) / WHEP (draft-ietf-wish-whep) の PATCH リクエストで使う
`application/trickle-ice-sdpfrag` の body を組み立てる。

- RFC 9725 Section 4.3: 201 Created を受信するまで candidate をバッファし、 受信後に
  バッファした candidate を 1 つの HTTP PATCH でまとめて送る (SHOULD)。 PATCH body の
  組み立ては RFC 8840 Section 4.4 に従う
- RFC 9725 Section 4.3: bundle ポリシーでは offerer-tagged の `m=` 行のみを含める
"""


def normalize_candidate(candidate: str) -> str:
    """candidate を SDP の属性行 (`a=candidate:...`) の形にする"""
    if candidate.startswith("a="):
        return candidate
    return f"a={candidate}"


def build_sdp_fragment(
    local_sdp: str, candidates: list[str], end_of_candidates: bool = True
) -> str:
    """SDP fragment を組み立てる (RFC 9725 Section 4.3 / RFC 8840)

    local_sdp は libdatachannel が生成した SDP、 candidates は on_local_candidate で
    得た candidate の一覧。 最初の `m=` セクションの ICE 資格情報と candidate だけを
    含む fragment を返す。 end_of_candidates が偽の場合は `a=end-of-candidates` を
    付けない (candidate の収集が終わっていない場合は付けてはいけない)。
    """
    lines = [line.strip() for line in local_sdp.replace("\r\n", "\n").split("\n")]

    fragment: list[str] = []
    # セッションレベルの BUNDLE グループ (RFC 9725 Section 4.3 の例に合わせる)
    fragment.extend(line for line in lines if line.startswith("a=group:BUNDLE"))

    # bundle ポリシーでは offerer-tagged の m= 行のみを含める
    media_index = next((i for i, line in enumerate(lines) if line.startswith("m=")), None)
    if media_index is None:
        raise ValueError("SDP に m= 行がない")
    fragment.append(lines[media_index])

    # m= セクションの属性から mid と ICE 資格情報を取り出す
    mid = None
    ufrag = None
    pwd = None
    for line in lines[media_index + 1 :]:
        if line.startswith("m="):
            break
        if line.startswith("a=mid:") and mid is None:
            mid = line
        elif line.startswith("a=ice-ufrag:") and ufrag is None:
            ufrag = line
        elif line.startswith("a=ice-pwd:") and pwd is None:
            pwd = line

    # ICE 資格情報がセッションレベルにある SDP にも対応する
    if ufrag is None:
        ufrag = next((line for line in lines if line.startswith("a=ice-ufrag:")), None)
    if pwd is None:
        pwd = next((line for line in lines if line.startswith("a=ice-pwd:")), None)
    if ufrag is None or pwd is None:
        raise ValueError("SDP に ICE 資格情報 (a=ice-ufrag / a=ice-pwd) がない")

    if mid is not None:
        fragment.append(mid)
    fragment.append(ufrag)
    fragment.append(pwd)
    fragment.extend(normalize_candidate(candidate) for candidate in candidates)
    if end_of_candidates:
        fragment.append("a=end-of-candidates")

    return "\r\n".join(fragment) + "\r\n"
