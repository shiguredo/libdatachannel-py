# 生成された型スタブに、 __init__.py が定義するエイリアスを追記する
#
# nanobind_add_stub が生成するのは extension module のスタブで、 __init__.py の
# エイリアス (AACRtpPacketizer など 6 件) を含まない。 型チェッカーは .pyi を .py より
# 優先するため、 スタブ側にもエイリアスが無いと利用者の import が失敗する。
file(READ ${RAW} _stub)
file(READ ${ALIASES} _aliases)
file(WRITE ${OUTPUT} "${_stub}\n${_aliases}")
