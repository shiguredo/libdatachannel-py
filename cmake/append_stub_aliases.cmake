# 生成された型スタブに、 __init__.py が定義するエイリアスを追記する
#
# nanobind_add_stub が生成するのは extension module のスタブで、 __init__.py の
# エイリアス (AACRtpPacketizer など) を含まない。 型チェッカーは .pyi を .py より
# 優先するため、 スタブ側にもエイリアスが無いと利用者の import が失敗する。
#
# エイリアスの一覧は __init__.py から取り出す (別ファイルに写すと更新が漏れるため)。
# CMake の正規表現の ^ は入力全体の先頭にしか一致しないので、 改行も条件に含める。
file(READ ${INIT_PY} _init_py)
string(REGEX MATCHALL "\n[A-Za-z_][A-Za-z0-9_]* = [A-Za-z_][A-Za-z0-9_]*"
       _alias_matches "\n${_init_py}")
set(_alias_lines "")
foreach(_match IN LISTS _alias_matches)
  string(STRIP "${_match}" _line)
  string(APPEND _alias_lines "${_line}\n")
endforeach()
if(_alias_lines STREQUAL "")
  message(FATAL_ERROR "Failed to extract aliases from ${INIT_PY}")
endif()

file(READ ${RAW} _stub)
file(WRITE ${OUTPUT} "${_stub}\n${_alias_lines}")
