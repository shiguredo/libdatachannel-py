"""エラーログの共通処理

whip.py と whep.py で共有する。 標準の logging ではなく structlog を使っている。
"""

import structlog

logger = structlog.get_logger(__name__)


def handle_error(context: str, error: Exception) -> None:
    """エラーハンドリング

    structlog の logger は logging の isEnabledFor を持たないため、 レベル判定は
    自前で行わない。 スタックトレースは logger.debug に任せる (structlog は出力
    しないレベルでは例外を整形しないため、 debug ログが無効なら負荷はかからない)。
    """
    logger.error(f"Error {context}: {error}")
    logger.debug(f"Error {context}", exc_info=error)
