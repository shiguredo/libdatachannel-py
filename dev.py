import argparse
import subprocess
import sys


# ファイルを読み込み、バージョンを更新
def update_version(file_path: str, dry_run: bool) -> str | None:
    with open(file_path, "r", encoding="utf-8") as f:
        current_version: str = f.read().strip()

    # バージョンが .devX を持っている場合の更新
    if ".dev" in current_version:
        # dev バージョンをインクリメント
        base_version, dev_suffix = current_version.rsplit(".dev", 1)
        new_version = f"{base_version}.dev{int(dev_suffix) + 1}"
    else:
        # .devX がない場合、次のマイナーバージョンにして .dev0 を追加
        parts = current_version.split(".")
        if len(parts) != 3:
            raise ValueError("Version format in VERSION file is not X.Y.Z")
        major, minor, _ = map(int, parts)
        new_version = f"{major}.{minor + 1}.0.dev0"

    print(f"Current version: {current_version}")
    print(f"New version: {new_version}")
    confirmation: str = input("Do you want to update the version? (Y/n): ").strip().lower()

    if confirmation != "y":
        print("Version update canceled.")
        return None

    # Dry-run 時の動作
    if dry_run:
        print("Dry-run: Version would be updated to:")
        print(new_version)
    else:
        with open(file_path, "w", encoding="utf-8") as f:
            # 末尾に改行を入れる。 改行が無いと commit 時に prek の end-of-file-fixer が
            # ファイルを書き換え、 フックが失敗して commit が中断される
            f.write(f"{new_version}\n")
        print(f"Version updated in {file_path} to {new_version}")

    return new_version


# prek の pytest フックはビルド済みの拡張モジュールを必要とする (prek.toml 参照)。
# 未ビルドの場合は make develop 相当を実行しておく
def ensure_extension(dry_run: bool) -> None:
    if dry_run:
        print("Dry-run: Would run 'make develop' when the extension is not built")
        return

    probe = subprocess.run(
        ["uv", "run", "python", "-c", "import libdatachannel"],
        check=False,
        capture_output=True,
        text=True,
    )
    if probe.returncode == 0:
        return

    print("Building the extension module for the pytest hook (make develop)...")
    subprocess.run(["make", "develop"], check=True)


# prek のフックを commit の前に実行する
#
# commit 時にも prek のフックが走るが、 失敗すると commit が中断されて tag と push まで
# 到達しない。 事前に実行して、 失敗した場合は理由を表示して終了する
def run_prek(dry_run: bool) -> None:
    if dry_run:
        print("Dry-run: Would run 'prek run --all-files'")
        return

    result = subprocess.run(["prek", "run", "--all-files"], check=False)
    if result.returncode != 0:
        print(
            "prek failed. Fix the reported issues and run this script again.",
            file=sys.stderr,
        )
        raise SystemExit(1)


# uv sync を実行し、uv.lock を git に追加
def run_uv_sync(dry_run: bool) -> None:
    if dry_run:
        print("Dry-run: Would run 'uv sync' and add 'uv.lock' to git")
    else:
        # uv sync の実行
        result = subprocess.run(["uv", "sync"], check=True, capture_output=True, text=True)
        print(result.stdout)

        # uv.lock ファイルを git に追加
        subprocess.run(["git", "add", "uv.lock"], check=True)
        print("Added 'uv.lock' to git")


# git コミット、タグ、プッシュを実行
def git_operations(new_version: str, dry_run: bool) -> None:
    if dry_run:
        print("Dry-run: Would run 'git add VERSION' and 'git add uv.lock'")
        print(f"Dry-run: Would run 'git commit -m Bump version to {new_version}'")
        print(f"Dry-run: Would run 'git tag {new_version}'")
        print("Dry-run: Would run 'git push'")
        print("Dry-run: Would run 'git push --tags'")
    else:
        subprocess.run(["git", "add", "VERSION"], check=True)
        subprocess.run(
            ["git", "commit", "-m", f"[canary] Bump version to {new_version}"], check=True
        )
        subprocess.run(["git", "tag", new_version], check=True)
        subprocess.run(["git", "push"], check=True)
        subprocess.run(["git", "push", "--tags"], check=True)


# メイン処理
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Update VERSION file, run uv sync, and commit changes."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Run in dry-run mode without making actual changes"
    )
    args = parser.parse_args()

    version_file_path: str = "VERSION"

    # バージョン更新
    new_version: str | None = update_version(version_file_path, args.dry_run)

    if not new_version:
        return  # ユーザーが確認をキャンセルした場合、処理を中断

    # uv sync 実行
    run_uv_sync(args.dry_run)

    # pytest フック用の拡張モジュールを用意
    ensure_extension(args.dry_run)

    # prek のフックを commit の前に実行
    run_prek(args.dry_run)

    # git 操作
    git_operations(new_version, args.dry_run)


if __name__ == "__main__":
    main()
