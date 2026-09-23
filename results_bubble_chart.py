# -*- mode: python; lexical-binding: t -*-

import json
import platform
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

SCRIPT_DIR = Path(__file__).resolve().parent


# %% パラメータ数
def get_model_size(model_name: str) -> float | None:
    """モデル名からパラメータ数 (例: 'phi4-mini:3.8b' -> 3.8) を取得して float で返す関数"""
    # 数値（小数含む）の直後に 'b' または 'B' が続くパターンにマッチ
    match = re.search(r"(\d+(?:\.\d+)?)[bB]", model_name)
    if match:
        return float(match.group(1))
    return None


# %% model, dataset 取り出し
def extract_args_from_command(command: str) -> tuple[str, str]:
    """command文字列から model_name と eval_dataset_name を正規表現で抽出します。"""
    model_match = re.search(r"(?:--model_name|-m)\s+([^\s]+)", command)
    dataset_match = re.search(r"--eval_dataset_name\s+([^\s]+)", command)

    model_name = model_match.group(1) if model_match else "-"
    dataset_name = dataset_match.group(1) if dataset_match else "-"

    return model_name, dataset_name


# %% 変換
def convert_jsonl_to_json(input_stream) -> list[dict[str, Any]]:
    """JSONL形式の入力から generate_answers.py のコマンドのみ抽出して変換します。"""
    results = []

    for line in input_stream:
        line = line.strip()
        if not line:
            continue

        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue

        command = data.get("command", "")

        # generate_answers.py が含まれているコマンドのみを対象にする
        if "generate_answers.py" not in command:
            continue

        model_name, dataset_name = extract_args_from_command(command)

        elapsed_time = int(data.get("処理時間", 0))

        if elapsed_time < 30:
            continue

        results.append(
            {
                "model_name": model_name,
                "モデルサイズ": get_model_size(model_name),
                "dataset_name": dataset_name,
                "処理時間": elapsed_time,
            }
        )

    return results


# %%
def analyze_score(judge_model: str) -> list[dict[str, str]]:

    json_file = SCRIPT_DIR / "data" / "elapsed_time.jsonl"

    with open(json_file, "r", encoding="utf-8") as f:
        json_data = convert_jsonl_to_json(f)

    for record in json_data:
        dataset_dir = record.get("dataset_name", "").replace("/", "__")
        model_file = record.get("model_name", "").replace("/", "__") + ".json"

        jsonl_file = (
            SCRIPT_DIR
            / "data"
            / "judgements"
            / f"judge_{judge_model}"
            / dataset_dir
            / model_file
        )

        model_name_alias_dict = {
            "yuiseki/sarashina2.2:3b": "sarashina2.2:3b",
            "yuiseki/sarashina2.2:1b": "sarashina2.2:1b",
            "dsasai/llama3-elyza-jp-8b:latest": "elyza-llama3:8b",
            "digitsflow/bonsai-8b:latest": "bonsai:8b",
        }

        # sarashina などのため
        model_name = record.get("model_name", "")
        if x := model_name_alias_dict.get(model_name):
            record["model_name"] = x
        else:
            record["model_name"] = model_name

        try:
            # JSONL ファイルを読み込み
            df = pd.read_json(jsonl_file, lines=True)
        except FileNotFoundError:
            print(f"エラー: 指定されたファイル '{jsonl_file}' が見つかりませんでした。")
            continue
        except (ValueError, KeyError) as e:
            print(f"予期せぬエラーが発生しました: {e}")
            continue

        # 2. 'judge_output' 列から FINAL SCORE の数値を抽出
        df["extracted_score"] = (
            df["judge_output"].str.extract(r"FINAL SCORE:\s*(\d+)").astype(float)
        )

        # 集計
        count = len(df)
        total_score = df["extracted_score"].sum()

        record["問題数"] = count
        record["スコア合計"] = total_score
        # record["平均スコア"] = total_score / count

    # print(json.dumps(json_data, ensure_ascii=False, indent=2))
    return json_data


# %%
def bubble_chart(json_data: list[dict[str, str]], judge_model: str):

    # 1. Seabornテーマ設定
    sns.set_theme(style="whitegrid", palette="colorblind")

    # 2. OSに応じた日本語フォント自動設定 (sns.set_themeの直後に配置)
    system_os = platform.system()
    if system_os == "Darwin":  # macOS
        plt.rcParams["font.family"] = "Hiragino Sans"
    elif system_os == "Windows":  # Windows
        plt.rcParams["font.family"] = ["MS Gothic", "Yu Gothic", "sans-serif"]
    else:  # Linux / クラウド環境など
        plt.rcParams["font.family"] = [
            "Noto Sans CJK JP",
            "IPAexGothic",
            "sans-serif",
        ]

    # 3. マイナス記号の文字化け防止
    plt.rcParams["axes.unicode_minus"] = False

    # 1. データセットごとの「1問あたりの満点」定義
    MAX_SCORE_PER_QUESTION = {
        "elyza/ELYZA-tasks-100": 5,
        "lightblue/tengu_bench": 10,
        "shisa-ai/ja-mt-bench-1shot": 10,
        "yuzuai/rakuda-questions": 10,
    }

    df = pd.DataFrame(json_data)

    # 3. 指標の正規化計算
    # 1問あたりの満点と全体の満点を計算
    df["1問あたり満点"] = df["dataset_name"].map(MAX_SCORE_PER_QUESTION)
    df["データセット満点合計"] = df["問題数"] * df["1問あたり満点"]

    # 各データセットごとの「得点率(%)」および「1問あたり処理時間(秒)」
    df["得点率"] = (df["スコア合計"] / df["データセット満点合計"]) * 100
    df["1問あたり処理時間"] = df["処理時間"] / df["問題数"]

    # 4. モデルごとに集計（全データセットの平均得点率と合計問題数ベースの1問あたり処理時間）
    model_summary = (
        df.groupby("model_name")
        .agg(
            {
                "モデルサイズ": "first",
                "処理時間": "sum",
                "問題数": "sum",
                "得点率": "mean",  # 4ベンチマークの平均得点率
            }
        )
        .reset_index()
    )

    # モデル全体の1問あたり平均処理時間 (秒/問)
    model_summary["1問あたり平均処理時間"] = (
        model_summary["処理時間"] / model_summary["問題数"]
    )

    # 5. バブルチャート描画
    _, ax = plt.subplots(figsize=(11, 7))

    # バブルサイズ（モデルパラメータ数に応じてスケーリング）
    bubble_sizes = model_summary["モデルサイズ"] * 45 + 120

    scatter = ax.scatter(
        model_summary["1問あたり平均処理時間"],
        model_summary["得点率"],
        s=bubble_sizes,
        c=model_summary["モデルサイズ"],
        cmap="viridis",
        alpha=0.75,
        edgecolors="black",
        linewidths=1.5,
        zorder=3,
    )

    # カラーバー設定
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("モデルサイズ (Billion Parameters)", fontsize=11, fontweight="bold")

    # ラベルオフセット（重なり防止用調整）
    label_offsets = {
        "gpt-oss:20b": (40, -5),
        "gemma4:26b": (40, 10),
        "gemma4:12b": (45, 0),
        "gemma4:e4b": (40, 0),
        "gemma4:e2b": (-30, 10),
        "gemma2:9b": (40, 0),
        "qwen3.8:27b": (50, 0),
        "qwen3.5:27b": (-52, 10),
        "qwen3.5:9b": (38, 0),
        "qwen3.5:4b": (38, 0),
        "qwen3.5:2b": (38, 0),
        "sarashina2.2:3b": (50, -8),
        "sarashina2.2:1b": (46, 6),
        "phi4:14b": (30, -8),
        "phi4-mini:3.8b": (45, -10),
        "elyza-llama3:8b": (45, -10),
        "bonsai:8b": (37, -7),
    }

    for _, row in model_summary.iterrows():
        name = row["model_name"]
        x = row["1問あたり平均処理時間"]
        y = row["得点率"]

        offset = label_offsets.get(name, (0, 12))

        ax.annotate(
            # f"{name}({size:.0f}B, {y:.1f}%)",
            f"{name}",
            (x, y),
            xytext=offset,
            textcoords="offset points",
            ha="center",
            fontsize=9,
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "white",
                "alpha": 0.8,
                "edgecolor": "none",
            },
            zorder=4,
        )

    # 軸ラベル・タイトル設定
    ax.set_title(
        "LLM Benchmark: 処理時間 vs 得点率 (円の大きさ: モデルサイズ)",
        fontsize=13,
        fontweight="bold",
        pad=16,
    )
    ax.set_xlabel("1問あたりの平均処理時間 (秒 / 問)", fontsize=11, fontweight="bold")
    ax.set_ylabel("平均得点率 (%)", fontsize=11, fontweight="bold")

    ax.set_ylim(0, 95)
    ax.axhline(
        80,
        color="gray",
        linestyle="--",
        linewidth=1,
        alpha=0.6,
        label="80% 得点率ライン",
    )

    # 👇 グラフの右下にコメントを追加
    ax.text(
        0.98,
        0.07,  # x: 右端近く(0.98), y: 下端近く(0.03)
        f"※ Judge mode: {judge_model}",
        transform=ax.transAxes,
        ha="right",  # 右揃え
        va="bottom",  # 下揃え
        fontsize=9,
        color="blue",  # 文字色
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "alpha": 0.8,
            "edgecolor": "lightgray",
        },  # 読みやすく背景枠をつける場合
    )

    sns.despine()
    plt.tight_layout(pad=1.5)

    # 保存
    pngfile = SCRIPT_DIR / "data" / f"bubble_chart_{judge_model}.png"
    plt.savefig(pngfile, dpi=150, bbox_inches="tight")
    print(f"グラフを出力しました: {pngfile}")
    plt.close()


# %%
if __name__ == "__main__":
    for judge_model in ["qwen3.8:27b", "gpt-oss:20b"]:
        json_data = analyze_score(judge_model)
        bubble_chart(json_data, judge_model)

# Local Variables:
# indent-tabs-mode: nil
# truncate-lines: t
# End:
