import os
import sys
import time
from pathlib import Path
import ollama

MODEL = os.getenv("OLLAMA_MODEL", "gemma4:31b-cloud")

PROMPT_TEMPLATE = Path("prompt.txt").read_text(encoding="utf-8")

def analyze_transcript(text: str) -> str:
    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "Ты аналитик B2B-разговоров. Отвечай строго в Markdown-таблице. Не придумывай факты.",
            },
            {
                "role": "user",
                "content": PROMPT_TEMPLATE.replace("{{TRANSCRIPT}}", text),
            },
        ],
        options={"temperature": 0.1, "num_predict": 2000},
    )
    return response["message"]["content"]

def main():
    input_dir  = Path("conversation")   # как у вас
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    files = sorted(input_dir.glob("*.txt"))
    if not files:
        print(f"Нет .txt файлов в {input_dir}/")
        sys.exit(1)

    print(f"Модель: {MODEL}")
    print(f"Найдено файлов: {len(files)}\n")

    for i, file in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {file.name}...", end=" ", flush=True)
        text = file.read_text(encoding="utf-8")
        t0 = time.time()
        try:
            result = analyze_transcript(text)
            dt = time.time() - t0
            (output_dir / f"{file.stem}.md").write_text(result, encoding="utf-8")
            print(f"✓ {dt:.1f}s")
        except Exception as e:
            print(f"✗ {e}")
            (output_dir / f"{file.stem}.md").write_text(f"ОШИБКА: {e}", encoding="utf-8")

if __name__ == "__main__":
    main()