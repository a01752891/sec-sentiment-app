import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import re
import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"
MAX_INPUT_TOKENS = 512

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

print(f"Loading {MODEL_NAME} on {device}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype).to(device)
print("Model ready!\n")


def extract_text(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        try:
            import pdfplumber
            with pdfplumber.open(file_path) as pdf:
                return "\n".join(page.extract_text() or "" for page in pdf.pages)
        except ImportError:
            return "pdfplumber not installed. Please upload a .txt or .htm file."

    if ext in (".htm", ".html"):
        from html.parser import HTMLParser

        class _Extractor(HTMLParser):
            def __init__(self):
                super().__init__()
                self.parts = []
                self._skip = False

            def handle_starttag(self, tag, attrs):
                if tag in ("script", "style"):
                    self._skip = True

            def handle_endtag(self, tag):
                if tag in ("script", "style"):
                    self._skip = False

            def handle_data(self, data):
                if not self._skip:
                    self.parts.append(data)

        parser = _Extractor()
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            parser.feed(f.read())
        return " ".join(parser.parts)

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def detect_sections(text: str) -> dict:
    """Split an SEC filing into its named items."""
    pattern = re.compile(r"((?:ITEM|Item)\s+\d+[A-Z]?\.?\s+\S[^\n]{4,60})")
    matches = list(pattern.finditer(text))

    sections = {"Full Document": text}
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        name = m.group(0).strip()[:70]
        sections[name] = text[start:end]

    return sections


def analyze_sentiment(text: str) -> str:
    if not text.strip():
        return "No text provided."

    tokens = tokenizer.encode(text)
    if len(tokens) > MAX_INPUT_TOKENS:
        text = tokenizer.decode(tokens[:MAX_INPUT_TOKENS], skip_special_tokens=True)
        text += "\n[...truncated for model context...]"

    messages = [
        {
            "role": "user",
            "content": (
                "Analyze the sentiment of the following SEC filing excerpt. "
                "Classify it as Positive, Negative, or Neutral and give a brief one-sentence reason.\n\n"
                f"Text:\n{text}\n\n"
                "Reply in this exact format:\n"
                "Sentiment: [Positive / Negative / Neutral]\n"
                "Reason: [one sentence]"
            ),
        }
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


# ── Global state (single-user local app) ────────────────────────────────────
_sections: dict[str, str] = {}


def load_file(file):
    if file is None:
        return "", gr.update(choices=[], value=None), ""

    path = file if isinstance(file, str) else file.name
    text = extract_text(path)

    _sections.clear()
    _sections.update(detect_sections(text))

    choices = list(_sections.keys())
    preview = _sections[choices[0]][:3000]
    return preview, gr.update(choices=choices, value=choices[0]), ""


def on_section_change(section_name):
    if section_name and section_name in _sections:
        return _sections[section_name][:3000]
    return ""


def run_analysis(section_name: str, custom_text: str) -> str:
    text = custom_text.strip()
    if not text:
        text = _sections.get(section_name, "")
    return analyze_sentiment(text)


# ── UI ───────────────────────────────────────────────────────────────────────
with gr.Blocks(title="SEC Sentiment Analyzer") as demo:
    gr.Markdown("# SEC Filing Sentiment Analyzer")
    gr.Markdown(
        "Upload an SEC filing and analyze the sentiment of the full document "
        "or a specific section using **Qwen2-0.5B-Instruct** — no API key, runs 100 % locally."
    )

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(
                label="Upload SEC File (.txt / .pdf / .htm)",
                file_types=[".txt", ".pdf", ".htm", ".html"],
            )
            section_dropdown = gr.Dropdown(
                label="Select section to analyze",
                choices=[],
                interactive=True,
            )
            text_preview = gr.Textbox(
                label="Section preview (first 3 000 chars)",
                lines=12,
                interactive=False,
            )

        with gr.Column(scale=1):
            custom_text = gr.Textbox(
                label="Or paste text directly (overrides section selection)",
                lines=12,
                placeholder="Paste any SEC text here…",
            )
            analyze_btn = gr.Button("Analyze Sentiment", variant="primary")
            result_box = gr.Textbox(
                label="Sentiment Result",
                lines=5,
                interactive=False,
            )

    file_input.change(
        fn=load_file,
        inputs=file_input,
        outputs=[text_preview, section_dropdown, result_box],
    )
    section_dropdown.change(
        fn=on_section_change,
        inputs=section_dropdown,
        outputs=text_preview,
    )
    analyze_btn.click(
        fn=run_analysis,
        inputs=[section_dropdown, custom_text],
        outputs=result_box,
    )

if __name__ == "__main__":
    demo.launch()
