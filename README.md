# PHPAiModel-GRU (toy GRU chatbot in pure PHP)

Lightweight, educational **character-level GRU** you can **train in PHP** on `.txt` datasets and **chat with** via a simple web UI.

> ⚠️ This is a toy/learning project. Training in PHP is **slow** and single-threaded. Keep hyper-params small (see presets below) or pre-train elsewhere and import weights.

---

## Features

- **Pure PHP** GRU: forward/backward, SGD, gradient clipping.
- **UTF-8 safe** tokenization (works with Russian + English).
- **Trainer UI + SSE** progress stream with detailed log lines.
- **Chat UI** with model picker (newest preselected), Temp/Top-K controls.
- **Portable model format** (`Models/*.json`) shared by trainer/inference.
- PHP 7.4+ compatible.

---

## Project structure

```
PHPAiModel-GRU/
├── index.php                # Chat UI (light theme), model picker, generation controls
├── aicore.php               # Inference core: loads model, UTF-8 tokenize, GRU step + sampling
├── generator_weights.php    # Trainer UI + SSE backend (one file)
├── Models/                  # Saved models (*.json)
└── Datasets/                # Text corpora (*.txt)
```

---

## Requirements

- **PHP 7.4+** (8.x recommended)
- Web server or PHP’s built-in:  
  `php -S 127.0.0.1:8000`
- UTF-8 files **without BOM**
- For SSE: disable output buffering / compression if needed (see Troubleshooting).

---

## Quick start

1) **Folders**
```bash
mkdir -p Models Datasets
```

2) **Dataset**  
Put one or more UTF-8 `.txt` files into `Datasets/`.  
Format can be simple dialogue lines, e.g.:
```
Hello!
Hi there!

Привет! Как дела?
Отлично, спасибо!
```
(It's character-level, so pairs are just text separated by newlines.)

3) **Train**  
Open `generator_weights.php` in your browser, select dataset(s), set params (e.g. `H=128, SEQ=128, EPOCHS=20, LR=0.01`) and click **Start training**.  
You’ll see a header like:
```
Tokens: 12345
Vocab: 220
Planned steps: 960 (≈ 48 / epoch)
```
and periodic lines:
```
Progress:   3.54% | ETA 00:12:34 | Spent 00:00:29 | epoch 1/20 | step 17/48 | avg loss 2.98765
```
A `.json` model will be saved to `Models/`.

4) **Chat**  
Open `index.php`. The newest model is preselected. Type a message and send.  
Tweak **Temperature** (0.6–0.9) and **Top-K** (20–50) for fluency.

---

## Recommended presets

**Small dataset (≈ 1–10k chars, e.g., 50 pairs):**
- Trainer: `H=128, SEQ=128, EPOCHS=20, LR=0.01`
- Chat: `Temp=0.7, Top-K=30, Max=300`

**Medium (50k–200k chars):**
- `H=256, SEQ=128, EPOCHS=10, LR=0.01`

**Large (~400k+ chars):**
- `H=256, SEQ=128, EPOCHS=15–20, LR=0.005–0.008`

> Tip: step time grows ~ **O(H·(H+V))**, so big `H` and large vocabulary `V` slow things down.

---

## Model format (`Models/<name>.json`)

```json
{
  "H": 128,
  "V": 220,
  "vocab": { "a": 0, "b": 1, "…": 219 },
  "ivocab": ["a","b","…"],
  "W": { "Wz": [[...]], "Wr": [[...]], "Wh": [[...]],
         "Wy": [[...]], "bz": [...], "br": [...], "bh": [...], "by": [...] },
  "meta": {
    "dataset_files": ["dialog_ru_en.txt"],
    "epochs": 20, "seq_len": 128, "lr": 0.01,
    "created_at": "2025-08-27T12:34:56Z",
    "stop_on_newline": true
  }
}
```

---

## Notes on data and vocab (UTF-8)

- The trainer builds a **character** vocabulary via `preg_split('//u', ...)`.
- **Newlines** are real tokens; they help separate turns.
- Keep datasets clean: normalize quotes/long dashes, avoid tabs, ensure LF line-endings.
- If `Vocab` gets huge (thousands), consider normalizing or applying a **whitelist** of allowed characters before building vocab (see code comments).

---

## Troubleshooting

**“Incorrect response from aicore.php” in Chat UI**  
- Ensure `aicore.php` outputs **only JSON** (no notices/warnings before `json_encode`).  
- In `aicore.php` we disable error display and clean buffers at the top.

**SSE doesn’t stream / stalls**  
- In `generator_weights.php` SSE section:
  - `header('Content-Type: text/event-stream; charset=utf-8');`
  - Disable buffering:  
    ```php
    @ini_set('zlib.output_compression','0');
    @ini_set('output_buffering','0');
    while (ob_get_level() > 0) { @ob_end_flush(); }
    @ob_implicit_flush(true);
    ```
- Reverse proxies may buffer; for Nginx add `X-Accel-Buffering: no`.

**Gibberish text (“oy, pe, Ре o,efI”)**  
- Lower **LR** to `0.01` or `0.005`; increase **EPOCHS**; ensure **UTF-8** processing everywhere.  
- During generation: `Temp 0.6–0.8`, `Top-K 20–50`.

---

## Roadmap / Ideas

- Checkpoints per epoch & resume training.
- Top-p (nucleus) sampling.
- Data normalization helpers & vocab whitelist toggle.
- Import/export from Python frameworks.
- Byte-pair / wordpiece tokenization (non-char-level variants).

---

## License

MIT © 2025 · Artur Strazewicz

---

## Credits

- Concept, architecture, PHP GRU runtime, UI — **Artur Strazewicz**  
  GitHub: https://github.com/iStark/PHPAiModel-RNN  
  LinkedIn: https://www.linkedin.com/in/arthur-stark/  
  X (Twitter): https://x.com/strazewicz  
  TruthSocial: https://truthsocial.com/@strazewicz

---

## FAQ

**Is it really training a GRU in PHP?**  
Yes. It’s slow but instructive: full forward/backward pass in plain PHP.

**Do I need pairs (`Q/A`) or just text?**  
Either works. It’s character-level; pairs are simply text separated by newlines.

**Will it run on shared hosting?**  
Usually yes for small models. SSE may be buffered by hosting configs; the Chat UI works regardless.

**How do I pick the newest model automatically?**  
`index.php` sorts by file modification time and preselects the newest model.
