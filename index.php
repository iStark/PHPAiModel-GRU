<?php
/*
 * PHPAiModel-GRU — index.php
 * Chat UI (light theme) for a toy character-level GRU written in pure PHP.
 *
 * Features:
 *   • Model picker from /Models (newest by mtime is preselected).
 *   • Generation controls: Temperature, Top-K, Max tokens.
 *   • Sending via button and Ctrl/Cmd+Enter; auto-scroll; session reset.
 *   • Robust response parsing: read as text, then JSON.parse with error surfacing.
 *
 * How it works:
 *   • POST → aicore.php (application/json):
 *       { "model": "<file.json>", "prompt": "<text>", "temperature"?: number, "top_k"?: int, "max_tokens"?: int }
 *     Expected response: { "ok": true, "reply": "..." }.
 *
 * Requirements:
 *   • PHP 7.4+ (8.x recommended), UTF-8 (no BOM), model files in /Models/*.json.
 *
 * Developed by: Artur Strazewicz — concept, architecture, PHP GRU runtime, UI.
 * Year: 2025. License: MIT.
 *
 * Links:
 *   GitHub:      https://github.com/iStark/PHPAiModel-RNN
 *   LinkedIn:    https://www.linkedin.com/in/arthur-stark/
 *   TruthSocial: https://truthsocial.com/@strazewicz
 *   X (Twitter): https://x.com/strazewicz
 */
declare(strict_types=1);

// ---- Paths ----
$MODELS_DIR = __DIR__ . DIRECTORY_SEPARATOR . 'Models';
@mkdir($MODELS_DIR, 0777, true);

// ---- Helpers (PHP 7.4 safe) ----
function ends_with(string $haystack, string $needle): bool {
    $len = strlen($needle);
    if ($len === 0) return true;
    return substr($haystack, -$len) === $needle;
}

/**
 * Возвращает список моделей (*.json) отсортированный по времени изменения (новые сверху).
 * Если папка пустая — вернёт [].
 */
function list_models_by_mtime(string $dir): array {
    if (!is_dir($dir)) return [];
    $files = scandir($dir) ?: [];
    $mods = [];
    foreach ($files as $f) {
        $path = $dir . DIRECTORY_SEPARATOR . $f;
        if (!is_file($path)) continue;
        if (!ends_with(strtolower($f), '.json')) continue;
        $mods[] = ['name'=>$f, 'mtime'=>@filemtime($path) ?: 0];
    }
    usort($mods, static function($a,$b){ return $b['mtime'] <=> $a['mtime']; }); // DESC
    return array_map(static function($x){ return $x['name']; }, $mods);
}

$models = list_models_by_mtime($MODELS_DIR);
?>
<!doctype html>
<html lang="ru">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>PHPAiModel-GRU — Chat</title>
    <style>
        :root{
            --bg:#f7f8fb; --card:#ffffff; --text:#0f172a; --muted:#6b7280; --line:#e5e7eb; --accent:#2563eb;
            --chip:#eef2ff; --bot:#f9fafb; --danger:#ef4444;
        }
        *{box-sizing:border-box}
        html,body{height:100%}
        body{margin:0;background:var(--bg);color:var(--text);font-family:system-ui,-apple-system,Segoe UI,Roboto,Inter,Arial}
        header{position:sticky;top:0;z-index:5;background:var(--card);border-bottom:1px solid var(--line);padding:12px 16px;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
        header h1{font-size:16px;margin:0}
        header .sp{flex:1}
        select,button,textarea,input{border:1px solid var(--line);border-radius:10px;padding:8px 10px;font:inherit;background:#fff;color:var(--text)}
        select, input{background:#fff}
        button{background:var(--accent);color:#fff;border-color:transparent;cursor:pointer;transition:.15s opacity}
        button:disabled{opacity:.6;cursor:not-allowed}
        main{max-width:980px;margin:0 auto;padding:16px;}
        .hint{font-size:12px;color:var(--muted);margin:0 0 12px}
        .chatwrap{background:var(--card);border:1px solid var(--line);border-radius:16px;display:flex;flex-direction:column;min-height:70vh}
        .chat{flex:1;overflow:auto;padding:16px}
        .bubble{max-width:80%;padding:10px 12px;margin:8px 0;border-radius:12px;box-shadow:0 1px 0 rgba(0,0,0,.03);white-space:pre-wrap;word-wrap:break-word}
        .me{align-self:flex-end;background:var(--chip)}
        .bot{align-self:flex-start;background:var(--bot)}
        .composer{display:flex;gap:8px;border-top:1px solid var(--line);padding:12px}
        .composer textarea{flex:1;resize:vertical;min-height:54px}
        .row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
        .pill{display:flex;gap:6px;align-items:center;background:#fff;border:1px solid var(--line);border-radius:999px;padding:6px 10px}
        .pill input{width:70px}
        .error{color:var(--danger);font-size:12px}
        footer{background:#222;color:#eee;text-align:center;padding:20px;font-family:Arial, sans-serif;font-size:14px}
        footer a{color:#aaa;margin:0 8px;text-decoration:none}
        code{background:#eef2ff;border-radius:6px;padding:2px 6px}
    </style>
</head>
<body>
<header>
    <h1>PHPAiModel-GRU Chat</h1>
    <div class="sp"></div>
    <div class="row">
        <label for="model">Модель:</label>
        <select id="model" title="Выберите модель из /Models">
            <?php if (empty($models)): ?>
                <option value="">— нет моделей —</option>
            <?php else: ?>
                <?php foreach($models as $i=>$m): ?>
                    <option value="<?= htmlspecialchars($m, ENT_QUOTES|ENT_SUBSTITUTE, 'UTF-8') ?>"
                        <?= $i===0 ? 'selected' : '' ?>>
                        <?= htmlspecialchars($m, ENT_QUOTES|ENT_SUBSTITUTE, 'UTF-8') ?>
                    </option>
                <?php endforeach; ?>
            <?php endif; ?>
        </select>
        <div class="pill">
            <span>Temp</span><input id="temp" type="number" step="0.05" min="0.1" max="2" value="0.4">
        </div>
        <div class="pill">
            <span>Top-K</span><input id="topk" type="number" min="0" max="500" value="20">
        </div>
        <div class="pill">
            <span>Max</span><input id="maxtok" type="number" min="1" max="2000" value="300">
        </div>
        <button id="clear" title="Сбросить диалог">Сброс</button>
    </div>
</header>

<main>
    <p class="hint">Совет: обучи модель в <code>generator_weights.php</code>, файл появится в <code>/Models</code>, затем выбери её здесь и начни диалог.</p>
    <div class="chatwrap">
        <div id="chat" class="chat"></div>
        <div class="composer">
            <textarea id="prompt" placeholder="Напишите сообщение…"></textarea>
            <button id="send">Отправить</button>
        </div>
    </div>
    <div id="err" class="error" style="margin-top:8px"></div>
</main>

<hr style="margin-top:40px; border:0; border-top:1px solid #ccc;">

<footer>
    <div style="margin-bottom:10px;">
        <strong>PHPAiModel-GRU</strong> © 2025 — MIT License
    </div>
    <div style="margin-bottom:10px;">
        Developed by <a href="https://www.linkedin.com/in/arthur-stark/">Artur Strazewicz</a>
    </div>
    <div>
        <a href="https://github.com/iStark/PHPAiModel-GRU">GitHub</a> |
        <a href="https://x.com/strazewicz">X (Twitter)</a> |
        <a href="https://truthsocial.com/@strazewicz">TruthSocial</a>
    </div>
</footer>

<script>
    const elChat = document.getElementById('chat');
    const elPrompt = document.getElementById('prompt');
    const elSend = document.getElementById('send');
    const elModel = document.getElementById('model');
    const elTemp = document.getElementById('temp');
    const elTopK = document.getElementById('topk');
    const elMaxTok = document.getElementById('maxtok');
    const elClear = document.getElementById('clear');
    const elErr = document.getElementById('err');

    function addBubble(text, who){
        const div = document.createElement('div');
        div.className = 'bubble ' + (who==='me'?'me':'bot');
        div.textContent = text;
        elChat.appendChild(div);
        elChat.scrollTop = elChat.scrollHeight;
    }
    function setError(msg){ elErr.textContent = msg || ''; }

    async function sendMessage(){
        setError('');
        const model = elModel.value.trim();
        const prompt = elPrompt.value.trim();

        if (!model) { setError('Сначала выбери модель в списке.'); return; }
        if (!prompt) { setError('Введите сообщение.'); return; }

        addBubble(prompt, 'me');
        elPrompt.value = '';
        elPrompt.focus();
        elSend.disabled = true;

        try {
            const body = {
                model,
                prompt,
                temperature: parseFloat(elTemp.value || '0.4'),
                top_k: parseInt(elTopK.value || '20'),
                max_tokens: parseInt(elMaxTok.value || '300')
            };
            const res = await fetch('aicore.php', {
                method: 'POST',
                headers: {'Content-Type':'application/json'},
                body: JSON.stringify(body)
            });

            // читаем как текст, затем пытаемся распарсить JSON — чтобы поймать любые предупреждения/мусор
            const text = await res.text();
            let js;
            try {
                js = JSON.parse(text);
            } catch (e) {
                throw new Error('Некорректный ответ от aicore.php: ' + text.slice(0, 300));
            }

            if (!js.ok) throw new Error(js.error || 'Ошибка инференса');
            addBubble(js.reply || '(пусто)', 'bot');
        } catch (err) {
            setError(err.message || String(err));
            addBubble('Ошибка: ' + (err.message || String(err)), 'bot');
        } finally {
            elSend.disabled = false;
        }
    }

    // hotkeys
    elPrompt.addEventListener('keydown', (e)=>{
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            e.preventDefault(); sendMessage();
        }
    });
    elSend.addEventListener('click', ()=> sendMessage());

    // reset chat session
    elClear.addEventListener('click', async ()=>{
        setError('');
        try { await fetch('aicore.php?reset=1'); } catch(_){}
        elChat.innerHTML = '';
        addBubble('Диалог сброшен.', 'bot');
    });
</script>
</body>
</html>
