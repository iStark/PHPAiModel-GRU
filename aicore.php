<?php
/*
 * PHPAiModel-GRU — aicore.php
 * Inference core for a character-level GRU in pure PHP: model loading, UTF-8 tokenization,
 * GRU step, sampling (temperature + top-k), and a tiny session chat history.
 *
 * API:
 *   • POST (application/json):
 *       {
 *         "model": "<file.json>",               // filename inside /Models
 *         "prompt": "<user text>",
 *         "temperature": 0.1..2.0,              // optional
 *         "top_k": 1..500,                      // optional
 *         "max_tokens": 1..2000                 // optional
 *       }
 *     → JSON: { "ok": true, "reply": "<text>" }
 *   • GET ?reset=1 — clears chat history (PHP session).
 *
 * Model format (Models/<name>.json):
 *   {
 *     "H": int, "V": int,
 *     "vocab": { "<char>": id, ... },
 *     "ivocab": [ "<char0>", "<char1>", ... ],
 *     "W": { "Wz","Wr","Wh","Wy","bz","br","bh","by" },
 *     "meta": { "dataset_files":[], "epochs":..., "seq_len":..., "lr":..., "created_at":"...", "stop_on_newline": true }
 *   }
 *
 * Notes:
 *   • UTF-8 safe tokenization via preg_split('//u', ...).
 *   • No str_ends_with usage (PHP 7.4 safe) — extension checks use substr(..., -N).
 *   • Outputs clean JSON only (error_reporting(0); no echoes before json_encode()).
 *   • Educational speed — keep H/SEQ and max_tokens reasonable.
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
session_start();
// aicore.php — GRU inference (character-level). Maintains a small session chat history.
// POST JSON { model, prompt, temperature, top_k, top_p, max_tokens }
// Жестко выключаем любые варнинги/нотис и буферы
error_reporting(0);
@ini_set('display_errors','0');
@ini_set('zlib.output_compression','0');
@ini_set('output_buffering','0');
while (ob_get_level() > 0) { @ob_end_clean(); }

header('Content-Type: application/json; charset=utf-8');

$MODELS_DIR = __DIR__ . DIRECTORY_SEPARATOR . 'Models' . DIRECTORY_SEPARATOR;

if (isset($_GET['reset'])) { unset($_SESSION['history']); echo json_encode(['ok'=>true]); exit; }

$raw = file_get_contents('php://input') ?: '';
$body = json_decode($raw, true) ?? [];
$model = (string)($body['model'] ?? '');
if (
    $model === ''
    || !preg_match('/^[A-Za-z0-9._\-]+$/', $model)
    || substr(strtolower($model), -5) !== '.json'
) {
    http_response_code(400);
    echo json_encode(['ok'=>false,'error'=>'invalid model']);
    exit;
}
$path = realpath($MODELS_DIR . basename($model));
if (!$path || !is_file($path)) { http_response_code(404); echo json_encode(['ok'=>false,'error'=>'model not found']); exit; }

$cfg = json_decode(file_get_contents($path) ?: '[]', true);
if (!$cfg || !isset($cfg['vocab'], $cfg['ivocab'], $cfg['H'], $cfg['W'])) {
    http_response_code(500); echo json_encode(['ok'=>false,'error'=>'bad model file']); exit;
}

// --- helpers ---
function softmax(array $v): array {
    $m = max($v); $sum = 0.0; $out = [];
    foreach ($v as $x) { $e = exp($x - $m); $out[] = $e; $sum += $e; }
    if ($sum <= 0) { $n = count($out); return array_fill(0,$n,1.0/$n); }
    foreach ($out as $i=>$_) { $out[$i] /= $sum; }
    return $out;
}

function sample_topk(array $probs, int $top_k = 0, float $temperature = 0.4): int {
    $V = count($probs);
    if ($V === 0) return 0;

    // safety: корректируем нули
    $eps = 1e-12;
    for ($i=0; $i<$V; $i++) {
        if (!is_finite($probs[$i]) || $probs[$i] < 0) $probs[$i] = 0.0;
    }
    $sum = array_sum($probs);
    if ($sum <= 0) {
        // fallback: равномерно
        return random_int(0, $V-1);
    }
    // нормализуем входные вероятности, на всякий случай
    for ($i=0; $i<$V; $i++) $probs[$i] /= $sum;

    // переводим в логиты и применяем температуру
    $logits = [];
    for ($i=0; $i<$V; $i++) {
        $p = max($probs[$i], $eps);
        $logits[$i] = log($p); // surrogate logits
        if ($temperature > 0) {
            $logits[$i] /= $temperature;
        }
    }

    // top-k: оставляем k наибольших логитов
    if ($top_k > 0 && $top_k < $V) {
        // получим индексы топ-K
        $idx = range(0, $V-1);
        usort($idx, function($a,$b) use ($logits){ return $logits[$b] <=> $logits[$a]; });
        $keep = array_slice($idx, 0, $top_k);
        $keepSet = array_fill_keys($keep, true);
        // маскируем остальные минус бесконечностью (большой -INF)
        $NEG_INF = -1e30;
        for ($i=0; $i<$V; $i++) {
            if (!isset($keepSet[$i])) $logits[$i] = $NEG_INF;
        }
    }

    // стабилизация softmax
    $maxLogit = max($logits);
    $exps = [];
    $sumExp = 0.0;
    for ($i=0; $i<$V; $i++) {
        // если был -INF, exp станет 0
        $e = exp($logits[$i] - $maxLogit);
        $exps[$i] = $e;
        $sumExp += $e;
    }
    if ($sumExp <= 0) {
        // на всякий: выбираем argmax логитов
        $best = 0; $bestVal = -INF;
        for ($i=0; $i<$V; $i++) if ($logits[$i] > $bestVal) { $bestVal = $logits[$i]; $best = $i; }
        return $best;
    }

    // превращаем в распределение и семплим по CDF
    $r = mt_rand() / mt_getrandmax();
    $acc = 0.0;
    for ($i=0; $i<$V; $i++) {
        $acc += $exps[$i] / $sumExp;
        if ($r <= $acc) return $i;
    }
    return $V-1; // из-за численной погрешности
}

// --- state ---
$vocab = $cfg['vocab'];           // char => id
$ivocab = $cfg['ivocab'];         // id => char
$H = (int)$cfg['H'];
$V = (int)$cfg['V'];
$W = $cfg['W'];                   // weights

// persistent hidden state? For simple chat we reset per generation.
$history = (string)($_SESSION['history'] ?? '');
$prompt = trim((string)($body['prompt'] ?? ''));
if ($prompt === '') { echo json_encode(['ok'=>false,'error'=>'empty prompt']); exit; }

$temperature = max(0.1, (float)($body['temperature'] ?? 0.4));
$top_k = max(1, (int)($body['top_k'] ?? 20));
$max_tokens = max(1, min(2000, (int)($body['max_tokens'] ?? 300)));

// Compose input text (very simple chat formatting)
$input = $history . "User: " . $prompt . "\nAssistant: ";

// --- forward step
function gru_step(array $W, array $h_prev, int $x_id, int $H, int $V): array {
    // Wz, Wr, Wh: [H][H+V] ; bz, br, bh: [H]
    // Wy: [V][H]; by:[V]
    [$Wz,$Wr,$Wh,$Wy,$bz,$br,$bh,$by] = [$W['Wz'],$W['Wr'],$W['Wh'],$W['Wy'],$W['bz'],$W['br'],$W['bh'],$W['by']];
    // concat helpers: W[:, 0:H] * h_prev  +  W[:, H+ x_id]
    $z = array_fill(0,$H,0.0); $r = $z; $h_tilde = $z;
    for($i=0;$i<$H;$i++){
        $acc = $bz[$i];
        $row = $Wz[$i];
        for($j=0;$j<$H;$j++){ $acc += $row[$j]*$h_prev[$j]; }
        $acc += $row[$H+$x_id];
        $z[$i] = 1.0/(1.0+exp(-$acc));
    }
    $u = array_fill(0,$H,0.0); // r ⊙ h_prev
    for($i=0;$i<$H;$i++){
        $acc = $br[$i];
        $row = $Wr[$i];
        for($j=0;$j<$H;$j++){ $acc += $row[$j]*$h_prev[$j]; }
        $acc += $row[$H+$x_id];
        $ri = 1.0/(1.0+exp(-$acc));
        $r[$i] = $ri; $u[$i] = $ri*$h_prev[$i];
    }
    for($i=0;$i<$H;$i++){
        $acc = $bh[$i];
        $row = $Wh[$i];
        // first H columns multiply u = r ⊙ h_prev
        for($j=0;$j<$H;$j++){ $acc += $row[$j]*$u[$j]; }
        $acc += $row[$H+$x_id];
        $h_tilde[$i] = tanh($acc);
    }
    $h = array_fill(0,$H,0.0);
    for($i=0;$i<$H;$i++){ $h[$i] = (1.0-$z[$i])*$h_prev[$i] + $z[$i]*$h_tilde[$i]; }
    // logits and probs
    $logits = array_fill(0,$V,0.0);
    for($v=0;$v<$V;$v++){
        $acc = $by[$v]; $row = $Wy[$v];
        for($j=0;$j<$H;$j++){ $acc += $row[$j]*$h[$j]; }
        $logits[$v] = $acc;
    }
    // softmax
    $probs = softmax($logits);
    return [$h,$probs,$z,$r,$h_tilde];
}

// tokenize chars → ids
function to_ids(string $s, array $vocab): array {
    $s = str_replace("\r", "", $s);
    $arr = preg_split('//u', $s, -1, PREG_SPLIT_NO_EMPTY) ?: [];
    $ids = [];
    foreach ($arr as $ch) { $ids[] = $vocab[$ch] ?? ($vocab["\u{FFFD}"] ?? 0); }
    return $ids;
}
function from_ids(array $ids, array $ivocab): string { $out=''; foreach($ids as $i){ $out .= $ivocab[$i] ?? '?'; } return $out; }

$ids = to_ids($input, $vocab);
$h = array_fill(0,$H,0.0);
$last_id = end($ids); if ($last_id === false) { $last_id = 0; }

// warmup on the prompt
foreach ($ids as $id) { [$h,$p] = gru_step($W, $h, $id, $H, $V); }

// generate
$out_ids = [];
$cur = $last_id; // seed with last char
for ($t = 0; $t < $max_tokens; $t++) {
    // gru_step должен вернуть [h, probs] где probs — массив длины V с вероятностями
    [$h, $probs] = gru_step($W, $h, $cur, $H, $V);

    // sanity-check: массив? есть NaN/Inf?
    $valid = is_array($probs) && count($probs) === $V;
    $nanCount = 0;
    if ($valid) {
        foreach ($probs as $q) {
            if (!is_finite($q) || $q < 0) { $nanCount++; }
        }
    }

    // топ-5 для удобства
    $topK = 5;
    $pairs = [];
    if ($valid) {
        foreach ($probs as $idx => $pv) { $pairs[] = [$idx, $pv]; }
        usort($pairs, function($a,$b){ return $b[1] <=> $a[1]; });
        $topView = array_slice($pairs, 0, $topK);
    } else {
        $topView = [];
    }

    // лог
    file_put_contents('debug.log',
        sprintf(
            "Step %d: valid=%s, nan=%d, V=%d, cur=%d ('%s'), top=%s\n",
            $t,
            $valid ? 'yes' : 'no',
            $nanCount,
            $V,
            $cur,
            $ivocab[$cur] ?? '',
            json_encode(array_map(function($p) use ($ivocab) {
                [$i,$pv] = $p;
                return ['i'=>$i, 'p'=>$pv, 'ch'=>$ivocab[$i] ?? ''];
            }, $topView), JSON_UNESCAPED_UNICODE)
        ),
        FILE_APPEND
    );

    // выбор следующего токена
    $cur = sample_topk($probs, $top_k, $temperature);
    $out_ids[] = $cur;

    // стоп по переводу строки (если включено в метаданных модели)
    if (($cfg['meta']['stop_on_newline'] ?? true) && (($ivocab[$cur] ?? '') === "\n")) {
        break;
    }
}
$reply = from_ids($out_ids, $ivocab);
$_SESSION['history'] = $history . "User: " . $prompt . "\nAssistant: " . $reply . "\n"; // keep convo

echo json_encode(['ok'=>true,'reply'=>$reply], JSON_UNESCAPED_UNICODE | JSON_UNESCAPED_SLASHES);