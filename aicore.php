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

function sample_topk(array $probs, int $k=50, float $temp=1.0): int {
    $N = count($probs); $k = max(1, min($k, $N));
    // temp
    if ($temp <= 0) $temp = 1e-6;
    $logits = [];
    foreach ($probs as $p) { $logits[] = log(max($p,1e-9)); }
    $logits = array_map(fn($x)=>$x/$temp, $logits);
    // pick top-k
    $idxs = range(0,$N-1);
    array_multisort($logits, SORT_DESC, $idxs);
    $top = array_slice($idxs, 0, $k);
    $exps = []; $sum=0.0;
    foreach ($top as $i) { $e = exp($logits[$i]-$logits[$top[0]]); $exps[$i] = $e; $sum += $e; }
    $r = mt_rand() / mt_getrandmax(); $acc=0.0;
    foreach ($top as $i) { $acc += $exps[$i]/$sum; if ($r <= $acc) return $i; }
    return $top[array_key_last($top)];
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

$temperature = max(0.1, (float)($body['temperature'] ?? 0.9));
$top_k = max(1, (int)($body['top_k'] ?? 50));
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
for($t=0;$t<$max_tokens;$t++){
    [$h,$p] = gru_step($W, $h, $cur, $H, $V);
    $cur = sample_topk($p, $top_k, $temperature);
    $out_ids[] = $cur;
    // simple stop on newline
    if ($cfg['meta']['stop_on_newline'] ?? true) {
        $ch = $ivocab[$cur] ?? '';
        if ($ch === "\n") break;
    }
}

$reply = from_ids($out_ids, $ivocab);
$_SESSION['history'] = $history . "User: " . $prompt . "\nAssistant: " . $reply . "\n"; // keep convo

echo json_encode(['ok'=>true,'reply'=>$reply], JSON_UNESCAPED_UNICODE | JSON_UNESCAPED_SLASHES);