<?php
/*
 * PHPAiModel-GRU — generator_weights.php
 * Unified Trainer UI + SSE backend. Trains a character-level GRU on .txt datasets and
 * saves the model to /Models/*.json. Streams a detailed live log in the format:
 *   "Progress:  0.10% | ETA 02:08:28 | Spent 00:08 | epoch 1/20 | step 400/19948 | avg loss 170.56032"
 *
 * UI (GET without params):
 *   • Light theme, multi-select of datasets from /Datasets/*.txt.
 *   • Fields: H, SEQ, EPOCHS, LR, OUT (output model filename).
 *   • Progress bar + live log (header + periodic Progress lines).
 *
 * Backend (SSE, GET action=train):
 *   • Params: dataset[]=file.txt&H=..&SEQ=..&EPOCHS=..&LR=..&OUT=optional.json
 *   • Events:
 *       - progress: { percent, msg, note }   // note contains "Progress: ..." lines
 *       - done:     { ok, out_file, out_path, header?, footer? }
 *
 * Training:
 *   • UTF-8 safe vocabulary (preg_split('//u', ...)); SGD with gradient clipping.
 *   • Step complexity ~ O(H·(H+V)); keep H and V realistic.
 *   • Final save: /Models/<name>.json (compatible with aicore.php).
 *
 * PHP/SSE notes:
 *   • Works on PHP 7.4+; avoids str_ends_with (uses ends_with()/substr()).
 *   • For SSE: header('Content-Type: text/event-stream'), disable buffering:
 *       while (ob_get_level() > 0) ob_end_flush(); ob_implicit_flush(true);
 *     and preferably disable zlib.output_compression.
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
@ini_set('memory_limit','2048M');
@set_time_limit(0);

$DATASETS_DIR = __DIR__ . DIRECTORY_SEPARATOR . 'Datasets' . DIRECTORY_SEPARATOR;
$MODELS_DIR   = __DIR__ . DIRECTORY_SEPARATOR . 'Models'   . DIRECTORY_SEPARATOR;
@mkdir($DATASETS_DIR, 0777, true);
@mkdir($MODELS_DIR, 0777, true);

function ends_with(string $haystack, string $needle): bool {
    $len = strlen($needle);
    if ($len === 0) return true;
    return substr($haystack, -$len) === $needle;
}
function list_files(string $dir, string $ext): array {
    if (!is_dir($dir)) return [];
    $files = scandir($dir) ?: [];
    $out = [];
    foreach ($files as $f) {
        $lf = strtolower($f);
        if (ends_with($lf, '.' . strtolower($ext))) $out[] = $f;
    }
    sort($out);
    return $out;
}

$action = $_GET['action'] ?? '';

if ($action !== 'train') {
    // ---------- UI (light theme) ----------
    $datasets = list_files($DATASETS_DIR, 'txt');
    ?>
    <!doctype html>
    <html lang="ru">
    <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>GRU Trainer — PHPAiModel-GRU</title>
        <style>
            :root { --bg:#f9fafc; --panel:#ffffff; --ink:#1e1e2e; --muted:#667085; --acc:#3b82f6; }
            body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 system-ui,Segoe UI,Roboto,Arial}
            .wrap{max-width:880px;margin:0 auto;padding:24px}
            h1{margin:8px 0 16px;font-size:24px}
            .card{background:var(--panel);border:1px solid #e5e7eb;border-radius:16px;box-shadow:0 4px 12px rgba(0,0,0,.05);margin-bottom:16px}
            .card h2{font-size:18px;margin:0;padding:16px;border-bottom:1px solid #f0f0f0}
            .card .body{padding:16px}
            .muted{color:var(--muted)}
            .row2{display:grid;gap:10px;grid-template-columns:1fr 1fr 1fr}
            label{display:block;margin:8px 0 6px}
            select,input{width:100%;padding:10px 12px;border-radius:10px;border:1px solid #cbd5e1;background:white;color:var(--ink)}
            button{cursor:pointer;border:1px solid #cbd5e1;background:#f9fafc;color:var(--ink);padding:10px 14px;border-radius:12px}
            button.primary{background:var(--acc);color:white;border:none}
            .progress{height:10px;background:#f3f4f6;border-radius:999px;overflow:hidden;border:1px solid #e5e7eb}
            .bar{height:100%;width:0;background:linear-gradient(90deg,#3b82f6,#60a5fa)}
            .log{font-family:ui-monospace,Consolas,monospace;font-size:12px;height:300px;overflow:auto;background:#fdfdfd;border:1px solid #e5e7eb;border-radius:10px;padding:10px;white-space:pre;line-height:1.45}
            .row{display:flex;gap:8px;flex-wrap:wrap}
        </style>
    </head>
    <body>
    <div class="wrap">
        <h1>GRU Trainer <span class="muted">· PHPAiModel-GRU</span></h1>

        <div class="card">
            <h2>Параметры обучения</h2>
            <div class="body">
                <label>Датасеты (*.txt)</label>
                <select id="datasets" multiple size="6">
                    <?php foreach ($datasets as $d): ?>
                        <option value="<?= htmlspecialchars($d) ?>"><?= htmlspecialchars($d) ?></option>
                    <?php endforeach; ?>
                </select>

                <div class="row2">
                    <div>
                        <label>Hidden size (H)</label>
                        <input id="H" type="number" min="16" max="512" step="16" value="256">
                    </div>
                    <div>
                        <label>Seq len</label>
                        <input id="SEQ" type="number" min="16" max="256" step="16" value="128">
                    </div>
                    <div>
                        <label>Epochs</label>
                        <input id="EPOCHS" type="number" min="1" max="100" value="15">
                    </div>
                </div>

                <div class="row2" style="margin-top:8px">
                    <div>
                        <label>Learning rate</label>
                        <input id="LR" type="number" step="0.001" min="0.0001" max="1" value="0.03">
                    </div>
                    <div>
                        <label>Output name (optional)</label>
                        <input id="OUT" placeholder="gru_ruen_H64.json">
                    </div>
                    <div style="display:flex;align-items:flex-end;gap:8px">
                        <button class="primary" id="train">Start training</button>
                    </div>
                </div>

                <div style="margin-top:12px" class="progress"><div id="bar" class="bar"></div></div>
                <div id="status" class="muted" style="margin-top:6px">Idle</div>
                <div id="log" class="log"></div>
            </div>
        </div>
    </div>

    <script>
        const elDatasets = document.getElementById('datasets');
        const elH = document.getElementById('H');
        const elSEQ = document.getElementById('SEQ');
        const elEPOCHS = document.getElementById('EPOCHS');
        const elLR = document.getElementById('LR');
        const elOUT = document.getElementById('OUT');
        const elBar = document.getElementById('bar');
        const elLog = document.getElementById('log');
        const elStatus = document.getElementById('status');

        let es;

        function appendLog(line){
            elLog.textContent += line + "\n";
            elLog.scrollTop = elLog.scrollHeight;
        }
        function startTraining(){
            if (es) es.close();
            elLog.textContent='';
            elBar.style.width='0%';
            elStatus.textContent='Starting…';

            const ds = Array.from(elDatasets.selectedOptions).map(o=>o.value);
            if (ds.length===0) { alert('Выберите хотя бы один датасет'); return; }

            const params = new URLSearchParams({
                action:'train', H:elH.value, SEQ:elSEQ.value, EPOCHS:elEPOCHS.value, LR:elLR.value, OUT:elOUT.value
            });
            ds.forEach(d=>params.append('dataset[]', d));

            es = new EventSource('generator_weights.php?' + params.toString());
            es.addEventListener('progress', (e)=>{
                const j = JSON.parse(e.data);
                if (j.percent != null) elBar.style.width = j.percent.toFixed(2) + '%';
                if (j.msg) elStatus.textContent = j.msg;
                if (j.note) appendLog(j.note);
            });
            es.addEventListener('done', (e)=>{
                const j = JSON.parse(e.data);
                elBar.style.width = '100%';
                elStatus.textContent = 'Saved: ' + j.out_file;
                if (j.header) appendLog(j.header);
                if (j.footer) appendLog(j.footer);
                appendLog('Model saved to: ' + j.out_path);
                es.close();
            });
            es.addEventListener('error', ()=>{
                elStatus.textContent = 'Stream closed';
            });

        }
        document.getElementById('train').onclick = startTraining;
    </script>
    </body>
    </html>
    <?php
    exit;
}

// ---------- SSE TRAINING BACKEND ----------
header('Content-Type: text/event-stream');
header('Cache-Control: no-cache');
header('X-Accel-Buffering: no'); // nginx
// аккуратно снимаем все буферы вывода (если есть)
while (ob_get_level() > 0) { @ob_end_flush(); }
// включаем небуферизованный вывод
@ob_implicit_flush(true);

function ev(string $event, array $data){ echo "event: $event\n"; echo 'data: '.json_encode($data, JSON_UNESCAPED_UNICODE|JSON_UNESCAPED_SLASHES)."\n\n"; @ob_flush(); @flush(); }
function read_files(string $dir, array $names): string {
    $buf=''; foreach($names as $n){ $p=realpath($dir . basename($n)); if($p && is_file($p)) $buf .= file_get_contents($p); }
    return $buf;
}
function randn(): float { // Box-Muller
    static $z2 = null; static $use_last=false;
    if($use_last){ $use_last=false; return $z2; }
    $u1 = mt_rand()/mt_getrandmax(); $u2 = mt_rand()/mt_getrandmax();
    $R = sqrt(-2.0*log(max($u1,1e-12))); $theta = 2.0*M_PI*$u2; $z1 = $R*cos($theta); $z2 = $R*sin($theta); $use_last=true; return $z1;
}
function init_mat(int $rows, int $cols, float $scale): array { $M=[]; for($i=0;$i<$rows;$i++){ $row=[]; for($j=0;$j<$cols;$j++){ $row[$j] = randn()*$scale; } $M[$i]=$row; } return $M; }
function zeros(int $n): array { return array_fill(0,$n,0.0); }
function clip(&$v, float $c=5.0){ if(is_array($v)) foreach($v as &$x) clip($x,$c); else { if($v>$c)$v=$c; elseif($v<-$c)$v=-$c; } }
function fmt_hms(float $sec): string { if ($sec<0) $sec=0; $s=(int)round($sec); $h=intdiv($s,3600); $m=intdiv($s%3600,60); $ss=$s%60; return sprintf('%02d:%02d:%02d',$h,$m,$ss); }

// params
$H      = max(16, min(512, (int)($_GET['H'] ?? 256)));
$SEQ    = max(16, min(256, (int)($_GET['SEQ'] ?? 128)));
$EPOCHS = max(1,  min(100, (int)($_GET['EPOCHS'] ?? 15)));
$LR     = (float)($_GET['LR'] ?? 0.03);
$OUT    = trim((string)($_GET['OUT'] ?? ''));
$ds     = $_GET['dataset'] ?? [];

if (!is_array($ds) || count($ds)===0) { ev('progress',['msg'=>'No datasets selected','note'=>'Ошибка: датасеты не выбраны','percent'=>0]); exit; }

$text = read_files($DATASETS_DIR, $ds);
if ($text === '') { ev('progress',['msg'=>'Datasets empty','note'=>'Ошибка: файлы пустые или не найдены','percent'=>0]); exit; }
$text = str_replace("\r", "", $text); // CRLF → LF

// UTF-8 safe char split
$charArr = preg_split('//u', $text, -1, PREG_SPLIT_NO_EMPTY) ?: [];
$chars = [];
foreach ($charArr as $ch) { $chars[$ch] = true; }
$chars["\u{FFFD}"] = true;

// vocab
$ivocab = array_values(array_keys($chars));
sort($ivocab);
$vocab = [];
foreach ($ivocab as $i => $ch) $vocab[$ch] = $i;
$V = count($ivocab);

// encode text → ids
$ids = [];
foreach ($charArr as $ch) { $ids[] = $vocab[$ch] ?? $vocab["\u{FFFD}"]; }
$N = count($ids);

// steps
$total_steps_per_epoch = max(1, intdiv($N-1, $SEQ));
$planned_total_steps   = $EPOCHS * $total_steps_per_epoch;

// weights
$inZ = $H + $V; // for z,r gates
$inH = $H + $V; // for candidate (first H for u=r⊙h_prev)
$scaleZ = 1.0/sqrt($inZ); $scaleH = 1.0/sqrt($inH); $scaleY = 1.0/sqrt($H);
$W = [
    'Wz' => init_mat($H, $inZ, $scaleZ),
    'Wr' => init_mat($H, $inZ, $scaleZ),
    'Wh' => init_mat($H, $inH, $scaleH),
    'Wy' => init_mat($V, $H, $scaleY),
    'bz' => zeros($H), 'br'=>zeros($H), 'bh'=>zeros($H), 'by'=>zeros($V)
];

// forward (one-hot x)
function step_forward(array &$W, array $hprev, int $x_id, int $H, int $V): array {
    $Wz=$W['Wz']; $Wr=$W['Wr']; $Wh=$W['Wh']; $Wy=$W['Wy']; $bz=$W['bz']; $br=$W['br']; $bh=$W['bh']; $by=$W['by'];
    $z=$r=$h_tilde=$h=$u=array_fill(0,$H,0.0);

    for($i=0;$i<$H;$i++){ $acc=$bz[$i]; $row=$Wz[$i]; for($j=0;$j<$H;$j++){ $acc += $row[$j]*$hprev[$j]; } $acc += $row[$H+$x_id]; $z[$i]=1.0/(1.0+exp(-$acc)); }
    for($i=0;$i<$H;$i++){ $acc=$br[$i]; $row=$Wr[$i]; for($j=0;$j<$H;$j++){ $acc += $row[$j]*$hprev[$j]; } $acc += $row[$H+$x_id]; $ri=1.0/(1.0+exp(-$acc)); $r[$i]=$ri; $u[$i]=$ri*$hprev[$i]; }
    for($i=0;$i<$H;$i++){ $acc=$bh[$i]; $row=$Wh[$i]; for($j=0;$j<$H;$j++){ $acc += $row[$j]*$u[$j]; } $acc += $row[$H+$x_id]; $h_tilde[$i]=tanh($acc); }
    for($i=0;$i<$H;$i++){ $h[$i]=(1.0-$z[$i])*$hprev[$i] + $z[$i]*$h_tilde[$i]; }

    $logits=array_fill(0,$V,0.0);
    for($v=0;$v<$V;$v++){ $acc=$by[$v]; $row=$Wy[$v]; for($j=0;$j<$H;$j++){ $acc += $row[$j]*$h[$j]; } $logits[$v]=$acc; }

    $m=max($logits); $sum=0.0; $probs=[];
    foreach($logits as $x){ $e=exp($x-$m); $probs[]=$e; $sum+=$e; }
    if ($sum<=0) { $Vv=count($probs); $probs=array_fill(0,$Vv,1.0/$Vv); }
    else { foreach($probs as $i=>$p){ $probs[$i]=$p/$sum; } }

    return [$h,$z,$r,$u,$h_tilde,$probs,$logits];
}

// header log
$ds_names = array_values($ds);
$header_lines = [];
$header_lines[] = "Запуск обучения…";
$header_lines[] = "Dataset: " . implode(',', $ds_names);
$header_lines[] = "Tokens: " . $N;
$header_lines[] = "Vocab: " . $V;
$header_lines[] = "H: $H  SEQ: $SEQ  Epochs: $EPOCHS  LR: $LR";
$header_lines[] = "Planned steps: $planned_total_steps (≈ $total_steps_per_epoch / epoch)";
ev('progress', ['msg'=>'Training started','note'=>implode("\n", $header_lines)."\n"]);

$start = microtime(true);
$seen_steps = 0;
$loss_acc = 0.0; $tokens_acc = 0;

for($epoch=1;$epoch<=$EPOCHS;$epoch++){
    $h = array_fill(0,$H,0.0);

    for($s=0,$i=0; $s<$total_steps_per_epoch; $s++, $i+=$SEQ){
        // slice
        $xs = array_slice($ids, $i, $SEQ);
        $ys = array_slice($ids, $i+1, $SEQ);
        $T = count($ys);

        // forward & loss
        $cache = [];
        for($t=0;$t<$T;$t++){
            [$h,$z,$r,$u,$h_tilde,$probs,$logits] = step_forward($W, $h, $xs[$t], $H, $V);
            $cache[$t] = ['x'=>$xs[$t], 'y'=>$ys[$t], 'h'=>$h, 'z'=>$z, 'r'=>$r, 'u'=>$u, 'h_tilde'=>$h_tilde, 'probs'=>$probs];
            $p = $probs[$ys[$t]] ?? 1e-9;
            $loss_acc += -log(max($p,1e-9));
            $tokens_acc++;
        }

        // init grads
        $d = [
            'Wz'=>array_fill(0,$H,array_fill(0,$H+$V,0.0)),
            'Wr'=>array_fill(0,$H,array_fill(0,$H+$V,0.0)),
            'Wh'=>array_fill(0,$H,array_fill(0,$H+$V,0.0)),
            'Wy'=>array_fill(0,$V,array_fill(0,$H,0.0)),
            'bz'=>array_fill(0,$H,0.0), 'br'=>array_fill(0,$H,0.0), 'bh'=>array_fill(0,$H,0.0), 'by'=>array_fill(0,$V,0.0)
        ];
        $dh_next = array_fill(0,$H,0.0);
        $zeroH = array_fill(0,$H,0.0);

        // backward
        for($t=$T-1;$t>=0;$t--){
            $xid = $cache[$t]['x']; $yid = $cache[$t]['y']; $h_t = $cache[$t]['h'];
            $z = $cache[$t]['z']; $r = $cache[$t]['r']; $u = $cache[$t]['u']; $h_tilde = $cache[$t]['h_tilde']; $probs = $cache[$t]['probs'];
            $h_prev = ($t>0) ? $cache[$t-1]['h'] : $zeroH;

            $dy = $probs; $dy[$yid] -= 1.0;
            for($v=0;$v<$V;$v++){
                $d['by'][$v] += $dy[$v];
                $row =& $d['Wy'][$v];
                for($j=0;$j<$H;$j++){ $row[$j] += $dy[$v]*$h_t[$j]; }
            }
            $dh = $dh_next;
            for($j=0;$j<$H;$j++){ $acc=0.0; for($v=0;$v<$V;$v++){ $acc += $W['Wy'][$v][$j]*$dy[$v]; } $dh[$j] += $acc; }

            $dz_raw = array_fill(0,$H,0.0);
            $dh_tilde_raw = array_fill(0,$H,0.0);
            $dh_prev = array_fill(0,$H,0.0);
            for($j=0;$j<$H;$j++){
                $dz_raw[$j] = $dh[$j] * ($h_tilde[$j] - $h_prev[$j]);
                $dh_tilde_raw[$j] = $dh[$j] * $z[$j];
                $dh_prev[$j] += $dh[$j] * (1.0 - $z[$j]);
            }
            $dz_pre = array_fill(0,$H,0.0);
            $dh_tilde_pre = array_fill(0,$H,0.0);
            for($j=0;$j<$H;$j++){
                $dz_pre[$j] = $dz_raw[$j] * $z[$j]*(1.0-$z[$j]);
                $dh_tilde_pre[$j] = $dh_tilde_raw[$j] * (1.0 - $h_tilde[$j]*$h_tilde[$j]);
            }

            for($i2=0;$i2<$H;$i2++){
                $row =& $d['Wh'][$i2];
                for($j=0;$j<$H;$j++){ $row[$j] += $dh_tilde_pre[$i2] * $u[$j]; }
                $row[$H+$xid] += $dh_tilde_pre[$i2];
                $d['bh'][$i2] += $dh_tilde_pre[$i2];
            }
            $du = array_fill(0,$H,0.0);
            for($j=0;$j<$H;$j++){ $acc=0.0; for($i2=0;$i2<$H;$i2++){ $acc += $W['Wh'][$i2][$j] * $dh_tilde_pre[$i2]; } $du[$j] = $acc; }
            $dr_raw = array_fill(0,$H,0.0);
            for($j=0;$j<$H;$j++){ $dr_raw[$j] = $du[$j] * $h_prev[$j]; $dh_prev[$j] += $du[$j] * $r[$j]; }
            $dr_pre = array_fill(0,$H,0.0);
            for($j=0;$j<$H;$j++){ $dr_pre[$j] = $dr_raw[$j] * $r[$j]*(1.0-$r[$j]); }

            for($i2=0;$i2<$H;$i2++){
                $rowz =& $d['Wz'][$i2]; $rowr =& $d['Wr'][$i2];
                for($j=0;$j<$H;$j++){ $rowz[$j] += $dz_pre[$i2] * $h_prev[$j]; $rowr[$j] += $dr_pre[$i2]*$h_prev[$j]; }
                $rowz[$H+$xid] += $dz_pre[$i2];
                $rowr[$H+$xid] += $dr_pre[$i2];
                $d['bz'][$i2] += $dz_pre[$i2];
                $d['br'][$i2] += $dr_pre[$i2];
            }
            for($j=0;$j<$H;$j++){ $acc=0.0; for($i2=0;$i2<$H;$i2++){ $acc += $W['Wz'][$i2][$j]*$dz_pre[$i2]; } $dh_prev[$j] += $acc; }
            for($j=0;$j<$H;$j++){ $acc=0.0; for($i2=0;$i2<$H;$i2++){ $acc += $W['Wr'][$i2][$j]*$dr_pre[$i2]; } $dh_prev[$j] += $acc; }

            $dh_next = $dh_prev;
        }

        clip($d, 5.0);
        foreach(['Wz','Wr','Wh'] as $M){ for($i2=0;$i2<$H;$i2++){ for($j=0;$j<$H+$V;$j++){ $W[$M][$i2][$j] -= $LR*$d[$M][$i2][$j]; } }}
        for($v=0;$v<$V;$v++){ for($j=0;$j<$H;$j++){ $W['Wy'][$v][$j] -= $LR*$d['Wy'][$v][$j]; } $W['by'][$v] -= $LR*$d['by'][$v]; }
        for($i2=0;$i2<$H;$i2++){ $W['bz'][$i2]-=$LR*$d['bz'][$i2]; $W['br'][$i2]-=$LR*$d['br'][$i2]; $W['bh'][$i2]-=$LR*$d['bh'][$i2]; }

        // progress
        $seen_steps++;
        if (($seen_steps % 50) === 0 || $seen_steps === 1) {
            $pct   = 100.0 * $seen_steps / $planned_total_steps;
            $spent = microtime(true) - $start;
            $eta   = $spent>0 ? ($planned_total_steps - $seen_steps) * ($spent / max(1,$seen_steps)) : 0.0;
            $avg   = $tokens_acc>0 ? $loss_acc/$tokens_acc : 0.0;

            $line = sprintf(
                "Progress: %7.2f%% | ETA %s | Spent %s | epoch %d/%d | step %d/%d | avg loss %.5f",
                $pct,
                fmt_hms($eta),
                fmt_hms($spent),
                $epoch, $EPOCHS,
                $s+1, $total_steps_per_epoch,
                $avg
            );
            ev('progress', [
                'percent'=>$pct,
                'msg'=>sprintf('Epoch %d/%d · step %d/%d', $epoch,$EPOCHS,$s+1,$total_steps_per_epoch),
                'note'=>$line
            ]);
        }
    }
}

// save model
$meta = [
    'dataset_files' => $ds_names,
    'epochs' => $EPOCHS,
    'seq_len' => $SEQ,
    'lr' => $LR,
    'created_at' => date('c'),
    'stop_on_newline' => true
];
$out_name = $OUT !== '' ? basename($OUT) : ('gru_'.($ds_names[0] ?? 'dataset')."_H{$H}_E{$EPOCHS}_".date('Ymd_His').'.json');
$out_path = $MODELS_DIR . $out_name;

$model = ['H'=>$H,'V'=>$V,'vocab'=>$vocab,'ivocab'=>$ivocab,'W'=>$W,'meta'=>$meta];
file_put_contents($out_path, json_encode($model, JSON_UNESCAPED_UNICODE|JSON_UNESCAPED_SLASHES));

ev('done', [
    'ok'=>true,
    'out_file'=>$out_name,
    'out_path'=>$out_path,
    'header'=>implode("\n", $header_lines),
    'footer'=>"Готово."
]);
