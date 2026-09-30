$ErrorActionPreference = 'Continue'
$sshOpts = @('-o','BatchMode=yes','-p','44664','root@connect.westb.seetacloud.com')
$deadline = (Get-Date).AddHours(15)
$interval = 600
$logfile = "E:\MyOwn\ProgramStudy\MAS\reports\auto_pull.log"

function Log($msg) {
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $msg"
    Add-Content -Path $logfile -Value $line
    Write-Output $line
}

New-Item -ItemType Directory -Force -Path "E:\MyOwn\ProgramStudy\MAS\reports" | Out-Null
Log "start polling GPU 400 completion (every 10 min)"

$done = $false
while ((Get-Date) -lt $deadline) {
    $r = & ssh @sshOpts "grep -q 'experiment_summary_400.csv' /root/autodl-tmp/MAS/run400.log && echo DONE"
    if ("$r" -match 'DONE') {
        $done = $true
        Log "detected 400 done, compressing"
        break
    }
    Start-Sleep -Seconds $interval
}

if (-not $done) {
    Log "timeout (15h) without done marker, exit"
    exit 1
}

$tarCmd = "cd /root/autodl-tmp/MAS && tar -czf results_400.tar.gz run400.log reports/batch_summary.csv reports/experiment_summary_400.csv reports/delta_recall.json reports/eval_400.csv reports/bm25_endtoend.json reports/hard_negative.json reports/negative_exp_manifest_400_error.json 2>/dev/null; ls -la results_400.tar.gz"
$tarOut = & ssh @sshOpts $tarCmd
Log "tar result: $tarOut"

$dst = "E:\MyOwn\ProgramStudy\MAS\reports\results_400.tar.gz"
scp -o BatchMode=yes -P 44664 "root@connect.westb.seetacloud.com:/root/autodl-tmp/MAS/results_400.tar.gz" $dst
if ($LASTEXITCODE -eq 0 -and (Test-Path $dst)) {
    Log "OK pulled to $dst ($((Get-Item $dst).Length) bytes)"
} else {
    Log "WARN scp failed, pull manually later"
}
Log "polling task finished"
