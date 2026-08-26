<#
Watchdog for train_v2.py: if the python process dies for any reason (crash,
CUDA OOM, uncaught exception) while Windows itself stays up, relaunch it with
--auto-resume so it picks up from the last mid-epoch or last-completed
checkpoint automatically, with no manual epoch bookkeeping.

This does NOT by itself survive a full OS reboot (Kernel-Power 41 style power
loss) - Windows killing this script's own process along with python. To also
auto-resume after a reboot, register this script as a Task Scheduler task
that runs "At log on" - ask before setting that up, it's a persistent system
change.

Logging: routes python's output through cmd.exe's own `2>&1` rather than
PowerShell's redirection operators - PowerShell 5.1 wraps a native exe's
stderr lines in NativeCommandError objects when redirected with `2>&1`/`*>`,
which is noisy and unreliable to grep. cmd.exe merges the streams as plain
text before PowerShell ever sees them. Log is UTF-8.

Usage:
  powershell -File run_train_v2_resilient.ps1 --img-size 384 --ckpt-tag 384 --epochs 30 --warmup-epochs 5 --batch-size 64 --num-instances 4
#>
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$TrainArgs
)

$LogPath = "run_train_v2.log"

function Log($msg) {
    $line = "[$( Get-Date -Format 'yyyy-MM-dd HH:mm:ss' )] $msg"
    Write-Output $line
    Add-Content -Path $LogPath -Value $line -Encoding utf8
}

$argString = ($TrainArgs -join ' ')
$attempt = 0
while ($true) {
    $attempt++
    Log "=== Attempt ${attempt}: python train_v2.py --auto-resume $argString ==="

    # -u: unbuffered stdout/stderr. Without it, Python fully block-buffers stdout when
    # piped (not a TTY) - print()-based lines (epoch summaries, resume confirmations)
    # can sit unflushed for minutes while tqdm's stderr writes look fine, making the
    # log misleadingly quiet during genuinely healthy training.
    cmd /c "python -u train_v2.py --auto-resume $argString 2>&1" | Out-File -FilePath $LogPath -Append -Encoding utf8
    $exitCode = $LASTEXITCODE

    if ($exitCode -eq 0) {
        Log "Training exited cleanly (code 0). Done."
        break
    }

    Log "Training exited with code $exitCode - restarting in 15s (Ctrl+C to abort)..."
    Start-Sleep -Seconds 15
}
