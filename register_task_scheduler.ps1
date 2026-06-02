# Run this script ONCE (as Administrator) to register the 0DTE trader in Windows Task Scheduler.
# After that it will launch automatically each weekday at 9:30 AM Eastern.

$taskName   = "0DTE-IronCondor-Trader"
$scriptDir  = Split-Path -Parent $MyInvocation.MyCommand.Path
$batFile    = Join-Path $scriptDir "run_dte0_trader.bat"

# 9:30 AM Eastern — adjust if your machine runs in a different timezone
$triggerTime = "09:30"

$action  = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c `"$batFile`"" -WorkingDirectory $scriptDir

# Mon–Fri weekly trigger
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At $triggerTime

$settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit (New-TimeSpan -Hours 8) `
    -StartWhenAvailable `
    -DontStopOnIdleEnd

# Run as current user; no password needed for interactive sessions
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Highest

Register-ScheduledTask `
    -TaskName   $taskName `
    -Action     $action `
    -Trigger    $trigger `
    -Settings   $settings `
    -Principal  $principal `
    -Force

Write-Host ""
Write-Host "✅  Task '$taskName' registered successfully." -ForegroundColor Green
Write-Host "    It will run every weekday at $triggerTime from:"
Write-Host "    $batFile"
Write-Host ""
Write-Host "To verify: open Task Scheduler and look under 'Task Scheduler Library'."
Write-Host "To run it now for testing: Start-ScheduledTask -TaskName '$taskName'"
