function Run-Step($name, $command) {
    Write-Host ""
    Write-Host ">> $name"

    Invoke-Expression $command

    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR en: $name"
        exit 1
    }
}

Run-Step "Checking imports" "python -m compileall ."
Run-Step "Running ruff" "ruff check ."
Run-Step "Running mypy" "mypy ."

Write-Host ""
Write-Host "Starting app test..."

$process = Start-Process uvicorn -ArgumentList "main:app --port 9999" -PassThru
Start-Sleep -Seconds 3

if (!$process.HasExited) {
    Stop-Process -Id $process.Id
}

Write-Host ""
Write-Host "OK - Todo correcto"