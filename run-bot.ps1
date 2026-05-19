param(
    [string]$Path = ".env"
)

if (-not (Test-Path $Path)) {
    Write-Error "File '$Path' not found."
    exit 1
}

Get-Content $Path | ForEach-Object {
    $line = $_.Trim()
    if ([string]::IsNullOrEmpty($line) -or $line.StartsWith('#')) {
        return
    }
    if ($line -match '^export\s+') {
        $line = $line -replace '^export\s+', ''
    }
    $eqIndex = $line.IndexOf('=')
    if ($eqIndex -le 0) {
        return
    }
    $key   = $line.Substring(0, $eqIndex).Trim()
    $value = $line.Substring($eqIndex + 1).Trim()
    if ($value.Length -ge 2) {
        $first = $value[0]
        $last  = $value[$value.Length - 1]
        if (($first -eq '"' -and $last -eq '"') -or ($first -eq "'" -and $last -eq "'")) {
            $value = $value.Substring(1, $value.Length - 2)
        }
    }
    Set-Item -Path "env:$key" -Value $value
}

& .\.venv\Scripts\python.exe .\main.py
