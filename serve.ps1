[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $ZolaArgs
)

$ErrorActionPreference = 'Stop'
$repositoryRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$previousLocation = Get-Location
$watcher = $null

try {
    Set-Location $repositoryRoot

    $zolaCommand = Get-Command zola -CommandType Application -ErrorAction SilentlyContinue
    if (-not $zolaCommand) {
        throw 'Zola was not found in PATH. Install Zola and make zola.exe available in PATH.'
    }

    $uvCommand = Get-Command uv -CommandType Application -ErrorAction SilentlyContinue
    if (-not $uvCommand) {
        throw 'uv was not found in PATH. Install uv and make it available in PATH.'
    }

    & $uvCommand.Source run --locked python -m scripts.prepare_assets
    if ($LASTEXITCODE -ne 0) {
        throw "Asset preparation failed with exit code $LASTEXITCODE."
    }

    $watcher = Start-Process -FilePath $uvCommand.Source `
        -ArgumentList @('run', '--locked', 'python', '-m', 'scripts.prepare_assets', '--watch') `
        -NoNewWindow -PassThru

    & $zolaCommand.Source serve --extra-watch-path data @ZolaArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Zola exited with code $LASTEXITCODE."
    }
}
finally {
    if ($watcher -and -not $watcher.HasExited) {
        & taskkill.exe /PID $watcher.Id /T /F 2>$null | Out-Null
        $watcher.WaitForExit()
    }
    Set-Location $previousLocation
}
