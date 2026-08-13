[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $ZolaArgs
)

$ErrorActionPreference = 'Stop'
$repositoryRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$previousLocation = Get-Location

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

    & $uvCommand.Source run --locked python scripts/build_font_subsets.py
    if ($LASTEXITCODE -ne 0) {
        throw "CJK font subset generation failed with exit code $LASTEXITCODE."
    }

    & $zolaCommand.Source build @ZolaArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Zola exited with code $LASTEXITCODE."
    }

    & $uvCommand.Source run --locked python scripts/build_font_subsets.py --check-rendered public
    if ($LASTEXITCODE -ne 0) {
        throw "Rendered CJK font coverage check failed with exit code $LASTEXITCODE."
    }
}
finally {
    Set-Location $previousLocation
}
