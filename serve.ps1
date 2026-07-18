[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $ZolaArgs
)

$ErrorActionPreference = 'Stop'

$zolaCommand = Get-Command zola -CommandType Application -ErrorAction SilentlyContinue
if (-not $zolaCommand) {
    throw 'Zola was not found in PATH. Install Zola and make zola.exe available in PATH.'
}

& $zolaCommand.Source serve --extra-watch-path data @ZolaArgs
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
