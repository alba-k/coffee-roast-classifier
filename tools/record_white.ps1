<#
Record the second (clean white) background, class by class.
Usage (from the project root, with .venv active):
    powershell -ExecutionPolicy Bypass -File tools\record_white.ps1 -Camera 0
    powershell -ExecutionPolicy Bypass -File tools\record_white.ps1 -Camera 0 -Only quemado,oscuro
Photos go to dataset_white\raw\<class>. When a class reaches its target the
recorder pauses: press Q to move on. Re-running continues where it stopped.
#>
param(
    [int]$Camera = 0,
    [int]$Interval = 3,
    [string]$Out = "dataset_white",
    [string[]]$Only = @()
)

$plan = @(
    @{ label = "quemado";      n = 150 },
    @{ label = "oscuro";       n = 150 },
    @{ label = "medio";        n = 180 },
    @{ label = "medio_oscuro"; n = 100 },
    @{ label = "claro";        n = 100 },
    @{ label = "verde";        n = 100 },
    @{ label = "nada_otro";    n = 100 }
)

foreach ($c in $plan) {
    if ($Only.Count -gt 0 -and $Only -notcontains $c.label) { continue }
    Write-Host ""
    Write-Host "== $($c.label): $($c.n) photos, 1 every $Interval s ==" -ForegroundColor Magenta
    Write-Host "Clean white sheet, same light. Move/rotate beans every ~15 photos."
    Write-Host "Layouts: G group, D spread, I single bean (press the key in the window)."
    Read-Host "Ready? Press Enter to open the camera"
    python main.py --camera $Camera --output $Out --label $c.label --target $c.n --interval $Interval --layout grupo --start
}
Write-Host "Done. Tell Claude: listo" -ForegroundColor Green
