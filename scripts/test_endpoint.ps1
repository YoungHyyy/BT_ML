$ErrorActionPreference = "Stop"

$health = Invoke-RestMethod -Method Get -Uri "http://localhost:3000/health"
if ($health.status -ne "healthy") {
    throw "Health check failed"
}

$body = @{ features = @(0.2, -1.1, 0.5, 1.3) } | ConvertTo-Json
$prediction = Invoke-RestMethod -Method Post `
    -Uri "http://localhost:3000/api/v1/predict" `
    -ContentType "application/json" `
    -Body $body

$prediction | ConvertTo-Json -Depth 5
