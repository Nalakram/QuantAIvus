param ([switch]$Force)

$modules = @{
    "api" = @("api_server")
    "data" = @("DataLoader", "Preprocessor")
    "inference" = @("ModelInference", "FeatureEngineering")
    "backtesting" = @("Trainer", "Evaluator")
}

Write-Host "[1/4] Creating directory structure..." -ForegroundColor Green
New-Item -Path "include", "src\api", "src\data", "src\inference", "src\backtesting" -ItemType Directory -Force | Out-Null

Write-Host "[2/4] Creating header/source files with class templates..." -ForegroundColor Green
foreach ($subdir in $modules.Keys) {
    foreach ($class in $modules[$subdir]) {
        $header = "src\$subdir\$class.h"
        $source = "src\$subdir\$class.cpp"
        Write-Host "Generating $subdir\$class..."

        # Generate additional methods based on class
        $extraMethods = ""
        if ($class -eq "api_server") {
            $extraMethods = @"
    void start_server();
    void handle_prediction_request();
"@
        } elseif ($class -eq "DataLoader") {
            $extraMethods = @"
    void load_data(const std::string& path);
"@
        } elseif ($class -eq "Preprocessor") {
            $extraMethods = @"
    void preprocess_data();
"@
        } elseif ($class -eq "ModelInference") {
            $extraMethods = @"
    void predict();
"@
        } elseif ($class -eq "FeatureEngineering") {
            $extraMethods = @"
    void extract_features();
"@
        } elseif ($class -eq "Trainer") {
            $extraMethods = @"
    void train_model();
"@
        } elseif ($class -eq "Evaluator") {
            $extraMethods = @"
    void evaluate_model();
"@
        }

        if (-not (Test-Path $header) -or $Force) {
            $guard = "STUCKPREDICTION_$($class.ToUpper())_H"
            $content = @"
#ifndef $guard
#define $guard

// src/$subdir/$class.h
// TODO: Add class functionality for $class

#include <string>

class $class {
public:
    $class();
    ~$class();
$extraMethods
private:
    // TODO: Add private members and methods
};

#endif // $guard
"@
            # Convert to CRLF and write
            $content = $content -replace "`r`n|`n", "`r`n"
            [System.IO.File]::WriteAllText($header, $content)
        } else {
            Write-Host "Skipping $header (already exists, use -Force to overwrite)." -ForegroundColor Yellow
        }

        if (-not (Test-Path $source) -or $Force) {
            $methodImpl = ""
            if ($extraMethods) {
                $methodImpl = ($extraMethods -split "`n" | ForEach-Object {
                    if ($_ -match "^\s*(void)\s+(\w+\([^)]*\))\s*;") {
                        $methodNameWithParams = $Matches[2]  # e.g., "predict()"
                        $methodName = $methodNameWithParams -replace "\([^)]*\)", ""  # e.g., "predict"
                        return "void $class::$methodNameWithParams {`n    // TODO: Implement $methodName`n}"
                    }
                }) -join "`n`n"
            }
            $content = @"
#include "$class.h"

$class::$class() {
    // TODO: Initialize $class
}

$class::~$class() {
    // TODO: Cleanup $class
}

$methodImpl
"@
            # Convert to CRLF and write
            $content = $content -replace "`r`n|`n", "`r`n"
            [System.IO.File]::WriteAllText($source, $content)
        } else {
            Write-Host "Skipping $source (already exists, use -Force to overwrite)." -ForegroundColor Yellow
        }
    }
}

Write-Host "[3/4] Creating main.cpp..." -ForegroundColor Green
if (-not (Test-Path "src\main.cpp") -or $Force) {
    $content = @"
#include <iostream>

// src/main.cpp
// Main entry point for StockPredictionApp backend

int main() {
    std::cout << "StockPredictionApp backend running..." << std::endl;
    // TODO: Initialize API server, data loader, etc.
    return 0;
}
"@
    # Convert to CRLF and write
    $content = $content -replace "`r`n|`n", "`r`n"
    [System.IO.File]::WriteAllText("src\main.cpp", $content)
} else {
    Write-Host "Skipping src\main.cpp (already exists, use -Force to overwrite)." -ForegroundColor Yellow
}

Write-Host "[4/4] Verifying generated files..." -ForegroundColor Green
$missing = $false
$files = @(
    "src\api\api_server.cpp", "src\api\api_server.h",
    "src\data\DataLoader.cpp", "src\data\DataLoader.h",
    "src\data\Preprocessor.cpp", "src\data\Preprocessor.h",
    "src\inference\ModelInference.cpp", "src\inference\ModelInference.h",
    "src\inference\FeatureEngineering.cpp", "src\inference\FeatureEngineering.h",
    "src\backtesting\Trainer.cpp", "src\backtesting\Trainer.h",
    "src\backtesting\Evaluator.cpp", "src\backtesting\Evaluator.h",
    "src\main.cpp"
)
foreach ($file in $files) {
    if (-not (Test-Path $file)) {
        Write-Host "WARNING: Missing file $file" -ForegroundColor Red
        $missing = $true
    }
}
if ($missing) {
    Write-Host "Some files are missing. Check warnings above." -ForegroundColor Red
} else {
    Write-Host "All expected files generated successfully." -ForegroundColor Green
}

Write-Host "`n✅ Setup complete. Ready to open in Visual Studio 2022." -ForegroundColor Green
if (-not $env:CI) {
    Read-Host "Press Enter to continue..."
}
