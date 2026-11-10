$here = Split-Path -Path $MyInvocation.MyCommand.Definition -Parent
Set-Location $here
python .\run_app.py
