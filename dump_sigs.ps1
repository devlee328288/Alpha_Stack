cd E:\QUANT\Alpha_Stack
Remove-Item sigs.txt -ErrorAction SilentlyContinue

$files = @(
  'features\model_dataset.py',
  'models\lightgbm.py',
  'models\experiment.py',
  'backtest\backtest_strategies.py',
  'evaluation\evaluation_backtest.py',
  'evaluation\evaluation_risk.py',
  'evaluation\walk_forward.py'
)

foreach ($f in $files) {
  "`n===== $f =====" | Out-File -Append -Encoding utf8 sigs.txt
  Get-Content $f |
    Select-String -Pattern '^\s*(def |class |return |"""|@)' -Context 0,3 |
    Out-String -Width 200 |
    Out-File -Append -Encoding utf8 sigs.txt
}

notepad sigs.txt