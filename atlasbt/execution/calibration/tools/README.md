Extraction tools used to build `../nbbo_*.csv.gz` and `../iex_*.csv.gz` (see docs/SPREADS.md §2).
Work directory: `$SPREAD_WORK` (default /tmp/spread_work) with `out/` and `dl/`.

    gcc -O2 -o taq taq.c -lm ; gcc -O2 -o iex iex.c -lm
    ./pget.sh "https://ftp.nyse.com/Historical%20Data%20Samples/DAILY%20TAQ/EQY_US_ALL_NBBO_20260701.gz" nbbo.gz 12
    ./pget.sh ".../EQY_US_ALL_TRADE_20260701.gz" trade.gz 8
    ./taq <(zcat nbbo.gz) <(zcat trade.gz) > $SPREAD_WORK/out/taq_20260701.csv     # optional 3rd arg: resume after symbol
    curl -A 'atlasbt-research/0.1' "<IEX HIST TOPS link>" | zcat | ./iex -4 etp.csv > $SPREAD_WORK/out/iex_YYYYMMDD.csv  # -4 EDT / -5 EST
    python3 features.py YYYYMMDD ...     # PIT features from DoltHub ohlcv (dl/ohlcv/ohlcv_2016plus.parquet, dl/dolt_2026/*.csv)
    python3 assemble.py nbbo:20260701 iex:20170322 ...   # -> ../nbbo_*.csv.gz, ../iex_*.csv.gz (needs out/ref_master_20260701.txt)
    python -m atlasbt.execution.calibration.build_calibration --oos-days 20260702 --write
