#!/bin/bash
# parallel ranged download: pget.sh URL OUTFILE [NPARTS]
URL=$1; OUT=$2; N=${3:-12}
SIZE=$(curl -sSI -A 'atlasbt-research/0.1' "$URL" | tr -d '\r' | awk 'tolower($1)=="content-length:"{print $2}')
PS=$(( (SIZE + N - 1) / N ))
seq 0 $((N-1)) | xargs -P 8 -I{} bash -c '
  i={}; a=$((i*'$PS')); b=$((a+'$PS'-1)); [ $b -ge '$SIZE' ] && b=$(('$SIZE'-1))
  for k in $(seq 1 15); do
    curl -sS -f -A "atlasbt-research/0.1" -r $a-$b "'$URL'" -o "'$OUT'.part$i" && [ $(stat -c %s "'$OUT'.part$i") -eq $((b-a+1)) ] && exit 0
    sleep $((k*5))
  done; echo "FAILED part $i" >&2; exit 1'
for i in $(seq 0 $((N-1))); do cat "$OUT.part$i" >> "$OUT" && rm "$OUT.part$i"; done
echo "size $(stat -c %s $OUT) expected $SIZE"
