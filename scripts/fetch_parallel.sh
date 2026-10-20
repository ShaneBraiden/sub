#!/usr/bin/env bash
# Parallel, resumable HTTP range download (the network here is slow per connection and resets
# long transfers). Each part is fetched with its own retry loop; parts are concatenated once all
# are complete. Safe to re-run after an interruption:
#   * bytes are appended only when the server honoured the range (HTTP 206), so a mirror or proxy
#     answering with the whole file or an error page cannot corrupt a part;
#   * every run writes a new owner token and each part loop checks it before appending, so loops
#     left over from an interrupted run (killing the parent shell does not kill its background
#     subshells on Windows) stop instead of writing into the same part.
#   bash scripts/fetch_parallel.sh <out-file> <size-bytes> <parts> <url> [<mirror-url> ...]
set -u
out=$1; size=$2; parts=$3; shift 3
urls=("$@")
dir="$out.parts"
mkdir -p "$dir"
chunk=$(( (size + parts - 1) / parts ))
run="$$-$RANDOM"
echo "$run" > "$dir/owner"

fetch_part() {
    local k=$1 start=$(( $1 * chunk )) end=$(( ($1 + 1) * chunk - 1 ))
    (( end >= size )) && end=$(( size - 1 ))
    local want=$(( end - start + 1 )) f="$dir/part$k" tmp="$dir/part$k.$run.tmp" try=0
    while :; do
        [[ $(cat "$dir/owner" 2>/dev/null) == "$run" ]] || { echo "part $k: superseded by a newer run"; return 1; }
        local have=0
        [[ -f $f ]] && have=$(stat -c %s "$f")
        (( have > want )) && truncate -s "$want" "$f"
        (( have >= want )) && { echo "part $k done"; return 0; }
        local url=${urls[$(( (k + try) % ${#urls[@]} ))]}
        local code
        code=$(curl -s -S -L --fail --connect-timeout 30 --speed-time 60 --speed-limit 20000 \
               -r $(( start + have ))-$end -o "$tmp" -w "%{http_code}" "$url")
        if [[ $code == 206 && -s $tmp && $(cat "$dir/owner" 2>/dev/null) == "$run" ]]; then
            cat "$tmp" >> "$f"
        else
            sleep 10                     # failed / range ignored / DNS or link outage
        fi
        rm -f "$tmp"
        try=$(( try + 1 ))
        (( try > 3000 )) && { echo "part $k FAILED"; return 1; }
    done
}

for (( k = 0; k < parts; k++ )); do fetch_part "$k" & done
wait
total=0
for (( k = 0; k < parts; k++ )); do total=$(( total + $(stat -c %s "$dir/part$k") )); done
if (( total != size )); then echo "size mismatch: $total != $size"; exit 1; fi
: > "$out"
for (( k = 0; k < parts; k++ )); do cat "$dir/part$k" >> "$out"; done
rm -rf "$dir"
if [[ $out == *.gz ]] && ! gzip -t "$out"; then echo "gzip check FAILED for $out"; exit 1; fi
echo "DONE $out ($size bytes)"
