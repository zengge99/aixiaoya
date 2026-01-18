#!/bin/bash

server=http://113.65.22.166:5678
user="guest"
password="guest_Api789"
tasks=(
    "/🏷️我的115分享|115share.txt"
    "/每日更新|daily.txt"
)

prog="$0"
while [ -h "${prog}" ]; do
	newProg=`/bin/ls -ld "${prog}"`

	newProg=`expr "${newProg}" : ".* -> \(.*\)$"`
	if expr "x${newProg}" : 'x/' >/dev/null; then
		prog="${newProg}"
	else
		progdir=`dirname "${prog}"`
		prog="${progdir}/${newProg}"
	fi
done
progdir=`dirname "${prog}"`
cd "${progdir}"

unset LD_PRELOAD
machine=$(uname -m)
if [[ "$machine" == *"arm"* || "$machine" == *"aarch"* ]]; then
    arch="arm64"
else
    arch="amd64"
fi

chmod 755 movie_extractor_linux_$arch
chmod 755 webdav_linux_$arch
chmod 755 getalist_linux_$arch
killall movie_extractor_linux_$arch >/dev/null 2>&1
killall webdav_linux_$arch >/dev/null 2>&1
./movie_extractor_linux_$arch --srv 8889 >/dev/null 2>&1 &
{
	sleep 60
	./webdav_linux_$arch --file "*.txt" --url "$server" --user guest 
}&

for task in "${tasks[@]}"; do
    url="$server${task%%|*}"
    output="${task##*|}"
    echo "正在处理: $url -> $output"
    ./getalist_linux_$arch --url "$url" --output "$output"
done
