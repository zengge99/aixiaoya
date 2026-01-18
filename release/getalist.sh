#!/bin/bash

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
echo "arch:$arch"
chmod 755 movie_extractor_linux_$arch
./movie_extractor_linux_$arch --srv 8889
