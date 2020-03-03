#!/bin/sh

for i in `ls`
do
    if [ -d $i ]
    then
    ./withhist.out $i 100
    fi
done