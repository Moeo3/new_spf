#!/bin/sh

for i in `ls`
do
    if [ -d $i ]
    then
    ./re.out $i 100
    fi
done