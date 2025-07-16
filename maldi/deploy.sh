#!/usr/bin/env sh



rsync -avzP  \
    --rsh="ssh -p2222" \
    --exclude .git/ \
    --exclude .gitignore \
    --exclude TODO \
    ~/Desktop/mLibra/repo/mlibra root@localhost:/myhome/
