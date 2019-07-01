#!/bin/bash

cut -d ' ' -f 1 ./600h_post.lst > t1
cut -d ' ' -f 4 ./600h_post.lst > t2
cut -d ' ' -f 5 ./600h_post.lst > t3
paste -d ' ' t1 t2 t3  > 600h

