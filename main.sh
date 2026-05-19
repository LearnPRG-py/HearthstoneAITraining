#!/bin/bash
# main.sh
TOTAL_VIDEOS=109973
NUM_WORKERS=4
CHUNK_SIZE=$((TOTAL_VIDEOS / NUM_WORKERS))

for i in $(seq 0 $((NUM_WORKERS - 1))); do
    START=$((i * CHUNK_SIZE))
    if [ $i -eq $((NUM_WORKERS - 1)) ]; then
        END=$TOTAL_VIDEOS
    else
        END=$(((i + 1) * CHUNK_SIZE))
    fi
    
    bash worker_with_restarts.sh $i $START $END > worker_$i.log 2>&1 &
done

wait
echo "All workers complete!"
