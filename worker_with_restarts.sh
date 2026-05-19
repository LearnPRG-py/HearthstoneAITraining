#!/bin/bash
# worker_with_restarts.sh

WORKER_ID=$1
START=$2
END=$3
BATCH_SIZE=100

echo "Worker $WORKER_ID: Processing videos $START to $END"

for i in $(seq $START $BATCH_SIZE $END); do
    BATCH_END=$((i + BATCH_SIZE))
    if [ $BATCH_END -gt $END ]; then
        BATCH_END=$END
    fi
    
    echo "Worker $WORKER_ID: Batch $i to $BATCH_END"
    /usr/local/bin/python3 preprocessing.py $i $BATCH_END
    
    # Kill any lingering processes
    pkill -9 -f "preprocessing.py $i"
    sleep 1
done

echo "Worker $WORKER_ID: Complete"
