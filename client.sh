#!/bin/bash

# Configuration
BINARY="./target/release/rusty-eyes"
ARGS="--config config.toml --remote-dgx-url http://jowestdgxe:50051"
LOG_FILE="client.log"
PID_FILE=".client.pid"

function start() {
    if [ -f "$PID_FILE" ]; then
        if ps -p $(cat "$PID_FILE") > /dev/null; then
            echo "Client is already running (PID: $(cat $PID_FILE))."
            return
        fi
    fi

    echo "Starting Rusty Eyes Client..."
    echo "Command: $BINARY $ARGS"
    
    nohup $BINARY $ARGS > "$LOG_FILE" 2>&1 &
    PID=$!
    echo $PID > "$PID_FILE"
    
    echo "Started with PID $PID. Logs redirected to $LOG_FILE."
    echo "Run './client.sh log' to view logs."
}

function stop() {
    if [ ! -f "$PID_FILE" ]; then
        echo "No PID file found. Is the client running?"
        return
    fi
    
    PID=$(cat "$PID_FILE")
    if ps -p $PID > /dev/null; then
        echo "Stopping client (PID: $PID)..."
        kill $PID
        # Wait for it to exit
        sleep 1
        if ps -p $PID > /dev/null; then
             echo "Force killing..."
             kill -9 $PID
        fi
        rm "$PID_FILE"
        echo "Stopped."
    else
        echo "Client process $PID not found. Cleaning up PID file."
        rm "$PID_FILE"
    fi
}

function log() {
    echo "Tailing log file: $LOG_FILE"
    tail -f "$LOG_FILE"
}

function status() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p $PID > /dev/null; then
            echo "Client is RUNNING (PID: $PID)"
        else
            echo "Client is STOPPED (PID file exists but process passed away)"
        fi
    else
        echo "Client is STOPPED"
    fi
}

case "$1" in
    start)
        start
        ;;
    startnlog)
        start
        sleep 1
        log
        ;;
    stop)
        stop
        ;;
    restart)
        stop
        sleep 1
        start
        ;;
    log|logs)
        log
        ;;
    status)
        status
        ;;
    *)
    *)
        echo "Usage: $0 {start|startnlog|stop|restart|log|status}"
        exit 1
        ;;
esac
