#!/bin/bash

# Configuration
BINARY="./target/release/rusty-eyes"
ARGS="--remote-dgx http://jowestdgxe:50051"
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

    # Export DISPLAY for GUI on physical monitor (SSH)
    export DISPLAY=:0
    
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
    tail -n 200 -f "$LOG_FILE"
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

function completion() {
    cat <<EOF
_client_sh_completion() {
    local cur prev opts
    COMPREPLY=()
    cur="\${COMP_WORDS[COMP_CWORD]}"
    opts="start startnlog stop restart log status version completion"

    if [[ \${cur} == * ]] ; then
        COMPREPLY=( \$(compgen -W "\${opts}" -- \${cur}) )
        return 0
    fi
}
complete -F _client_sh_completion ./client.sh
complete -F _client_sh_completion client.sh
EOF
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
    check-build)
        if pgrep -f "cargo build" > /dev/null; then
            echo "Build is RUNNING"
            pgrep -fl "cargo build"
        else
            echo "Build is NOT running"
        fi
        ;;
    kill)
        echo "Force killing all client processes..."
        pkill -9 -f "rusty-eyes" || echo "No rusty-eyes process found"
        pkill -9 -f "overlay_linux" || echo "No overlay_linux process found"
        rm -f "$PID_FILE"
        echo "Done."
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
    build-log)
        echo "Tailing build log..."
        tail -n 200 -f build.log
        ;;
    build)
        echo "Starting Build..."
        # Capture both stdout and stderr to build.log
        # Use -j 2 to prevent OOM on Jetson Nano
        ~/.cargo/bin/cargo build --release --no-default-features -j 2 > build.log 2>&1
        echo "Build Complete. Check build.log for details."
        ;;
    version)
        $BINARY --version
        ;;
    completion)
        completion
        ;;
    *)
        echo "Usage: $0 {start|startnlog|stop|kill|restart|log|status|check-build|build-log|version|completion}"
        exit 1
        ;;
esac
