#!/bin/bash
# Sync clocks on Jetson and DGX
# Usage: ./scripts/sync_clocks.sh [NTP_SERVER]

NTP_SERVER=${1:-"pool.ntp.org"}

echo "=== Syncing Clocks ==="
echo "Target NTP: $NTP_SERVER"

# 1. Jetson
echo ""
echo "--- Syncing Jetson (jetsone) ---"
ssh -t jetsone "sudo timedatectl set-ntp false && sudo date && sudo ntpdate -u $NTP_SERVER && sudo timedatectl set-ntp true && date"

# 2. DGX
echo ""
echo "--- Syncing DGX (jowestdgxe) ---"
ssh -t jowestdgxe "sudo timedatectl set-ntp false && sudo date && sudo ntpdate -u $NTP_SERVER && sudo timedatectl set-ntp true && date"

echo ""
echo "=== Done ==="
