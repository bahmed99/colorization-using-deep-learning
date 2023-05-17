#!/bin/bash

./run_client.sh &
# ./run_client.sh >/dev/null 2>/dev/null &

./run_server.sh &
# ./run_server.sh >/dev/null 2>/dev/null &

trap 'exit 0' TERM
trap 'kill -- -$$' INT

read -d ''
