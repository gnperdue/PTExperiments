#!/bin/bash

rm -f TODO.md
grep -r TODO * | grep -vi "run_update_todos.sh" > TODO.md
