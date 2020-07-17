#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CLANG_ROOT=$(readlink -f $SCRIPT_DIR)

find $CLANG_ROOT/include -regex '.*\.\(cpp\|h\|hpp\|cc\|c\|cxx\|inc\)' | xargs clang-format-5.0 -i || exit 1
find $CLANG_ROOT/modules -regex '.*\.\(cpp\|h\|hpp\|cc\|c\|cxx\|inc\)' | xargs clang-format-5.0 -i || exit 1
find $CLANG_ROOT/sample -regex '.*\.\(cpp\|h\|hpp\|cc\|c\|cxx\|inc\)' | xargs clang-format-5.0 -i || exit 1