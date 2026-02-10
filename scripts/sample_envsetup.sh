#!/bin/sh
# Description: Environment setup script for TDL SDK samples.
# This script automatically traverses the sample directory to find lib folders
# and appends them to LD_LIBRARY_PATH.

# POSIX compatible way to detect SDK_INSTALL_DIR
# Try to detect where we are running from.
# When sourced in 'sh', $0 might be the shell itself, so we can't rely on it.
# We assume the user sources this script from the SDK install root (install/CV181X) 
# or from the scripts directory.

SDK_INSTALL_DIR=""

if [ -n "$BASH_SOURCE" ]; then
    # If running in bash, use BASH_SOURCE
    SCRIPT_DIR="$(cd "$(dirname "$BASH_SOURCE")" && pwd)"
    SDK_INSTALL_DIR="$(dirname "$SCRIPT_DIR")"
else
    # Fallback for sh/ash/dash
    # Check if we are in the install root (install/CV181X)
    if [ -d "$(pwd)/sample" ] && [ -d "$(pwd)/lib" ]; then
        SDK_INSTALL_DIR="$(pwd)"
    # Check if we are in scripts directory
    elif [ -d "$(pwd)/../sample" ] && [ -d "$(pwd)/../lib" ]; then
        SDK_INSTALL_DIR="$(cd .. && pwd)"
    else
        # Try to infer from $0 if it looks like a script path
        case "$0" in
            *sample_envsetup.sh)
                SCRIPT_PATH="$0"
                 # Handle relative path
                if [ "${SCRIPT_PATH#/}" = "$SCRIPT_PATH" ]; then
                     SCRIPT_PATH="$(pwd)/$SCRIPT_PATH"
                fi
                SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"
                SDK_INSTALL_DIR="$(dirname "$SCRIPT_DIR")"
                ;;
            *)
                # If we can't find it, default to current directory but warn
                SDK_INSTALL_DIR="$(pwd)"
                ;;
        esac
    fi
fi

SAMPLE_DIR="${SDK_INSTALL_DIR}/sample"
MAIN_LIB_DIR="${SDK_INSTALL_DIR}/lib"

# Requirement 7: Check if sample directory exists
if [ ! -d "$SAMPLE_DIR" ]; then
    echo "Error: Sample directory not found at $SAMPLE_DIR"
    echo "Please source this script from the SDK install root (e.g. install/CV181X)"
    return 1 2>/dev/null || exit 1
fi

# Initialize paths variable
PATHS_TO_ADD=""

# Requirement 4: Add main lib directory first
if [ -d "$MAIN_LIB_DIR" ]; then
    PATHS_TO_ADD="$MAIN_LIB_DIR"
else
    echo "Warning: Main lib directory not found at $MAIN_LIB_DIR"
fi

# Requirement 1 & 5: Traverse sample subdirectories for 'lib' folders (including nested ones)
# Requirement 4: Alphabetical order (sort handles this)
# Using find to locate all directories named 'lib' or residing inside a 'lib' directory
FOUND_LIBS=$(find "$SAMPLE_DIR" -type d \( -name "lib" -o -path "*/lib/*" \) | sort)

for lib_path in $FOUND_LIBS; do
    # Requirement 7: Verify existence
    if [ -d "$lib_path" ]; then
        if [ -z "$PATHS_TO_ADD" ]; then
            PATHS_TO_ADD="$lib_path"
        else
            PATHS_TO_ADD="${PATHS_TO_ADD}:${lib_path}"
        fi
    fi
done

# Requirement 2: Append to LD_LIBRARY_PATH
if [ -n "$PATHS_TO_ADD" ]; then
    if [ -z "$LD_LIBRARY_PATH" ]; then
        export LD_LIBRARY_PATH="$PATHS_TO_ADD"
    else
        export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${PATHS_TO_ADD}"
    fi
fi

# Requirement 6: Script output format
# Print the added paths with the specified formatting
if [ -n "$PATHS_TO_ADD" ]; then
    echo "LD_LIBRARY_PATH added:"
    
    # POSIX compatible string splitting using IFS and set
    OLD_IFS="$IFS"
    IFS=':'
    set -- $PATHS_TO_ADD
    
    COUNT=$#
    I=1
    
    for path in "$@"; do
        if [ "$I" -eq "$COUNT" ]; then
            echo "$path"
        else
            echo "$path:\\"
        fi
        I=$((I + 1))
    done
    
    IFS="$OLD_IFS"
fi
