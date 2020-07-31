# For normal CI check
init || exit 1

# For daily build tests
if [[ "$1" != "daily" ]]; then
  exit 0
fi