import os

os.system('git ls-files --others --exclude-standard > untracked_files.txt')

os.system('tree -ifa . > rls_info.txt')

with open("rls_info.txt", "r") as f:

    lines = f.read().splitlines()


with open("untracked_files.txt", "r") as f:

    untracked_files = f.read().splitlines()

os.remove("./untracked_files.txt")

skip_first_level = [
    "clang",
    ".git",
    "rls_info",
    "tmp",
    "install",
    "regression",
    "README_INTERNAL",
    ".vscode"
]


skip_second_level = [
    "quick_build_tdl",
    "regression",
    "credential",
    "DeepSeek"
]

with open("rls_info.txt", "w") as f:
    f.write(".\n")
    f.write("./.gitignore\n")

    for line in lines[1:-2]:

        if line[2:] in untracked_files:
            continue

        split_info = line.split("/")

        if any(s in split_info[1] for s in skip_first_level):
            continue

        if len((split_info)) > 2 and any(s in split_info[2] for s in skip_second_level):
            continue

        f.write(line + "\n")

print("gen rls_info.txt done!")