#include "path.hpp"
#include <vector>
#include <algorithm>
#include <cstring>
#include <sys/stat.h>

using namespace std;

int mkpath(const string &path, mode_t mode) {
    vector<char> buffer(path.length() + 1);
    size_t offset = 0;
    size_t slash = 0;

    while (offset < path.length()) {
        slash = path.find('/', slash);

        memcpy(buffer.data() + offset, path.data() + offset, min(slash, path.length()) - offset);

        struct stat file_stat;

        if (slash > 0) {
            if (0 != stat(buffer.data(), &file_stat)) {
                if (0 != mkdir(buffer.data(), mode)) {
                    return -1;
                }
            } else if (!S_ISDIR(file_stat.st_mode)) {
                return -1;
            }
        }

        offset = min<size_t>(slash, path.length());
        slash = offset + 1;
    }

    return 0;
}
