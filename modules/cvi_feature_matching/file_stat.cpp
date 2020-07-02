#include "file_stat.hpp"
#include <vector>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include "path.hpp"

using namespace std;

namespace utils {
namespace file {

bool exists(const string &path) {
    struct stat statbuf;

    return 0 == stat(path.data(), &statbuf);
}

bool is_regular(const string &path) {
    struct stat statbuf;
    if (0 != stat(path.data(), &statbuf)) {
        return false;
    }

    return (statbuf.st_mode & S_IFMT) == S_IFREG;
}

bool is_directory(const string &path) {
    struct stat statbuf;
    if (0 != stat(path.data(), &statbuf)) {
        return false;
    }

    return (statbuf.st_mode & S_IFMT) == S_IFDIR;
}

string content(const string &path, size_t offset, size_t length) {
    vector<char> buf(length);
    FILE *fp = fopen(path.data(), "r");

    fseek(fp, offset, SEEK_SET);
    length = fread(buf.data(), 1, length, fp);
    fclose(fp);

    return string(buf.data(), length);
}

bool remove(const string &path) {
    return (0 == ::remove(path.data()));
}

string directory_name(const string &path) {
    auto slash = path.find_last_of('/');
    if (slash == string::npos) {
        return path;
    } else {
        return path.substr(0, slash + 1);
    }
}

bool mkdir(const string &path) {
    return mkpath(path, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == 0;
}

}
}