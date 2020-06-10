#include "file_utils.hpp"
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace qnn {
namespace utils {

using std::string;
using std::vector;

// file system helpers
string GetBasename(char c, const string &src) {
    size_t found = src.find_last_of(c);
    return src.substr(0, found);
}

string GetFilePostfix(char *path) {
    char *pos = strrchr(path, '.');
    if (pos) {
        string str(pos + 1);
        return str;
    }
    return string("");
}

string GetDirName(const string &src) {
    size_t found = src.find_last_of('/');
    if (found == src.size()) return string("");
    return src.substr(0, found + 1);
}

string ReplaceExt(const string &ext, const string &src) {
    string base = GetBasename('.', src);
    return base + ext;
}

string ReplaceFilename(const string &file, const string &src) {
    string base = GetDirName(src);
    return base + file;
}

string GetFilename(const string &src) {
    size_t p = src.find_last_of('/');
    if (string::npos == p) return src;
    return src.substr(p + 1);
}

string AddFilenamePrefix(const string &src, const string &prefix) {
    return ReplaceFilename(std::string(prefix + GetFilename(src)), src);
}

void DumpDataToFile(const string &name, char *data, int size) {
    std::fstream fp(name, std::ios::out | std::ios::binary);
    fp.write(data, size);
    fp.close();
}

void StringSplit(const string &src, vector<string> *results, const string &c) {
    string::size_type pos1, pos2;
    pos2 = src.find(c);
    pos1 = 0;
    while (string::npos != pos2) {
        results->emplace_back(src.substr(pos1, pos2 - pos1));
        pos1 = pos2 + c.size();
        pos2 = src.find(c, pos1);
    }
    if (pos1 != src.length()) results->emplace_back(src.substr(pos1));
}

std::unique_ptr<char[]> LoadDataFromFile(const string &name, int *len) {
    std::ifstream fp(name, std::ios::in | std::ios::binary);
    if (!fp) return nullptr;
    fp.seekg(0, fp.end);
    int length = fp.tellg();
    fp.seekg(0, fp.beg);
    std::unique_ptr<char[]> buffer(new char[length]);
    fp.read(buffer.get(), length);
    fp.close();
    *len = length;
    return buffer;
}

int FindPic(const string &target_dir, vector<string> &result) {
    DIR *dir;
    struct dirent *filename;
    dir = opendir(target_dir.c_str());
    if (dir == NULL) {
        std::cout << "Open " << target_dir << " failed!" << std::endl;
        return -1;
    }
    bool find_face = false;
    string type1(".jpg"), type2(".JPG"), type3(".png");
    while ((filename = readdir(dir)) != NULL) {
        string face_name(target_dir);
        face_name.append("/");
        face_name.append(filename->d_name);
        if (!FileExist(face_name)) continue;
        if (face_name.find(type1) == string::npos && face_name.find(type2) == string::npos &&
            face_name.find(type3) == string::npos)
            continue;
        result.emplace_back(face_name);
        find_face = true;
    }
    closedir(dir);
    if (!find_face) return -1;
    return 0;
}

bool DirExist(const string &dir_path) {
    struct stat s;
    if (stat(dir_path.c_str(), &s) < 0 || !S_ISDIR(s.st_mode)) {
        return false;
    }
    return true;
}

bool FileExist(const string &file_path) {
    struct stat s;
    if (stat(file_path.c_str(), &s) < 0 || (!S_ISREG(s.st_mode) && !S_ISLNK(s.st_mode))) {
        return false;
    }
    return true;
}

}  // namespace utils
}  // namespace qnn
