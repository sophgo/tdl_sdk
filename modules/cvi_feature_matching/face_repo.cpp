#include <string.h>
#include "face_repo.h"
#include "cached_repo.hpp"

typedef CachedRepo<512> Repository;

using namespace std;

void face_match(cvi_face_t *meta, char *repo_path, float threshold)
{
    if (!meta->size) {
        return;
    }

    static Repository repo(repo_path);

    for (int i = 0; i < meta->size; ++i) {
        std::vector<float> feature_v;
        feature_v.assign(meta->face_info[i].face_feature, meta->face_info[i].face_feature+NUM_FACE_FEATURE_DIM);

        auto result = repo.match(feature_v, threshold);
        strncpy(meta->face_info[i].name, repo.get_match_name(result).c_str(), sizeof(meta->face_info[i].name));
    }
}

void face_register(cvi_face_t *meta, char *repo_path, char *name)
{
    if (1 != meta->size) {
        cout << "Face count [" << meta->size << "]is not 1." << endl;
        return;
    }

    static Repository repo(repo_path);

    for (int i = 0; i < meta->size; ++i) {
        std::vector<float> feature_v;

        feature_v.assign(meta->face_info[i].face_feature, meta->face_info[i].face_feature+NUM_FACE_FEATURE_DIM);
        repo.add(name, feature_v);
    }
}
