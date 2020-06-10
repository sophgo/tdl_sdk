// Copyright 2018 Bitmain Inc.
// License
// Author Yangwen Huang <yangwen.huang@bitmain.com>

#include "anti_facespoofing/afs_utils.hpp"

namespace qnn {
namespace vision {

void TrackerQueue::GetTrakerResult(const cv::Rect &roi, const short &working_filter_num,
                                   const TrackerFilterVec filter_values, bool &is_real,
                                   bool use_simple) {
    is_real = false;
    // Use simple method
    if (use_simple) {
        ushort counter = 0;
        for (size_t i = 0; i < filter_values.size(); i++) {
            counter += 1 * (ushort)filter_values[i].second;
        }

        if (counter > ((float)working_filter_num / 2)) is_real = true;
        return;
    }

    // Use queue method
    bool is_pushed = false;
    for (auto it = m_trackers.begin(); it != m_trackers.end();) {
        float &&iou_val = (float)(it->region & roi).area() / (it->region | roi).area();
        it->life++;
        if (iou_val > m_iou_threshold) {
            // Update queue if valid
            is_pushed = true;
            it->region = roi;
            it->dequeue.push_back(filter_values);
            it->life = 0;

            // Skip if the queue is too short
            if (it->dequeue.size() < m_queue_decision_length) continue;

            // Erase outdated TrackerFilterVec elements
            int pop_length = (int)it->dequeue.size() - m_queue_decision_length;
            if (pop_length > 0) {
                it->dequeue.erase(it->dequeue.begin(), it->dequeue.begin() + pop_length);
            }

            // Voting by TrackerFilterVec info
            ushort pos = 0;
            for (TrackerFilterVec of_val : it->dequeue) {
                ushort counter = 0;
                for (size_t j = 0; j < filter_values.size(); j++) {
                    counter += 1 * (ushort)filter_values[j].second;
                }

                if (counter > ((float)working_filter_num / 2)) pos++;
            }

            if (pos > ((ushort)it->dequeue.size()) / 2) {
                is_real = true;
            }
        } else {
            // Erase outdated m_trackers queue elements
            if (it->life > m_outdated_threshold) {
                it = m_trackers.erase(it);
                continue;
            }
        }
        ++it;
    }
    // Create a new tracker queue if none of the existing queue matches
    if (!is_pushed) {
        TrackerQueueElements ele(roi, filter_values);
        m_trackers.push_back(ele);
    }
}
}  // namespace vision
}  // namespace qnn