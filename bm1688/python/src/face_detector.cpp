#include "face_detector.hpp"

#include <log/Logger.hpp>

#include "common/cv_utils.hpp"
NNFaceDetector::NNFaceDetector(const NNFactory *factory, int device_id /*=0*/,
                               std::string detector_type /*="cssd"*/) {
  if (detector_type == "cssd")
    detector_ = (FaceCSSD *)(factory->get_model(CSSD, device_id));
  else if (detector_type == "scrfd")
    detector_scrfd_ =
        (FaceSCRFD *)factory->get_model(NNBaseModel::SCRFD, device_id);
  landmark_ = (FaceLandmark *)(factory->get_model(DET3, device_id));
}

NNFaceDetector::~NNFaceDetector() {
  // TODO: crashed, when delete; might be freed by fdl
  if (landmark_ != nullptr) {
    delete landmark_;
  }
  if (detector_ != nullptr) {
    delete detector_;
  }
  if (detector_scrfd_ != nullptr) {
    delete detector_scrfd_;
  }
}

py::tuple NNFaceDetector::predict(py::list &inputs,
                                  int detect_landmark /*=1*/) {
  // std::cout << "start to multi predict" << std::endl;
  std::vector<cv::Mat> images;
  for (int i = 0; i < inputs.size(); i++) {
    py::array_t<unsigned char, py::array::c_style> np_image =
        py::cast<py::array_t<unsigned char, py::array::c_style>>(inputs[i]);
    py::buffer_info buf = np_image.request();
    cv::Mat image((int)buf.shape[0], (int)buf.shape[1], CV_8UC3, buf.ptr);
    images.push_back(image);
  }
  std::vector<std::vector<FaceRect>> detecteds;

  if (detector_ != nullptr)
    detector_->detect(images, 0.5, detecteds);
  else if (detector_scrfd_ != nullptr) {
    detector_scrfd_->detect(images, 0.5, detecteds);
  }
  std::cout << "detecteds.bbox" << detecteds[0][0].x1 << " "
            << detecteds[0][0].y1 << " " << detecteds[0][0].x2 << " "
            << detecteds[0][0].y2 << std::endl;
  if (detect_landmark) {
    std::cout << "-----------before landmark-----------" << std::endl;
    predict_face_landmark(images, 0.2, detecteds);
    std::cout << "detecteds.facepts.x.size()"
              << detecteds[0][0].facepts.x.size() << std::endl;
  }
  py::list bboxes_list;
  py::list points_list;
  py::list probs_list;
  py::list scores_list;
  for (int i = 0; i < images.size(); i++) {
    py::list res;
    py::list bboxes;
    py::list points;
    py::list probs;
    py::list scores;
    std::vector<FaceRect> &detected = detecteds[i];

    for (int j = 0; j < detected.size(); j++) {
      py::list bbox;
      FaceRect &rect = detected[j];
      py::list point;

      bbox.append(rect.x1);
      bbox.append(rect.y1);
      bbox.append(rect.x2);
      bbox.append(rect.y2);
      bboxes.append(bbox);
      std::cout << "rect.facepts.x.size()" << rect.facepts.x.size()
                << std::endl;
      if (rect.facepts.x.size() == 5) {
        for (int k = 0; k < 5; ++k) {
          point.append(rect.facepts.x[k]);
        }
        for (int k = 0; k < 5; ++k) {
          point.append(rect.facepts.y[k]);
        }
      } else {
        std::cout << "in else" << rect.facepts.x.size() << std::endl;
        if (detect_landmark) {
          continue;
        }
      }
      points.append(point);
      probs.append(rect.score);

      get_score(rect);
      scores.append(rect.head_score);
      // std::cout << "head_score" << rect.head_score << std::endl;
    }
    bboxes_list.append(bboxes);
    points_list.append(points);
    probs_list.append(probs);
    scores_list.append(scores);
  }

  py::tuple results =
      py::make_tuple(bboxes_list, points_list, probs_list, scores_list);
  // py::tuple results = py::make_tuple(bboxes_list, points_list, probs_list);
  return results;
}

py::tuple NNFaceDetector::predict(
    py::array_t<unsigned char, py::array::c_style> &input,
    int detect_landmark /*=1*/) {
  // std::cout << "start to single predict" << std::endl;
  py::list inputs;
  inputs.append(input);
  py::tuple results = predict(inputs, detect_landmark);
  py::list bboxes_list = results[0];
  py::list points_list = results[1];
  py::list probs_list = results[2];
  py::list scores_list = results[3];
  py::list bboxes = bboxes_list[0];
  py::list points = points_list[0];
  py::list probs = probs_list[0];
  py::list scores = scores_list[0];
  py::tuple result = py::make_tuple(bboxes, points, probs, scores);
  // py::tuple result = py::make_tuple(bboxes, points, probs);
  return result;
}

void NNFaceDetector::predict_face_landmark(
    std::vector<cv::Mat> &images, float ld_threshold,
    std::vector<std::vector<FaceRect>> &detects) {
  std::vector<cv::Mat> crop_faces;
  std::vector<cv::Rect> crop_rects;
  for (int i = 0; i < images.size(); i++) {
    for (auto &rect : detects[i]) {
      int x1 = rect.x1;
      int x2 = rect.x2;
      int y1 = rect.y1;
      int y2 = rect.y2;
      // crop using square box
      int boxw = x2 - x1 + 1;
      int boxh = y2 - y1 + 1;
      int maxwh = std::max(boxw, boxh);
      int square_x = (x1 + x2) / 2 - maxwh / 2;
      int square_y = (y1 + y2) / 2 - maxwh / 2;
      cv::Rect crop_rect(square_x, square_y, maxwh, maxwh);
      cv::Rect inside_rect =
          crop_rect & cv::Rect(0, 0, images[i].cols, images[i].rows);

      cv::Mat crop_face;
      rect.temp_x1 = rect.x1;
      rect.temp_y1 = rect.y1;
      rect.temp_x2 = rect.x2;
      rect.temp_y2 = rect.y2;
      crop_face = images[i](inside_rect);
      crop_rects.push_back(inside_rect);
      crop_faces.push_back(crop_face);
    }
  }

  std::vector<FacePts> facePts;
  landmark_->detect(crop_faces, ld_threshold, facePts);

  size_t lm_idx = 0;
  for (int i = 0; i < images.size(); i++) {
    for (auto &rect : detects[i]) {
      if (facePts[lm_idx].valid()) {
        for (auto &pt_x : facePts[lm_idx].x) {
          pt_x += crop_rects[lm_idx].x;
        }
        for (auto &pt_y : facePts[lm_idx].y) {
          pt_y += crop_rects[lm_idx].y;
        }
        rect.facepts = facePts[lm_idx];
      }
      ++lm_idx;
    }
  }
}

/*
 *
 *
 * */
NNFaceDetMark::NNFaceDetMark(const NNFactory *factory, ModelType m_detector,
                             ModelType m_landmark, int device_id /*=0*/)
    : detector_type(m_detector), landmark_type(m_landmark) {
  p_detector = factory->get_face_detector(m_detector, device_id);
  p_landmark = factory->get_face_landmark(m_landmark, device_id);
  LOG(INFO) << "[NNFaceDetMark] Constructor";
}

NNFaceDetMark::~NNFaceDetMark() {
  delete p_landmark;
  delete p_detector;
  LOG(INFO) << "[NNFaceDetMark] De-constructor";
}

// TODO: export threshold
// py::tuple NNFaceDetMark::predict(py::list& inputs, py::float_&
// detection_threshold, py::float_& point_threshold) {
py::tuple NNFaceDetMark::predict(py::list &inputs) {
  std::vector<cv::Mat> images;
  for (int i = 0; i < inputs.size(); i++) {
    py::array_t<unsigned char, py::array::c_style> np_image =
        py::cast<py::array_t<unsigned char, py::array::c_style>>(inputs[i]);
    py::buffer_info buf = np_image.request();
    cv::Mat image((int)buf.shape[0], (int)buf.shape[1], CV_8UC3, buf.ptr);
    images.push_back(image);
  }

  std::vector<std::vector<FaceRect>> detecteds;
  // fdl_->detectlandmark(images, 0.6, 0.7, detecteds);
  // forward(images, detection_threshold, point_threshold, detecteds);
  forward(images, 0.6, 0.7, detecteds);

  py::list bboxes_list;
  py::list points_list;
  py::list probs_list;
  // py::list scores_list;
  for (int i = 0; i < images.size(); i++) {
    py::list res;
    py::list bboxes;
    py::list points;
    py::list probs;
    // py::list scores;
    std::vector<FaceRect> &detected = detecteds[i];
    for (int j = 0; j < detected.size(); j++) {
      py::list bbox;
      FaceRect &rect = detected[j];
      // bbox.append(rect.temp_x1);
      // bbox.append(rect.temp_y1);
      // bbox.append(rect.temp_x2);
      // bbox.append(rect.temp_y2);

      bbox.append(rect.x1);
      bbox.append(rect.y1);
      bbox.append(rect.x2);
      bbox.append(rect.y2);

      bboxes.append(bbox);
      py::list point;
      for (int k = 0; k < 5; ++k) {
        point.append(rect.facepts.x[k]);
      }
      for (int k = 0; k < 5; ++k) {
        point.append(rect.facepts.y[k]);
      }
      points.append(point);
      probs.append(rect.score);
      // get_score(rect);
      // scores.append(rect.head_score);
    }
    bboxes_list.append(bboxes);
    points_list.append(points);
    probs_list.append(probs);
    // scores_list.append(scores);
  }
  py::tuple results = py::make_tuple(bboxes_list, points_list, probs_list);
  return results;
}

py::tuple NNFaceDetMark::predict(
    py::array_t<unsigned char, py::array::c_style> &input) {
  py::list inputs;
  inputs.append(input);
  py::tuple results = predict(inputs);
  py::list bboxes_list = results[0];
  py::list points_list = results[1];
  py::list probs_list = results[2];
  // py::list scores_list = results[3];
  py::list bboxes = bboxes_list[0];
  py::list points = points_list[0];
  py::list probs = probs_list[0];
  // py::list scores = scores_list[0];
  py::tuple result = py::make_tuple(bboxes, points, probs);
  return result;
}

/*
 * detector_type == NNBaseModel::CSSD && landmark_type == NNBaseModel::BMMARK
 * */
void NNFaceDetMark::forward(std::vector<cv::Mat> &images,
                            float detection_thresold, float point_threshold,
                            std::vector<std::vector<FaceRect>> &faceRects) {
  auto p_cssd = dynamic_cast<FaceCSSD *>(p_detector);
  // auto p_scrfd = dynamic_cast<FaceSCRFD*>(p_detector);
  auto p_bmmark = dynamic_cast<BMMark *>(p_landmark);
  p_cssd->detect(images, detection_thresold, faceRects);
  /*std::vector<std::vector<ObjectBox>> objs;
  p_scrfd->detect(images, detection_thresold, objs);
  std::cout<<"p_scrfd->detect:"<<391<<std::endl;
  for(size_t i = 0; i < objs.size();i++){
    std::vector<FaceRect> imgface;
    for(size_t j = 0; j < objs[i].size();j++){
      FaceRect frct;
      frct.x1 = objs[i][j].x1;
      frct.y1 = objs[i][j].y1;
      frct.x2 = objs[i][j].x2;
      frct.y2 = objs[i][j].y2;
      frct.score = objs[i][j].score;
      imgface.push_back(frct);
    }
    faceRects.push_back(imgface);
  }*/
  std::cout << "faceRects:" << 405 << std::endl;
  std::vector<cv::Mat> roi_list;
  for (int i = 0; i < images.size(); i++) {
    for (auto &rect : faceRects[i]) {
      int x1 = static_cast<int>(rect.x1);
      int x2 = static_cast<int>(rect.x2);
      int y1 = static_cast<int>(rect.y1);
      int y2 = static_cast<int>(rect.y2);
      int bbox_w = x2 - x1 + 1;
      int bbox_h = y2 - y1 + 1;
      cv::Rect bbox_rect(x1, y1, bbox_w, bbox_h);
      bbox_rect = bbox_rect & cv::Rect(0, 0, images[i].cols, images[i].rows);
      cv::Mat roi = images[i](bbox_rect);
      roi_list.push_back(roi);
    }
  }
  std::cout << "roi_list:" << 421 << std::endl;
  std::vector<FacePts> facePts;
  p_bmmark->detect(roi_list, point_threshold, facePts);
  std::cout << "p_bmmark->detect:" << 424 << std::endl;
  // post process
  size_t k = 0;
  for (int i = 0; i < images.size(); i++) {
    for (auto &rect : faceRects[i]) {
      if (facePts[k].valid()) {
        for (auto &pt_x : facePts[k].x) {
          pt_x += rect.x1;
        }
        for (auto &pt_y : facePts[k].y) {
          pt_y += rect.y1;
        }
        rect.facepts = facePts[k];
      }
      k++;
    }
  }
}
