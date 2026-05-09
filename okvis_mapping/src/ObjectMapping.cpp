/**
* This file is part of the FindAnything project, an add-on module for the OKVIS2-X framework.
*
* Copyright (c) 2015, Autonomous Systems Lab / ETH Zurich
* Copyright (c) 2020, Smart Robotics Lab / Imperial College London
* Copyright (c) 2026, Mobile Robotics Lab / Technical University of Munich 
* and ETH Zurich
*
* SPDX-License-Identifier: BSD-3-Clause, see LICENESE file for details
*/

#include <okvis/ObjectMapping.hpp>

#include <opencv2/imgcodecs.hpp> 
#include <okvis/timing/Timer.hpp>

namespace okvis{

    cv::Mat bitwise_and3(const cv::Mat& a, const cv::Mat& b, const cv::Mat& c) {
        assert(a.rows == b.rows);
        assert(a.cols == b.cols);
        assert(a.rows == c.rows);
        assert(a.cols == c.cols);
        assert(a.type() == b.type());
        assert(a.type() == c.type());
        assert(a.type() == CV_8UC1);

        cv::Mat out(a.rows, a.cols, a.type());

        cv::Size size = a.size();
        // If all Mats are continuous we can treat them as 1D arrays.
        if (a.isContinuous() && b.isContinuous() && c.isContinuous() && out.isContinuous()) {
            size.width *= size.height;
            size.height = 1;
        }
        size.width *= sizeof(uint8_t);

        for (int y = 0; y < size.height; y++) {
            const uint8_t* const a_row = a.ptr<uint8_t>(y);
            const uint8_t* const b_row = b.ptr<uint8_t>(y);
            const uint8_t* const c_row = c.ptr<uint8_t>(y);
            uint8_t* const out_row = out.ptr<uint8_t>(y);
            for (int x = 0; x < size.width; x++) {
                out_row[x] = a_row[x] & b_row[x] & c_row[x];
            }
        }
        return out;
    }

    void save_object_bounding_box_vtk(const okvis::SubmapObject& submapObject, std::string filename){

        // Get Maximum and Minimum of active submap
        Eigen::Vector3f map_min = submapObject.segmentInfo_.aabb.min();
        Eigen::Vector3f map_max = submapObject.segmentInfo_.aabb.max();

        // Open the file for writing.
        std::ofstream file(filename.c_str());
        if (!file.is_open()) {
            LOG(ERROR) << "Unable to write file " << filename;
            return;
        }

        // Write the header.
        file << "# vtk DataFile Version 1.0\n";
        file << "Bounding Box of Submap\n";
        file << "ASCII\n";
        file << "DATASET POLYDATA\n";

        file << "POINTS 9 float\n";
        file << map_min.x() << " " << map_min.y() << " " << map_min.z() << "\n"; // 0
        file << map_max.x() << " " << map_min.y() << " " << map_min.z() << "\n"; // 1
        file << map_max.x() << " " << map_max.y() << " " << map_min.z() << "\n"; // 2
        file << map_min.x() << " " << map_max.y() << " " << map_min.z() << "\n"; // 3
        file << map_min.x() << " " << map_min.y() << " " << map_max.z() << "\n"; // 4
        file << map_max.x() << " " << map_min.y() << " " << map_max.z() << "\n"; // 5
        file << map_max.x() << " " << map_max.y() << " " << map_max.z() << "\n"; // 6
        file << map_min.x() << " " << map_max.y() << " " << map_max.z() << "\n"; // 7
        file << submapObject.segmentInfo_.centroid.x() << " "
             << submapObject.segmentInfo_.centroid.y() << " "
             << submapObject.segmentInfo_.centroid.z() << "\n"; // centroid

        file << "LINES 12 36\n";
        file << "2 0 1\n2 1 2\n2 2 3\n2 3 0\n2 4 5\n2 5 6\n2 6 7\n2 7 4\n2 0 4\n2 1 5\n2 2 6\n2 3 7\n";
    }

    /** Convert a single-channel image containing per-pixel segment IDs into a multi-channel image each
     * channel of which contains the mask of a single segment.
     */
    std::map<se::id_t, cv::Mat_<uint8_t>> split(const cv::Mat_<se::id_t>& segments)
    {
        double min, max;
        cv::minMaxLoc(segments, &min, &max);

        std::map<se::id_t, cv::Mat_<uint8_t>> masks;
        for (int y = 0; y < segments.rows; y++) {
            for (int x = 0; x < segments.cols; x++) {
                const auto id = segments(y, x);
                if (id != se::g_no_id) {
                    const auto [it, _] = masks.try_emplace(id, segments.rows, segments.cols, 0);
                    it->second(y, x) = 1u;
                }
            }
        }
        return masks;
    }

    std::map<se::id_t, cv::Mat_<uint8_t>> split(se::Image<se::id_t>& segment_id_image)
    {
        return split(cv::Mat_<se::id_t>(
                segment_id_image.height(), segment_id_image.width(), segment_id_image.data()));
    }

    /**
     * Compute average feature descriptor given a clip image (height x width x channels) and a mask
     */
    size_t averageDescriptor(const okvis::CameraData::DeepLearningImageData& clip_image,
                             const cv::Mat& mask,
                             Descriptor& average_descriptor){

        int64_t height = clip_image.shape[0];
        int64_t width = clip_image.shape[1];
        average_descriptor.setZero();

        size_t num_evaluated_pixels = 0;

        int mask_width = mask.cols;
        int mask_height = mask.rows;

        for(int v = 0; v < mask_height; v++) {
            for(int u = 0; u < mask_width; u++) {
                if(mask.at<uchar>(v, u) == 0) {
                    continue;
                }

                int u_clip = float(u) / mask.cols * width;
                int v_clip = float(v) / mask.rows * height;

                num_evaluated_pixels++;  
                const Descriptor* const pixel_descriptor_ptr = reinterpret_cast<const Descriptor*>(clip_image.data_ptr.get()) + u_clip + v_clip * width;
                average_descriptor += *pixel_descriptor_ptr;
            }
        }
        // Average descriptor over evaluated pixels
        if(num_evaluated_pixels == 0) {
            return 0;
        }
        average_descriptor /= num_evaluated_pixels;

        return num_evaluated_pixels;
    }

    cv::Mat ObjectMap::processData(const map_id_t mapId,
                                     const cv::Mat& invalid_depth_mask,
                                     const okvis::CameraData::DeepLearningImageData& sam_masks,
                                     const okvis::CameraData::DeepLearningImageData& clip_features,
                                     se::Image<se::id_t>& surface_segment_id,
                                     float iou_threshold) {
        
        std::set<se::id_t> matched_segment_ids;
        TimerSwitchable trackSegmentsTimer("Segment tracking");
        cv::Mat matchedSegments = trackSegments(invalid_depth_mask, sam_masks, surface_segment_id, matched_segment_ids, 0.00049, iou_threshold);
        trackSegmentsTimer.stop();
        
        TimerSwitchable objManaging("Object managing");
        for(const auto id : matched_segment_ids) {
            auto it = submapObjects_.find(mapId);
            cv::Mat clipped_segment_mask = matchedSegments == id;
            if(it == submapObjects_.end()) {
                addNewObject(mapId, clipped_segment_mask, clip_features, id);
            } else if (it->second.find(id) == it->second.end()) {
                addNewObject(mapId, clipped_segment_mask, clip_features, id);
            } else {
                updateObject(mapId, id, clipped_segment_mask, clip_features);
            }
        }
        objManaging.stop();

        return matchedSegments;
    }

    cv::Mat ObjectMap::trackSegments(const cv::Mat& invalid_depth_mask,
                                      const okvis::CameraData::DeepLearningImageData& sam_masks,
                                      se::Image<se::id_t>& surface_segment_id,
                                      std::set<se::id_t>& matched_segment_ids,
                                      const float percentage_allocatable_pixels_sam_segment,
                                      float iou_threshold) {

        // Retrieve SE2 Raycast Masks
        // Obtain Per-Segment Masks for every segment ID in surface_segment_id Image
        std::map<se::id_t, cv::Mat_<uint8_t>> raycasted_segment_masks = split(surface_segment_id);

        // Now, retrieve information from current SAM segmentation
        if(sam_masks.shape.size() != 3 || sam_masks.shape[0] == 0 || !sam_masks.data_ptr) {
            return cv::Mat(invalid_depth_mask.rows, invalid_depth_mask.cols, CV_16UC1, cv::Scalar(se::g_no_id));
        }
        int64_t num_masks = sam_masks.shape[0];
        int64_t height = sam_masks.shape[1];
        int64_t width = sam_masks.shape[2];

        cv::Mat unique_segments(invalid_depth_mask.rows, invalid_depth_mask.cols, CV_16UC1, cv::Scalar(se::g_no_id));
        cv::Mat assignable_pixels(invalid_depth_mask.rows, invalid_depth_mask.cols, CV_8UC1, cv::Scalar(1));

        if(raycasted_segment_masks.size() == 0) {
            return unique_segments;
        }
        
        bool resize_sam = false;
        if(invalid_depth_mask.rows != height || invalid_depth_mask.cols != width) {
            resize_sam = true;
        }
        

        // Compute Ratio between SAM dimensions and clip dimensions
        // int64_t clip_height = clip_features.shape[0];
        // int64_t clip_width = clip_features.shape[1];
        // int64_t clip_channels = clip_features.shape[2];

        // int64_t ratio = height / clip_height;
        //assert((height % clip_height) == 0);
        // ToDo: Assert or exception if not truely divisible
        // Sort Masks according to size

        auto it = raycasted_segment_masks.find(se::g_not_mapped);
        cv::Mat unmapped_mask;
        if(it == raycasted_segment_masks.end()) {
            unmapped_mask = cv::Mat::zeros(invalid_depth_mask.rows, invalid_depth_mask.cols, CV_8UC1);
        }
        else {
            unmapped_mask = it->second;
            raycasted_segment_masks.erase(it);
        }

        const int num_min_pixels_sam_segment = int(percentage_allocatable_pixels_sam_segment * invalid_depth_mask.cols * invalid_depth_mask.rows);

        // (1) SAM masks
        std::vector<SegmentMetadata> sam_mask_sizes; // (mask, size)
        for(int64_t i = 0; i < num_masks; i++) {

            // Get mask from raw pointer first
            float* segment_ptr = sam_masks.data_ptr.get() + i * height * width;
            cv::Mat segment_mask(height, width, CV_32FC1, segment_ptr);
            const int num_pixels_per_mask_orig = cv::countNonZero(segment_mask);
            segment_mask.convertTo(segment_mask, CV_8UC1);
            // Mask out invalid depth

            if(resize_sam) {
                //We need to have the SAM masks and the SE2 masks having the same size for the IoU checks
                cv::resize(segment_mask, segment_mask, cv::Size(invalid_depth_mask.cols, invalid_depth_mask.rows));
            }

            segment_mask.setTo(cv::Scalar(0), invalid_depth_mask);

            // Reject too small segments or segments that are in the border of the near and far plane of integration
            //Nonetheless, if the segments are too big (e.g. the floor might have 50% of the segment being allocatable, but it is a segment we want to have and that is very big)
            const int num_pixels_per_mask = cv::countNonZero(segment_mask);
            float div = num_pixels_per_mask / float(num_pixels_per_mask_orig);
            if(num_pixels_per_mask < num_min_pixels_sam_segment || (div < 0.9 &&  num_pixels_per_mask < 10 * num_min_pixels_sam_segment))
            {
                continue;
            }

            sam_mask_sizes.emplace_back(SegmentMetadata{segment_mask, num_pixels_per_mask});
        }

        // Sort w.r.t. non-zero number (size of the mask)
        std::sort(sam_mask_sizes.begin(), sam_mask_sizes.end(),
                  [](const SegmentMetadata sam_mask1, const SegmentMetadata sam_mask2)
                  { return sam_mask1.numberPixels < sam_mask2.numberPixels; });

        // (2) SE2 masks
        std::vector<std::pair<se::id_t, size_t>> se2_mask_sizes;
        for(const auto& raycast_mask : raycasted_segment_masks) {
            const size_t nz = cv::countNonZero(raycast_mask.second);
            se2_mask_sizes.push_back(std::pair<se::id_t, size_t>(raycast_mask.first,nz));
        }
        // Sort w.r.t. non-zero number (size of the mask)
        std::sort(se2_mask_sizes.begin(), se2_mask_sizes.end(),
                  [](std::pair<se::id_t, size_t> se2_mask1,std::pair<se::id_t, size_t> se2_mask2)
                    { return se2_mask1.second < se2_mask2.second; });

        const float alfa = 1.2;
        const float oversegmentation_threshold = 0.8;

        //NOTE: Simon Boche, the update will be done in a separate function, this helps to unit test this function
        TimerSwitchable allSegmentTimer("All SAM segment tracking");

        cv::Mat intersection;
        for(size_t i = 0; i < sam_mask_sizes.size(); ++i) {
            TimerSwitchable perSegmentTimer("Per SAM segment tracking");
            const cv::Mat& sam_mask = sam_mask_sizes[i].segmentImage;
            cv::Mat sam_mask_se_allocated = sam_mask.clone();
            sam_mask_se_allocated.setTo(cv::Scalar(se::g_no_id), unmapped_mask);

            const int sam_mask_num_pixels = sam_mask_sizes[i].numberPixels;
            std::pair<se::id_t, float> max_IoU = {-1, 0.f};
            for(size_t j = 0; j < se2_mask_sizes.size(); j++) {
                cv::Mat& supereight_mask_id = raycasted_segment_masks.at(se2_mask_sizes[j].first);
                const int supereight_mask_id_area = se2_mask_sizes[j].second;
                intersection.create(supereight_mask_id.size(), supereight_mask_id.type());
                cv::bitwise_and(supereight_mask_id, sam_mask, intersection);
                const size_t numberIntersectingPixels =  cv::countNonZero(intersection);
                if(supereight_mask_id_area < sam_mask_num_pixels * alfa) {
                    //If Se2 segment is smaller than SAM, we try to oversegment our SAM segment given prior segmentation hypothesis
                    //Compute percentage of sam mask in a supereight mask
                    const float se_segment_overlap = numberIntersectingPixels / float(supereight_mask_id_area);
                    if(se_segment_overlap > oversegmentation_threshold) {
                        //We add this id to the final image in the areas which are unallocated yet
                        cv::Mat matched_pixels = bitwise_and3(supereight_mask_id, sam_mask, assignable_pixels);
                        // cv::bitwise_and(supereight_mask_id, sam_mask, matched_pixels);
                        // cv::bitwise_and(matched_pixels, assignable_pixels, matched_pixels);
                        unique_segments.setTo(cv::Scalar(se2_mask_sizes[j].first), matched_pixels);
                        assignable_pixels.setTo(cv::Scalar(0), matched_pixels);
                        matched_segment_ids.insert(se2_mask_sizes[j].first);
                    }
                }
 
                //TimerSwitchable iouComputation("IoU computation");
                //Compute IoU
                const float segment_IoU = float(numberIntersectingPixels) / (sam_mask_num_pixels + supereight_mask_id_area - numberIntersectingPixels);
                if(segment_IoU > max_IoU.second) {
                    max_IoU.first = se2_mask_sizes[j].first;
                    max_IoU.second = segment_IoU;
                }
                //iouComputation.stop();
            }

            //Here we extend to unallocated areas of the segment
            //TimerSwitchable leftoverSetting("Leftover setting");
            cv::Mat matched_pixels;
            cv::bitwise_and(sam_mask, assignable_pixels, matched_pixels);
            assignable_pixels.setTo(cv::Scalar(0), matched_pixels);
            int num_assignable_pixels = cv::countNonZero(matched_pixels);
            if(num_assignable_pixels > 0) {
                if(iou_threshold < max_IoU.second) {
                    matched_segment_ids.insert(max_IoU.first);
                    unique_segments.setTo(max_IoU.first, matched_pixels);
                } else {
                    se::id_t newId = obtainNewId();
                    matched_segment_ids.insert(newId);
                    unique_segments.setTo(cv::Scalar(newId), matched_pixels);
                }
            }
            //leftoverSetting.stop();
            perSegmentTimer.stop();
        }
        allSegmentTimer.stop();
        return unique_segments;

    }

    void ObjectMap::finishSubmapObjects(const map_id_t mapId)
    {
        #ifdef OKVIS_COLIDMAP
            if(submapObjects_.count(mapId) == 0) {
                LOG(ERROR) << "Trying to finish Objects for a non-existing map!";
                return;
            }

            DLOG(INFO) << "Completed submap " << mapId;
            Descriptor query(chair_embedding.data());

            const auto voxels_per_segment = voxelsPerId(&seSubmapLookup_.at(mapId).map->getOctree());
            std::ofstream submap_segment_histogram("/tmp/hist-map-" + std::to_string(mapId) + ".tsv");
            submap_segment_histogram << "Segment ID\tVoxels\n";
            for (const auto& p : voxels_per_segment) {
                submap_segment_histogram << p.first << "\t" << p.second << "\n";
            }
            submap_segment_histogram.close();
            colour_mesh_by_match(seMeshLookup_[mapId], submapObjects_[mapId], query, tinycolormap::ColormapType::Jet, 0.0, 0.5);
            // Get Segment Info for all segments in the map
            std::map<se::id_t, se::id::IdInfo> object_info = se::id::mesh_id_info(seMeshLookup_[mapId]);
        

            // Now iterate book-kept objects of the current map
            DLOG(INFO) << submapObjects_[mapId].size() << " objects exist in map " << mapId;

            for (auto object_iterator = submapObjects_[mapId].begin(); object_iterator != submapObjects_[mapId].end(); ) {
                auto object_info_it = object_info.find(object_iterator->first);
                if(object_info_it == object_info.end() ){
                    DLOG(INFO) << "Removing object " << object_iterator->first;
                    object_iterator = submapObjects_[mapId].erase(object_iterator);
                }
                else {
                    object_iterator->second.finishObject(object_info_it->second);
                    object_iterator++;
                }
            }

            DLOG(INFO) << submapObjects_[mapId].size() << " objects exist in map after erasing " << mapId;
            numberOfObjects_ += submapObjects_[mapId].size();
            DLOG(INFO) << "Global Number of objects updated to " << numberOfObjects_;


            //        for(auto& submapObject : submapObjects_[mapId]) {
            //            // Check if we have segment info for that
            //            LOG(INFO) << "Checking Segment " << submapObject.first;
            //            if(object_info.find(submapObject.first) != object_info.end()){
            //                // This extracts per object: centroid position, aabb and number of vertices
            //                submapObject.second.finishObject(object_info[submapObject.first]);
            ////                save_object_bounding_box_vtk(submapObject.second, "/tmp/dbg-map"+std::to_string(mapId)+"-segment"+std::to_string(submapObject.first)+".vtk");
            ////                LOG(INFO) << "Retrieved info for object " << submapObject.first;
            ////                LOG(INFO) << "--- position: " << submapObject.second.segmentInfo_.centroid.transpose();
            ////                LOG(INFO) << "--- num_vertices: " << submapObject.second.segmentInfo_.num_vertices;
            ////                LOG(INFO) << "--- aabb: [" << submapObject.second.segmentInfo_.aabb.min() << "] [" << submapObject.second.segmentInfo_.aabb.max() << "]";
            //            }
            //        }
        #else
            LOG(WARNING) << "The Supereight2 map type does not support semantics and colour, this function does an empty return. Please check mapTypedefs.hpp";
        #endif
    }

    se::id_t ObjectMap::obtainNewId(){
        objectCounter_++;
        return objectCounter_;
    }

    void ObjectMap::addNewObject(const map_id_t mapId,
                                 const cv::Mat& mask,
                                 const okvis::CameraData::DeepLearningImageData& clip_features,
                                 const se::id_t objectId){

        if(seSubmapLookup_.count(mapId) == 0) {
            LOG(ERROR) << "Trying to Create New Object for a non-existing map ID" << mapId << "!";
            return;
        }

        Descriptor vl_feature;
        size_t num_evaluated_pixels = averageDescriptor(clip_features, mask, vl_feature);
        if(num_evaluated_pixels == 0) {
            return;
        }
        submapObjects_[mapId].emplace(std::make_pair(objectId, SubmapObject(objectId, mapId, vl_feature, num_evaluated_pixels)));
    }

    void ObjectMap::updateObject(const map_id_t mapId,
                                 const se::id_t segmentId,
                                 const cv::Mat& mask,
                                 const okvis::CameraData::DeepLearningImageData& clip_features) {

        if(submapObjects_.count(mapId) == 0) {
            LOG(ERROR) << "Trying to update Objects for a non-existing map!";
            return;
        }
        if(submapObjects_.at(mapId).count(segmentId) == 0) {
            LOG(ERROR) << "Trying to update non-existing Object with id: " << segmentId << " in map Id " << mapId;
            return;
        }

        Descriptor vl_feature;
        size_t num_evaluated_pixels = averageDescriptor(clip_features, mask, vl_feature);
        if(num_evaluated_pixels == 0) {
            return;
        }

        submapObjects_.at(mapId).at(segmentId).updateVisionLanguageDescriptor(vl_feature, num_evaluated_pixels);

    }


    bool ObjectMap::saveObjects(const map_id_t mapId, const std::string root_path, const Eigen::Affine3f& T_WM){
        
        if(submapObjects_.find(mapId) == submapObjects_.end()) {
            return false;
        }
        std::fstream csvFile((root_path + "/objectMap" + std::to_string(mapId) + ".csv").c_str(), std::ios_base::out);
        bool success = csvFile.good();
        if (!success) {
            return false;
        }

        std::string header = "ObjectId, num_vert, c_x, c_y, c_z";

        for(int i = 0; i < 8; i++) {
            std::string temp = ", aabb_" + std::to_string(i);
            header += temp + "_x" + temp + "_y" + temp + "_z"; 
        }

        for(size_t i = 0; i < Descriptor::ColsAtCompileTime; i++) {
            header += ", f" + std::to_string(i);
        }

        csvFile << header << std::endl;

        for(const auto& object : submapObjects_[mapId]){
            std::string object_data = std::to_string(object.first) + ", " + std::to_string(object.second.segmentInfo_.num_vertices);
            Eigen::Vector3f centroid_W = T_WM.linear() * object.second.segmentInfo_.centroid + T_WM.translation();
            for(int i = 0; i < 3; i++) {
                object_data += ", " + std::to_string(centroid_W(i));
            }

            std::vector<Eigen::AlignedBox3f::CornerType> corners = {Eigen::AlignedBox3f::CornerType::BottomLeft, 
                                                                    Eigen::AlignedBox3f::CornerType::BottomRight, 
                                                                    Eigen::AlignedBox3f::CornerType::TopLeft, 
                                                                    Eigen::AlignedBox3f::CornerType::TopRight, 
                                                                    Eigen::AlignedBox3f::CornerType::BottomLeftCeil, 
                                                                    Eigen::AlignedBox3f::CornerType::BottomRightCeil, 
                                                                    Eigen::AlignedBox3f::CornerType::TopLeftCeil, 
                                                                    Eigen::AlignedBox3f::CornerType::TopRightCeil};

            for(auto& corner_enum : corners) {
              auto corner_W = T_WM * object.second.segmentInfo_.aabb.corner(corner_enum);
              for(int j = 0; j < corner_W.rows(); j++) {
                object_data += ", " + std::to_string(corner_W(j));
              }
            }

            for(int i = 0; i < object.second.vl_feature_.cols(); i++) {
                object_data += ", " + std::to_string(object.second.vl_feature_(i));
            }

            csvFile << object_data << std::endl;
        }
        

        return true;
    }

}
