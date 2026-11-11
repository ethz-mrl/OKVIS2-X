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

#ifndef INCLUDE_OKVIS_OBJECTMAPPING_HPP
#define INCLUDE_OKVIS_OBJECTMAPPING_HPP

#include <glog/logging.h>

#include <okvis/Measurements.hpp>
#include <okvis/assert_macros.hpp>
#include <okvis/kinematics/Transformation.hpp>
#include <okvis/FrameTypedefs.hpp>
#include <okvis/mapTypedefs.hpp>

#include <se/supereight.hpp>
#include <se/external/tinycolormap.hpp>


typedef uint64_t map_id_t;
typedef Eigen::Matrix<float, 1, 768> Descriptor;

namespace okvis{

    struct SegmentMetadata {
        cv::Mat segmentImage;
        int numberPixels;
    };

    struct SubmapObject {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        SubmapObject(se::id_t id,
                     uint64_t map_id,
                     Descriptor vl_feature,
                     size_t num_pixels) : id_(id), map_id_(map_id), vl_feature_(vl_feature), num_pixels_(num_pixels)
        {
            finished_ = false;
        }


        // This extracts per object: centroid position, aabb and number of vertices
        void finishObject(const se::id::IdInfo segmentInfo){
            // ToDo: get map from lookup and extract pose and point cloud
            finished_ = true;
            segmentInfo_ = segmentInfo;
        }

        void updateVisionLanguageDescriptor(const Descriptor& new_vl_feature, size_t num_observed_pixels){
            vl_feature_ = (num_pixels_*vl_feature_ + num_observed_pixels*new_vl_feature) / (num_pixels_ + num_observed_pixels);
            num_pixels_ += num_observed_pixels;
            // ToDo: clamping or similar?
        }

        se::id_t id_; ///< Object ID
        map_id_t map_id_; ///< Id of the submap the object belongs to
        Descriptor vl_feature_; ///< The (averaged) vl_feature in the object masks
        size_t num_pixels_; ///< Number of pixels that were used to compute the average feature
        bool finished_;

        se::id::IdInfo segmentInfo_;

//        Eigen::Isometry3f T_MO_;
//        std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> point_cloud_;
        // ToDo: add const reference to submap / mesh / ... which is needed to extract the pose and point cloud

    };

    struct GlobalObject {

    };

    class ObjectMap {

    public:

        ObjectMap(const AlignedUnorderedMap<uint64_t, se::Submap<SupereightMapType>>& seSubmapLookup,
                    AlignedUnorderedMap<uint64_t, se::TriangleMesh<SupereightMapType::DataType::col_,SupereightMapType::DataType::id_>>& seMeshLookup)
            : seSubmapLookup_(seSubmapLookup), seMeshLookup_(seMeshLookup) { }

        cv::Mat processData(const map_id_t mapId,
                    const cv::Mat& invalid_depth_mask,
                    const okvis::CameraData::DeepLearningImageData& sam_masks,
                    const okvis::CameraData::DeepLearningImageData& clip_features,
                    se::Image<se::id_t>& surface_segment_id,
                    float iou_threshold = 0.25);

        /**
         * Function returning a unique segmentation image with segment IDs to be integrated into supereight
         * @param sam_image
         * @param se2_raycast_image
         */

        cv::Mat trackSegments(const cv::Mat& invalid_depth_mask,
                               const okvis::CameraData::DeepLearningImageData& sam_masks,
                               se::Image<se::id_t>& surface_segment_id,
                               std::set<se::id_t>& matched_segment_ids,
                               const float percentage_allocatable_pixels_sam_segment = 0.002,
                               float iou_threshold = 0.25f);

        void addNewObject(const map_id_t mapId,
                          const cv::Mat& mask,
                          const okvis::CameraData::DeepLearningImageData& clip_features,
                          const se::id_t objectId);

        void updateObject(const map_id_t mapId,
                          const se::id_t segmentId,
                          const cv::Mat& mask,
                          const okvis::CameraData::DeepLearningImageData& clip_features);

        void finishSubmapObjects(const map_id_t mapId);

        bool saveObjects(const map_id_t mapId, const std::string root_path, const Eigen::Affine3f& T_WM);

        se::id_t obtainNewId();

        int numberOfObjects(){
            return numberOfObjects_;
        };

        void extractGlobalObjects(){
            // ToDo: go through all submaps and their objects to do vice-versa check which objects overlap
        }

        bool containsMap(map_id_t map_id) {
            if(submapObjects_.find(map_id) != submapObjects_.end()) {
                return true;
            }
            return false;
        }

        se::TriangleMesh<SupereightMapType::DataType::col_,SupereightMapType::DataType::id_>& mesh(map_id_t map_id){
            return seMeshLookup_.at(map_id); 
        }

        okvis::AlignedMap<se::id_t, SubmapObject>& submapObjects(map_id_t map_id){
            return submapObjects_.at(map_id);
        }

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    private:

        se::id_t objectCounter_ = 0; ///< Object counter to be incremented
        std::map<map_id_t, okvis::AlignedMap<se::id_t, SubmapObject>> submapObjects_; ///< Nested Map to store the Object Instances per Submap
        okvis::AlignedMap<se::id_t , GlobalObject> globalObjects_; ///< Map to store the Object Instances per Submap
        const AlignedUnorderedMap<uint64_t, se::Submap<SupereightMapType>>& seSubmapLookup_; ///< Lookup for se::Submaps as const reference
        AlignedUnorderedMap<uint64_t, se::TriangleMesh<SupereightMapType::DataType::col_,SupereightMapType::DataType::id_>>& seMeshLookup_; ///< Lookup for Meshes for each submap as const reference
        int numberOfObjects_ = 0; /// < Overall Total Number of Objects, always updated when submap is finished

    };

    template<typename FaceT>
    inline void colour_mesh_by_match(
            se::Mesh<FaceT>& mesh,
            const okvis::AlignedMap<se::id_t, SubmapObject>& segment_features,
            const Descriptor& query_feature,
            const tinycolormap::ColormapType colormap = tinycolormap::ColormapType::Parula,
            const float clamp_low = -1.0f,
            const float clamp_high = 1.0f,
            const bool enable_shading = false,
            const Eigen::Vector3f& light_dir_W = Eigen::Vector3f(-1, 0, -1),
            const se::RGB ambient = se::RGB{0x40, 0x40, 0x40})
    {
        const Eigen::Vector3f ambient_light_f(ambient.r, ambient.g, ambient.b);
        for (auto& face : mesh) {
            const auto id = face.id.id;
            if(id == 0 || !segment_features.count(id)) {
                const Eigen::Array3f color(tinycolormap::GetColor(0, colormap).data[0],
                                           tinycolormap::GetColor(0, colormap).data[1],
                                           tinycolormap::GetColor(0, colormap).data[2]);

                face.colour.face = se::RGB{std::uint8_t(UINT8_MAX * color.x()), std::uint8_t(UINT8_MAX * color.y()), std::uint8_t(UINT8_MAX * color.z())};
                continue; // Skip non-instance or not-yet-tracked regions and assign 0 color of colormap.
            }
            const float d = segment_features.at(id).vl_feature_.normalized().dot(query_feature.normalized());
            const float d_clamped = std::clamp(d, clamp_low, clamp_high);
            // Scale from [clamp_low, clamp_high] to [0, 1] inclusive.
            const float d_unit = (d_clamped - clamp_low) / (clamp_high - clamp_low);
            const Eigen::Array3f color(tinycolormap::GetColor(d_unit, colormap).data[0],
                                       tinycolormap::GetColor(d_unit, colormap).data[1],
                                       tinycolormap::GetColor(d_unit, colormap).data[2]);

            face.colour.face = se::RGB{std::uint8_t(UINT8_MAX * color.x()), std::uint8_t(UINT8_MAX * color.y()), std::uint8_t(UINT8_MAX * color.z())};

            if (enable_shading) {
                const Eigen::Vector3f diffuse_colour(
                        face.colour.face.value().r, face.colour.face.value().g, face.colour.face.value().b);
                const Eigen::Vector3f surface_normal_W =
                        se::math::plane_normal(face.vertexes[0], face.vertexes[1], face.vertexes[2]);
                const float intensity = std::max(surface_normal_W.dot(light_dir_W), 0.0f);
                Eigen::Vector3f col = intensity * diffuse_colour + ambient_light_f;
                se::eigen::clamp(col, Eigen::Vector3f::Zero(), Eigen::Vector3f::Constant(255.0f));
                face.colour.face.value().r = col.x();
                face.colour.face.value().g = col.y();
                face.colour.face.value().b = col.z();
            }
        }
    }
    
    template<typename OctreeT>
    inline std::enable_if_t<OctreeT::id_ == se::Id::On && OctreeT::col_ == se::Colour::On, std::map<se::id_t, size_t>>
    voxelsPerId(const OctreeT* submap_octree)
    {
        std::map<se::id_t, size_t> voxels_per_id;
        // IDs aren't integrated in free space so leaf nodes don't need to be considered.
        for (auto it = se::BlocksIterator<const OctreeT>(submap_octree); it != se::BlocksIterator<const OctreeT>(); ++it) {
            typename OctreeT::BlockType& block = *static_cast<typename OctreeT::BlockType*>(*it);
            // Iterate over the block data at the current scale.
            const int scale = block.current_scale;
            const int num_voxels_at_scale = se::math::cu(OctreeT::block_size >> scale);
            const size_t voxel_volume_at_scale = se::math::cu(size_t(se::octantops::scale_to_size(scale)));
            auto* data_at_scale = block.blockDataAtScale(scale);

            for (int voxel_idx = 0; voxel_idx < num_voxels_at_scale; voxel_idx++) {
                se::id_t id = data_at_scale[voxel_idx].id.id;
                // XXX: Hacky implementation until we have a unified API for iterating over voxel data
                // at the current scale.

                if (id != se::g_no_id) {
                    auto [it, _] = voxels_per_id.try_emplace(id, 0u);
                    it->second += voxel_volume_at_scale;
                }
            }
        }
        return voxels_per_id;
    }
}
#endif // INCLUDE_OKVIS_OBJECTMAPPING_HPP
