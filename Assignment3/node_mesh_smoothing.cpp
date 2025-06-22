#include <pxr/base/vt/array.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <set>
#include <vector>

#include "GCore/Components/MeshOperand.h"
#include "GCore/GOP.h"
#include "OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh"
#include "geom_node_base.h"
#include "nodes/core/def/node_def.hpp"

typedef OpenMesh::TriMesh_ArrayKernelT<> MyMesh;
using Point = MyMesh::Point;
using Normal = MyMesh::Normal;
using VertexHandle = MyMesh::VertexHandle;
using FaceHandle = MyMesh::FaceHandle;
using HalfedgeHandle = MyMesh::HalfedgeHandle;

typedef enum { kEdgeBased, kVertexBased } FaceNeighborType;

// 计算每个面的面积
void getFaceArea(MyMesh &mesh, std::vector<float> &areas)
{
    areas.resize(mesh.n_faces());
    for (const auto &face_handle : mesh.faces()) {
        areas[face_handle.idx()] = mesh.calc_face_area(face_handle);
    }
}

// 计算每个面的质心
void getFaceCentroid(MyMesh &mesh, std::vector<MyMesh::Point> &centroids)
{
    centroids.resize(mesh.n_faces());
    for (const auto &face_handle : mesh.faces()) {
        centroids[face_handle.idx()] = mesh.calc_face_centroid(face_handle);
    }
}

// 计算每个面的单位法向量
void getFaceNormal(MyMesh &mesh, std::vector<MyMesh::Normal> &normals)
{
    mesh.request_face_normals();
    mesh.update_normals();
    normals.resize(mesh.n_faces());
    for (const auto &face_handle : mesh.faces()) {
        normals[face_handle.idx()] = mesh.normal(face_handle).normalized();
    }
}

// 获取单个面的边邻接面
void getFaceNeighbor(
    MyMesh &mesh,
    MyMesh::FaceHandle fh,
    std::vector<MyMesh::FaceHandle> &face_neighbor)
{
    face_neighbor.clear();
    for (auto he_it = mesh.fh_iter(fh); he_it.is_valid(); ++he_it) {
        auto opposite_heh = mesh.opposite_halfedge_handle(*he_it);
        if (!mesh.is_boundary(opposite_heh)) {
            face_neighbor.push_back(mesh.face_handle(opposite_heh));
        }
    }
}

// 获取所有面的邻接面列表
void getAllFaceNeighbor(
    MyMesh &mesh,
    std::vector<std::vector<MyMesh::FaceHandle>> &all_face_neighbor,
    bool include_central_face)
{
    all_face_neighbor.resize(mesh.n_faces());
    for (const auto &face_handle : mesh.faces()) {
        const int face_idx = face_handle.idx();
        getFaceNeighbor(mesh, face_handle, all_face_neighbor[face_idx]);
        if (include_central_face) {
            all_face_neighbor[face_idx].push_back(face_handle);
        }
    }
}

// 根据平滑后的法向量场，更新顶点位置
void updateVertexPosition(
    MyMesh &mesh,
    std::vector<MyMesh::Normal> &filtered_normals,
    int iteration_number,
    bool fixed_boundary)
{
    if (iteration_number <= 0)
        return;

    std::vector<MyMesh::Point> face_centroids;
    getFaceCentroid(mesh, face_centroids);

    std::vector<MyMesh::Point> next_positions(mesh.n_vertices());

    for (int iter = 0; iter < iteration_number; ++iter) {
        for (const auto &vh : mesh.vertices()) {
            next_positions[vh.idx()] = mesh.point(vh);
        }

        for (const auto &vh : mesh.vertices()) {
            if (fixed_boundary && mesh.is_boundary(vh)) {
                continue;
            }

            const MyMesh::Point &current_pos = mesh.point(vh);
            MyMesh::Point displacement_accumulator(0.0f, 0.0f, 0.0f);
            int adjacent_face_count = 0;

            for (auto vf_it = mesh.cvf_iter(vh); vf_it.is_valid(); ++vf_it) {
                const auto &adjacent_face_h = *vf_it;
                const int face_idx = adjacent_face_h.idx();

                const MyMesh::Normal &target_normal =
                    filtered_normals[face_idx];
                const MyMesh::Point &face_center = face_centroids[face_idx];

                MyMesh::Point vec_to_centroid = face_center - current_pos;
                MyMesh::Point displacement =
                    target_normal *
                    OpenMesh::dot(vec_to_centroid, target_normal);

                displacement_accumulator += displacement;
                adjacent_face_count++;
            }

            if (adjacent_face_count > 0) {
                next_positions[vh.idx()] +=
                    displacement_accumulator /
                    static_cast<float>(adjacent_face_count);
            }
        }

        for (const auto &vh : mesh.vertices()) {
            mesh.set_point(vh, next_positions[vh.idx()]);
        }
    }
}

float getSigmaC(
    MyMesh &mesh,
    std::vector<MyMesh::Point> &face_centroid,
    float multiple_sigma_c)
{
    float sigma_c = 0.0;
    float num = 0.0;
    for (MyMesh::FaceIter f_it = mesh.faces_begin(); f_it != mesh.faces_end();
         f_it++) {
        MyMesh::Point ci = face_centroid[f_it->idx()];
        for (MyMesh::FaceFaceIter ff_it = mesh.ff_iter(*f_it); ff_it.is_valid();
             ff_it++) {
            MyMesh::Point cj = face_centroid[ff_it->idx()];
            sigma_c += (ci - cj).length();
            num++;
        }
    }
    if (num < 1.0f)
        return 1.0f;
    sigma_c *= multiple_sigma_c / num;
    return sigma_c;
}

void update_filtered_normals_local_scheme(
    MyMesh &mesh,
    std::vector<MyMesh::Normal> &filtered_normals,
    float multiple_sigma_c,
    int normal_iteration_number,
    float sigma_s)
{
    std::vector<std::vector<FaceHandle>> all_face_neighbor;
    getAllFaceNeighbor(mesh, all_face_neighbor, false);

    std::vector<MyMesh::Normal> previous_normals;
    getFaceNormal(mesh, previous_normals);

    std::vector<float> face_areas;
    getFaceArea(mesh, face_areas);

    std::vector<MyMesh::Point> face_centroids;
    getFaceCentroid(mesh, face_centroids);

    float sigma_c = getSigmaC(mesh, face_centroids, multiple_sigma_c);
    const float sigma_c_sq = sigma_c * sigma_c;
    const float sigma_s_sq = sigma_s * sigma_s;
    const float epsilon = 1e-9f;

    for (int iter = 0; iter < normal_iteration_number; ++iter) {
        std::vector<MyMesh::Normal> next_normals(mesh.n_faces());

        for (const auto &center_face_handle : mesh.faces()) {
            const int center_face_idx = center_face_handle.idx();

            const MyMesh::Point &center_centroid =
                face_centroids[center_face_idx];
            const MyMesh::Normal &center_normal =
                previous_normals[center_face_idx];

            MyMesh::Normal normal_accumulator(0.0f, 0.0f, 0.0f);
            float total_weight = 0.0f;

            const auto &neighbor_handles = all_face_neighbor[center_face_idx];
            for (const auto &neighbor_handle : neighbor_handles) {
                const int neighbor_idx = neighbor_handle.idx();

                const MyMesh::Point &neighbor_centroid =
                    face_centroids[neighbor_idx];
                const MyMesh::Normal &neighbor_normal =
                    previous_normals[neighbor_idx];

                const float spatial_dist_sq =
                    (center_centroid - neighbor_centroid).sqrnorm();
                const float spatial_weight =
                    std::exp(-0.5f * spatial_dist_sq / sigma_c_sq);

                const float range_dist_sq =
                    (center_normal - neighbor_normal).sqrnorm();
                const float range_weight =
                    std::exp(-0.5f * range_dist_sq / sigma_s_sq);

                const float final_weight =
                    face_areas[neighbor_idx] * spatial_weight * range_weight;

                normal_accumulator += neighbor_normal * final_weight;
                total_weight += final_weight;
            }

            if (total_weight > epsilon) {
                next_normals[center_face_idx] =
                    (normal_accumulator / total_weight).normalize();
            }
            else {
                next_normals[center_face_idx] = center_normal;
            }
        }
        previous_normals = next_normals;
    }
    filtered_normals = previous_normals;
}

void bilateral_normal_filtering(
    MyMesh &mesh,
    float sigma_s,
    int normal_iteration_number,
    float multiple_sigma_c)
{
    std::vector<MyMesh::Normal> filtered_normals;
    update_filtered_normals_local_scheme(
        mesh,
        filtered_normals,
        multiple_sigma_c,
        normal_iteration_number,
        sigma_s);

    updateVertexPosition(mesh, filtered_normals, normal_iteration_number, true);
}

NODE_DEF_OPEN_SCOPE

NODE_DECLARATION_FUNCTION(mesh_smoothing)
{
    b.add_input<Geometry>("Mesh");
    b.add_input<float>("Sigma_s").default_val(0.1).min(0).max(1);
    b.add_input<int>("Iterations").default_val(1).min(0).max(30);
    b.add_input<float>("Multiple Sigma C").default_val(1.0).min(0).max(10);
    b.add_output<Geometry>("Smoothed Mesh");
}

NODE_EXECUTION_FUNCTION(mesh_smoothing)
{
    auto geometry = params.get_input<Geometry>("Mesh");
    auto mesh = geometry.get_component<MeshComponent>();
    auto vertices = mesh->get_vertices();
    auto face_vertex_indices = mesh->get_face_vertex_indices();
    auto face_vertex_counts = mesh->get_face_vertex_counts();

    MyMesh omesh;
    for (size_t i = 0; i < vertices.size(); i++) {
        omesh.add_vertex(
            OpenMesh::Vec3f(vertices[i][0], vertices[i][1], vertices[i][2]));
    }
    size_t start = 0;
    for (int face_vertex_count : face_vertex_counts) {
        std::vector<MyMesh::VertexHandle> face;
        face.reserve(face_vertex_count);
        for (int j = 0; j < face_vertex_count; j++) {
            face.push_back(
                MyMesh::VertexHandle(face_vertex_indices[start + j]));
        }
        omesh.add_face(face);
        start += face_vertex_count;
    }

    float sigma_s = params.get_input<float>("Sigma_s");
    int iterations = params.get_input<int>("Iterations");
    float multiple_sigma_c = params.get_input<float>("Multiple Sigma C");
    bilateral_normal_filtering(omesh, sigma_s, iterations, multiple_sigma_c);

    pxr::VtArray<pxr::GfVec3f> smoothed_vertices;
    for (const auto &v : omesh.vertices()) {
        const auto &p = omesh.point(v);
        smoothed_vertices.push_back(pxr::GfVec3f(p[0], p[1], p[2]));
    }

    Geometry smoothed_geometry;
    auto smoothed_mesh = std::make_shared<MeshComponent>(&smoothed_geometry);
    smoothed_mesh->set_vertices(smoothed_vertices);
    smoothed_mesh->set_face_vertex_indices(face_vertex_indices);  // 拓扑不变
    smoothed_mesh->set_face_vertex_counts(face_vertex_counts);
    smoothed_geometry.attach_component(smoothed_mesh);
    params.set_output("Smoothed Mesh", smoothed_geometry);

    return true;
}

NODE_DECLARATION_UI(mesh_smoothing);

NODE_DEF_CLOSE_SCOPE
