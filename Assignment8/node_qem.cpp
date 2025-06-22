#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <cmath>
#include <iostream>
#include <memory>
#include <queue>   
#include "GCore/Components/MeshOperand.h"
#include "GCore/util_openmesh_bind.h"
#include "geom_node_base.h"
#include "nodes/core/def/node_def.hpp"

typedef OpenMesh::PolyMesh_ArrayKernelT<> MyMesh;


struct ContractionCandidate {
    float cost;
    MyMesh::VertexHandle vh1;
    MyMesh::VertexHandle vh2;
    Eigen::Vector3f optimal_pos;
    int v1_version;
    int v2_version;

    bool operator>(const ContractionCandidate& other) const
    {
        return cost > other.cost;
    }
};
Eigen::Matrix4f compute_face_quadric(
    const std::shared_ptr<MyMesh>& mesh,
    MyMesh::FaceHandle fh)
{
    auto fv_it = mesh->cfv_iter(fh);
    const auto& p0 = mesh->point(*fv_it++);
    const auto& p1 = mesh->point(*fv_it++);
    const auto& p2 = mesh->point(*fv_it);

    auto normal = (p1 - p0).cross(p2 - p0).normalized();
    float d = -normal.dot(p0);

    Eigen::Vector4f plane_eq(normal[0], normal[1], normal[2], d);

    return plane_eq * plane_eq.transpose();
}

float calculate_quadric_error(
    const Eigen::Matrix4f& Q,
    const Eigen::Vector3f& v)
{
    Eigen::Vector4f v_h(v.x(), v.y(), v.z(), 1.0f);
    return v_h.transpose() * Q * v_h;
}

ContractionCandidate create_contraction_candidate(
    const std::shared_ptr<MyMesh>& mesh,
    MyMesh::VertexHandle vh1,
    MyMesh::VertexHandle vh2,
    OpenMesh::VPropHandleT<Eigen::Matrix4f>& vprop_quadric,
    OpenMesh::VPropHandleT<int>& vprop_version)
{
    if (vh1.idx() > vh2.idx())
        std::swap(vh1, vh2);

    ContractionCandidate candidate;
    candidate.vh1 = vh1;
    candidate.vh2 = vh2;
    candidate.v1_version = mesh->property(vprop_version, vh1);
    candidate.v2_version = mesh->property(vprop_version, vh2);

    const Eigen::Matrix4f& Q1 = mesh->property(vprop_quadric, vh1);
    const Eigen::Matrix4f& Q2 = mesh->property(vprop_quadric, vh2);
    Eigen::Matrix4f Q_sum = Q1 + Q2;

    Eigen::Matrix3f A = Q_sum.block<3, 3>(0, 0);
    Eigen::Vector3f b = Q_sum.block<3, 1>(0, 3);

    bool solved = false;
    // Try to find the optimal point by solving the linear system.
    if (std::abs(A.determinant()) > 1e-9) {
        candidate.optimal_pos = A.ldlt().solve(-b);
        solved = true;
    }

    if (!solved) {
        const auto& p1 = mesh->point(vh1);
        const auto& p2 = mesh->point(vh2);
        Eigen::Vector3f v1_pos(p1[0], p1[1], p1[2]);
        Eigen::Vector3f v2_pos(p2[0], p2[1], p2[2]);
        Eigen::Vector3f mid_pos = (v1_pos + v2_pos) * 0.5f;

        float cost1 = calculate_quadric_error(Q_sum, v1_pos);
        float cost2 = calculate_quadric_error(Q_sum, v2_pos);
        float cost_mid = calculate_quadric_error(Q_sum, mid_pos);

        if (cost1 < cost2 && cost1 < cost_mid) {
            candidate.optimal_pos = v1_pos;
        }
        else if (cost2 < cost_mid) {
            candidate.optimal_pos = v2_pos;
        }
        else {
            candidate.optimal_pos = mid_pos;
        }
    }

    candidate.cost = calculate_quadric_error(Q_sum, candidate.optimal_pos);
    return candidate;
}

void qem(
    std::shared_ptr<MyMesh> halfedge_mesh,
    float simplification_ratio,
    float distance_threshold)
{
    if (!halfedge_mesh || simplification_ratio <= 0.0f ||
        simplification_ratio >= 1.0f) {
        return;
    }

    // Compute the Q matrices for all initial vertices.
    OpenMesh::VPropHandleT<Eigen::Matrix4f> vprop_quadric;
    halfedge_mesh->add_property(vprop_quadric);

    for (auto vh : halfedge_mesh->vertices()) {
        Eigen::Matrix4f Q = Eigen::Matrix4f::Zero();
        for (auto vf_it = halfedge_mesh->cvf_iter(vh); vf_it.is_valid();
             ++vf_it) {
            Q += compute_face_quadric(halfedge_mesh, *vf_it);
        }
        halfedge_mesh->property(vprop_quadric, vh) = Q;
    }

    OpenMesh::VPropHandleT<int> vprop_version;
    halfedge_mesh->add_property(vprop_version);
    for (auto vh : halfedge_mesh->vertices()) {
        halfedge_mesh->property(vprop_version, vh) = 0;
    }

    // Select pairs, compute costs, and place them in a priority queue.
    std::priority_queue<
        ContractionCandidate,
        std::vector<ContractionCandidate>,
        std::greater<ContractionCandidate>>
        pq;
    for (auto eh : halfedge_mesh->edges()) {
        auto vh1 = halfedge_mesh->to_vertex_handle(
            halfedge_mesh->halfedge_handle(eh, 0));
        auto vh2 = halfedge_mesh->to_vertex_handle(
            halfedge_mesh->halfedge_handle(eh, 1));
        pq.push(create_contraction_candidate(
            halfedge_mesh, vh1, vh2, vprop_quadric, vprop_version));
    }

    //Iteratively collapse pairs until the target face count is reached.
    const size_t initial_face_count = halfedge_mesh->n_faces();
    const size_t target_face_count =
        static_cast<size_t>(initial_face_count * simplification_ratio);
    size_t current_face_count = initial_face_count;

    while (current_face_count > target_face_count && !pq.empty()) {
        ContractionCandidate best_pair = pq.top();
        pq.pop();

        auto vh1 = best_pair.vh1;
        auto vh2 = best_pair.vh2;

        if (halfedge_mesh->status(vh1).deleted() ||
            halfedge_mesh->status(vh2).deleted() ||
            best_pair.v1_version !=
                halfedge_mesh->property(vprop_version, vh1) ||
            best_pair.v2_version !=
                halfedge_mesh->property(vprop_version, vh2)) {
            continue;
        }

        MyMesh::HalfedgeHandle heh_to_collapse;
        bool found_heh = false;
        for (auto heh : halfedge_mesh->voh_range(vh1)) {
            if (halfedge_mesh->to_vertex_handle(heh) == vh2) {
                heh_to_collapse = heh;
                found_heh = true;
                break;
            }
        }

        if (!found_heh || !halfedge_mesh->is_collapse_ok(heh_to_collapse)) {
            continue;
        }

        halfedge_mesh->set_point(
            vh2,
            MyMesh::Point(
                best_pair.optimal_pos.x(),
                best_pair.optimal_pos.y(),
                best_pair.optimal_pos.z()));
        halfedge_mesh->collapse(heh_to_collapse);

        halfedge_mesh->property(vprop_quadric, vh2) +=
            halfedge_mesh->property(vprop_quadric, vh1);
        halfedge_mesh->property(vprop_version, vh2)++;

        for (auto voh_it = halfedge_mesh->cvoh_iter(vh2); voh_it.is_valid();
             ++voh_it) {
            auto neighbor_vh = halfedge_mesh->to_vertex_handle(*voh_it);
            pq.push(create_contraction_candidate(
                halfedge_mesh, vh2, neighbor_vh, vprop_quadric, vprop_version));
        }

        current_face_count = 0;
        for (auto f_it = halfedge_mesh->faces_sbegin();
             f_it != halfedge_mesh->faces_end();
             ++f_it) {
            current_face_count++;
        }
    }

    // Clean up deleted elements from the mesh.
    halfedge_mesh->remove_property(vprop_quadric);
    halfedge_mesh->remove_property(vprop_version);
    halfedge_mesh->garbage_collection();
}
NODE_DEF_OPEN_SCOPE

NODE_DECLARATION_FUNCTION(qem)
{
    // Input-1: Original 3D mesh
    b.add_input<Geometry>("Input");
    // Input-2: Mesh simplification ratio, AKA the ratio of the number of
    // vertices in the simplified mesh to the number of vertices in the original
    // mesh
    b.add_input<float>("Simplification Ratio")
        .default_val(0.5f)
        .min(0.0f)
        .max(1.0f);
    // Input-3: Distance threshold for non-edge vertex pairs
    b.add_input<float>("Non-edge Distance Threshold")
        .default_val(0.01f)
        .min(0.0f)
        .max(1.0f);
    // Output-1: Simplified mesh
    b.add_output<Geometry>("Output");
}

NODE_EXECUTION_FUNCTION(qem)
{
    // Get the input mesh
    auto input_mesh = params.get_input<Geometry>("Input");
    auto simplification_ratio = params.get_input<float>("Simplification Ratio");
    auto distance_threshold =
        params.get_input<float>("Non-edge Distance Threshold");

    // Avoid processing the node when there is no input
    if (!input_mesh.get_component<MeshComponent>()) {
        std::cerr << "QEM: No input mesh provided." << std::endl;
        return false;
    }

    /* ----------------------------- Preprocess
     *-------------------------------
     ** Create a halfedge structure (using OpenMesh) for the input mesh. The
     ** half-edge data structure is a widely used data structure in
     *geometric
     ** processing, offering convenient operations for traversing and
     *modifying
     ** mesh elements.
     */

    // Initialization
    auto halfedge_mesh = operand_to_openmesh(&input_mesh);

    halfedge_mesh->request_vertex_status();
    halfedge_mesh->request_edge_status();
    halfedge_mesh->request_face_status();
    halfedge_mesh->request_halfedge_status();

    // QEM simplification
    qem(halfedge_mesh, simplification_ratio, distance_threshold);

    // Convert the simplified mesh back to the operand
    auto geometry = openmesh_to_operand(halfedge_mesh.get());

    auto mesh = geometry->get_component<MeshComponent>();

    // Set the output of the nodes
    params.set_output("Output", std::move(*geometry));

    return true;
}

NODE_DECLARATION_UI(qem);

NODE_DEF_CLOSE_SCOPE
