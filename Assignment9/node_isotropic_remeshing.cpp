#include <cmath>  // For std::abs
#include <iostream>
#include <unordered_set>
#include <utility>
#include <vector>

#include "GCore/Components/MeshOperand.h"
#include "GCore/util_openmesh_bind.h"
#include "OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh"
#include "geom_node_base.h"
#include "nodes/core/def/node_def.hpp"

typedef OpenMesh::TriMesh_ArrayKernelT<> MyMesh;

// Forward declaration of helper function
MyMesh::Normal calculate_vertex_normal(
    const std::shared_ptr<MyMesh>& halfedge_mesh,
    MyMesh::VertexHandle vertex);

void split_edges(std::shared_ptr<MyMesh> halfedge_mesh, float upper_bound)
{
    std::unordered_set<MyMesh::EdgeHandle> edges_to_split;
    for (const auto& edge : halfedge_mesh->edges()) {
        if (halfedge_mesh->calc_edge_length(edge) > upper_bound) {
            edges_to_split.insert(edge);
        }
    }

    while (!edges_to_split.empty()) {
        MyMesh::EdgeHandle edge = *edges_to_split.begin();
        edges_to_split.erase(edges_to_split.begin());

        const MyMesh::HalfedgeHandle heh =
            halfedge_mesh->halfedge_handle(edge, 0);
        if (!halfedge_mesh->is_valid_handle(heh)) {
            continue;
        }

        const MyMesh::VertexHandle v1 = halfedge_mesh->from_vertex_handle(heh);
        const MyMesh::VertexHandle v2 = halfedge_mesh->to_vertex_handle(heh);
        const MyMesh::Point midpoint =
            (halfedge_mesh->point(v1) + halfedge_mesh->point(v2)) * 0.5f;

        MyMesh::VertexHandle new_vh = halfedge_mesh->add_vertex(midpoint);
        halfedge_mesh->split(edge, new_vh);

        for (const auto& new_edge : halfedge_mesh->ve_range(new_vh)) {
            if (halfedge_mesh->calc_edge_length(new_edge) > upper_bound) {
                edges_to_split.insert(new_edge);
            }
        }
    }
    halfedge_mesh->garbage_collection();
}

void collapse_edges(std::shared_ptr<MyMesh> halfedge_mesh, float lower_bound)
{
    std::vector<MyMesh::EdgeHandle> edges_to_collapse;
    edges_to_collapse.reserve(halfedge_mesh->n_edges());
    for (const auto& edge : halfedge_mesh->edges()) {
        if (halfedge_mesh->calc_edge_length(edge) < lower_bound) {
            edges_to_collapse.push_back(edge);
        }
    }

    bool was_collapsed_in_pass = false;
    for (const auto& edge : edges_to_collapse) {
        if (halfedge_mesh->status(edge).deleted()) {
            continue;
        }

        if (halfedge_mesh->calc_edge_length(edge) < lower_bound) {
            const MyMesh::HalfedgeHandle heh =
                halfedge_mesh->halfedge_handle(edge, 0);

            if (!heh.is_valid()) {
                continue;
            }

            const MyMesh::VertexHandle v_to =
                halfedge_mesh->to_vertex_handle(heh);
            const MyMesh::VertexHandle v_from =
                halfedge_mesh->from_vertex_handle(heh);

            if (halfedge_mesh->is_boundary(v_to) ||
                halfedge_mesh->is_boundary(v_from)) {
                continue;
            }

            if (halfedge_mesh->is_collapse_ok(heh)) {
                const MyMesh::Point& p1 = halfedge_mesh->point(v_to);
                const MyMesh::Point& p2 = halfedge_mesh->point(v_from);
                halfedge_mesh->set_point(v_to, (p1 + p2) * 0.5f);
                halfedge_mesh->collapse(heh);
                was_collapsed_in_pass = true;
            }
        }
    }

    if (was_collapsed_in_pass) {
        halfedge_mesh->garbage_collection();
    }
}

void flip_edges(std::shared_ptr<MyMesh> halfedge_mesh)
{
    std::vector<MyMesh::EdgeHandle> edges_to_flip;
    for (const auto& edge : halfedge_mesh->edges()) {
        if (halfedge_mesh->is_boundary(edge) ||
            !halfedge_mesh->is_flip_ok(edge)) {
            continue;
        }

        const MyMesh::HalfedgeHandle heh =
            halfedge_mesh->halfedge_handle(edge, 0);
        const MyMesh::HalfedgeHandle opp_heh =
            halfedge_mesh->opposite_halfedge_handle(heh);

        const MyMesh::VertexHandle v1 = halfedge_mesh->to_vertex_handle(heh);
        const MyMesh::VertexHandle v2 = halfedge_mesh->from_vertex_handle(heh);
        const MyMesh::VertexHandle v3 = halfedge_mesh->to_vertex_handle(
            halfedge_mesh->next_halfedge_handle(heh));
        const MyMesh::VertexHandle v4 = halfedge_mesh->to_vertex_handle(
            halfedge_mesh->next_halfedge_handle(opp_heh));

        const int t1 = halfedge_mesh->is_boundary(v1) ? 4 : 6;
        const int t2 = halfedge_mesh->is_boundary(v2) ? 4 : 6;
        const int t3 = halfedge_mesh->is_boundary(v3) ? 4 : 6;
        const int t4 = halfedge_mesh->is_boundary(v4) ? 4 : 6;

        int deviation_before = std::abs((int)halfedge_mesh->valence(v1) - t1) +
                               std::abs((int)halfedge_mesh->valence(v2) - t2) +
                               std::abs((int)halfedge_mesh->valence(v3) - t3) +
                               std::abs((int)halfedge_mesh->valence(v4) - t4);

        int deviation_after =
            std::abs((int)halfedge_mesh->valence(v1) - 1 - t1) +
            std::abs((int)halfedge_mesh->valence(v2) - 1 - t2) +
            std::abs((int)halfedge_mesh->valence(v3) + 1 - t3) +
            std::abs((int)halfedge_mesh->valence(v4) + 1 - t4);

        if (deviation_after < deviation_before) {
            edges_to_flip.push_back(edge);
        }
    }

    for (const auto& edge : edges_to_flip) {
        if (!halfedge_mesh->status(edge).deleted() &&
            halfedge_mesh->is_flip_ok(edge)) {
            halfedge_mesh->flip(edge);
        }
    }

    if (!edges_to_flip.empty()) {
        halfedge_mesh->garbage_collection();
    }
}

void relocate_vertices(std::shared_ptr<MyMesh> halfedge_mesh, float lambda)
{
    const int n_verts = halfedge_mesh->n_vertices();
    std::vector<MyMesh::Point> centroids(n_verts);
    std::vector<float> vertex_areas(n_verts, 0.0f);

    for (const auto& face : halfedge_mesh->faces()) {
        auto fv_it = halfedge_mesh->cfv_iter(face);
        const auto& p0 = halfedge_mesh->point(*fv_it++);
        const auto& p1 = halfedge_mesh->point(*fv_it++);
        const auto& p2 = halfedge_mesh->point(*fv_it);
        const float face_area = ((p1 - p0) % (p2 - p0)).norm() * 0.5f;
        const float area_per_vertex = face_area / 3.0f;

        fv_it = halfedge_mesh->cfv_iter(face);  
        vertex_areas[(*fv_it++).idx()] += area_per_vertex;
        vertex_areas[(*fv_it++).idx()] += area_per_vertex;
        vertex_areas[(*fv_it).idx()] += area_per_vertex;
    }
    // vertex
    for (const auto& vertex : halfedge_mesh->vertices()) {
        if (halfedge_mesh->is_boundary(vertex)) {
            centroids[vertex.idx()] = halfedge_mesh->point(vertex);
            continue;
        }

        MyMesh::Point weighted_sum(0.0f, 0.0f, 0.0f);
        float total_area_sum = 0.0f;

        for (auto vv_it = halfedge_mesh->cvv_iter(vertex); vv_it.is_valid();
             ++vv_it) {
            const float neighbor_area = vertex_areas[vv_it->idx()];
            weighted_sum += halfedge_mesh->point(*vv_it) * neighbor_area;
            total_area_sum += neighbor_area;
        }

        if (total_area_sum > 0.0f) {
            centroids[vertex.idx()] = weighted_sum / total_area_sum;
        }
        else {
            centroids[vertex.idx()] = halfedge_mesh->point(vertex);
        }
    }

    for (const auto& vertex : halfedge_mesh->vertices()) {
        if (halfedge_mesh->is_boundary(vertex)) {
            continue;
        }

        const MyMesh::Point current_pos = halfedge_mesh->point(vertex);
        const MyMesh::Point& centroid = centroids[vertex.idx()];
        MyMesh::Point normal = calculate_vertex_normal(halfedge_mesh, vertex);

        if (normal.norm() <
            1e-6) 
        {
            continue;
        }
        normal.normalize();

        const MyMesh::Point direction = centroid - current_pos;
        const float dot_product = direction | normal;  // Dot product
        const MyMesh::Point tangential_move = direction - normal * dot_product;

        const MyMesh::Point new_pos = current_pos + lambda * tangential_move;
        halfedge_mesh->set_point(vertex, new_pos);
    }
}

MyMesh::Normal calculate_vertex_normal(
    const std::shared_ptr<MyMesh>& halfedge_mesh,
    MyMesh::VertexHandle vertex)
{
    MyMesh::Normal normal(0.0f, 0.0f, 0.0f);
    if (!halfedge_mesh->is_isolated(vertex)) {
        for (const auto& face : halfedge_mesh->vf_range(vertex)) {
            normal += halfedge_mesh->calc_face_normal(face);
        }
        float norm = normal.norm();
        if (norm > 0.0f) {
            normal /= norm;
        }
    }
    return normal;
}

void isotropic_remeshing(
    std::shared_ptr<MyMesh> halfedge_mesh,
    float target_edge_length,
    int num_iterations,
    float lambda)
{
    for (int i = 0; i < num_iterations; ++i) {
        //std::cout << num_iterations << "-"<<1 << std::endl;
        split_edges(halfedge_mesh, target_edge_length * 4.0f / 3.0f);
        //std::cout << num_iterations << "-" << 2 << std::endl;
        collapse_edges(halfedge_mesh, target_edge_length * 4.0f / 5.0f);

        //std::cout << num_iterations << "-" << 3 << std::endl;
        flip_edges(halfedge_mesh);
        //std::cout << num_iterations << "-" << 4 << std::endl;
        relocate_vertices(halfedge_mesh, lambda);
        //std::cout << num_iterations << "-" << 5 << std::endl;
    }
}

NODE_DEF_OPEN_SCOPE

NODE_DECLARATION_FUNCTION(isotropic_remeshing)
{
    // The input-1 is a mesh
    b.add_input<Geometry>("Mesh");

    // The input-2 is the target edge length
    b.add_input<float>("Target Edge Length")
        .default_val(0.1f)
        .min(0.01f)
        .max(10.0f);

    // The input-3 is the number of iterations
    b.add_input<int>("Iterations").default_val(10).min(0).max(20);

    // The input-4 is the lambda value for vertex relocation
    b.add_input<float>("Lambda").default_val(1.0f).min(0.0f).max(1.0f);

    // The output is a remeshed version of the input mesh
    b.add_output<Geometry>("Remeshed Mesh");
}

NODE_EXECUTION_FUNCTION(isotropic_remeshing)
{
    auto geometry = params.get_input<Geometry>("Mesh");
    auto mesh = geometry.get_component<MeshComponent>();

    //std::cout << 1 << std::endl;
    if (!mesh) {
        std::cerr << "Isotropic Remeshing Node: Failed to get MeshComponent "
                     "from input geometry."
                  << std::endl;
        return false;
    }
    auto halfedge_mesh = operand_to_openmesh_trimesh(&geometry);
    //std::cout << 2 << std::endl;

    float target_edge_length = params.get_input<float>("Target Edge Length");
    int num_iterations = params.get_input<int>("Iterations");
    float lambda = params.get_input<float>("Lambda");
    if (target_edge_length <= 0.0f) {
        std::cerr << "Isotropic Remeshing Node: Target edge length must be "
                     "greater than zero."
                  << std::endl;
        return false;
    }
    //std::cout << 3 << std::endl;

    if (num_iterations < 0) {
        std::cerr << "Isotropic Remeshing Node: Number of iterations must be "
                     "greater than zero."
                  << std::endl;
        return false;
    }

    //std::cout <<4  << std::endl;
    halfedge_mesh->request_vertex_status();
    halfedge_mesh->request_edge_status();
    halfedge_mesh->request_face_status();
    halfedge_mesh->request_halfedge_status();
    halfedge_mesh->request_face_normals();
    halfedge_mesh->request_vertex_normals();
    // implementation

    //std::cout << 5 << std::endl;
    isotropic_remeshing(
        halfedge_mesh, target_edge_length, num_iterations, lambda);

    //std::cout << 6 << std::endl;
    auto remeshed_geometry = openmesh_to_operand_trimesh(halfedge_mesh.get());
    params.set_output("Remeshed Mesh", std::move(*remeshed_geometry));

    return true;
}

NODE_DECLARATION_UI(isotropic_remeshing);
NODE_DEF_CLOSE_SCOPE
