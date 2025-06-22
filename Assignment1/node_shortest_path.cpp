#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>
#include <cstddef>
#include <functional>  // std::greater
#include <queue>       // std::priority_queue
#include <string>

#include "GCore/Components/MeshOperand.h"
#include "GCore/GOP.h"
#include "nodes/core/def/node_def.hpp"

typedef OpenMesh::TriMesh_ArrayKernelT<> MyMesh;

// Finds the shortest path on the mesh.
// Returns true on success, filling out path indices and distance.
bool find_shortest_path(
    const MyMesh::VertexHandle& start_vertex_handle,
    const MyMesh::VertexHandle& end_vertex_handle,
    const MyMesh& omesh,
    std::list<size_t>& shortest_path_vertex_indices,
    float& distance)
{
    // Validate input handles
    if (!start_vertex_handle.is_valid() || !end_vertex_handle.is_valid() ||
        start_vertex_handle.idx() >= static_cast<int>(omesh.n_vertices()) ||
        end_vertex_handle.idx() >= static_cast<int>(omesh.n_vertices())) {
        return false;
    }

    // Handle trivial case where start and end are the same
    if (start_vertex_handle == end_vertex_handle) {
        shortest_path_vertex_indices = { static_cast<size_t>(
            start_vertex_handle.idx()) };
        distance = 0.0f;
        return true;
    }

    // Dijkstra algorithm setup
    const size_t n_vertices = omesh.n_vertices();
    const float INF = std::numeric_limits<float>::infinity();

    std::vector<float> dist(n_vertices, INF);
    std::vector<int> prev(n_vertices, -1);

    using QueueItem = std::pair<float, int>;
    std::priority_queue<
        QueueItem,
        std::vector<QueueItem>,
        std::greater<QueueItem>>
        pq;

    dist[start_vertex_handle.idx()] = 0.0f;
    pq.emplace(0.0f, start_vertex_handle.idx());

    // Dijkstra main loop
    while (!pq.empty()) {
        const auto [du, ui] = pq.top();
        pq.pop();

        // Skip outdated entry in the queue
        if (du > dist[ui])
            continue;

        // Early exit if destination is reached
        if (ui == end_vertex_handle.idx())
            break;

        MyMesh::VertexHandle u(ui);

        // For each neighbor v of u
        for (auto voh_it = omesh.cvoh_iter(u); voh_it.is_valid(); ++voh_it) {
            OpenMesh::VertexHandle v = omesh.to_vertex_handle(*voh_it);
            float w = (omesh.point(u) - omesh.point(v)).norm();

            // Relaxation step
            float alt = du + w;
            if (alt < dist[v.idx()]) {
                dist[v.idx()] = alt;
                prev[v.idx()] = ui;
                pq.emplace(alt, v.idx());
            }
        }
    }

    // Check if a path was found
    if (dist[end_vertex_handle.idx()] == INF) {
        return false;  // No path found
    }

    // Reconstruct path by backtracking
    std::vector<size_t> rev_path;
    for (int v = end_vertex_handle.idx(); v != -1; v = prev[v]) {
        rev_path.push_back(static_cast<size_t>(v));
        if (v == start_vertex_handle.idx())
            break;
    }

    // Sanity check for backtracking
    if (rev_path.back() != static_cast<size_t>(start_vertex_handle.idx())) {
        return false;
    }

    // Reverse path to start->end order
    shortest_path_vertex_indices.clear();
    for (auto it = rev_path.rbegin(); it != rev_path.rend(); ++it)
        shortest_path_vertex_indices.push_back(*it);

    distance = dist[end_vertex_handle.idx()];
    return true;
}

NODE_DEF_OPEN_SCOPE

NODE_DECLARATION_FUNCTION(shortest_path)
{
    b.add_input<std::string>("Picked Mesh [0] Name");
    b.add_input<std::string>("Picked Mesh [1] Name");
    b.add_input<Geometry>("Picked Mesh");
    b.add_input<size_t>("Picked Vertex [0] Index");
    b.add_input<size_t>("Picked Vertex [1] Index");

    b.add_output<std::list<size_t>>("Shortest Path Vertex Indices");
    b.add_output<float>("Shortest Path Distance");
}

NODE_EXECUTION_FUNCTION(shortest_path)
{
    auto picked_mesh_0_name =
        params.get_input<std::string>("Picked Mesh [0] Name");
    auto picked_mesh_1_name =
        params.get_input<std::string>("Picked Mesh [1] Name");

    // Ensure that the two picked vertices are on the same mesh
    if (picked_mesh_0_name != picked_mesh_1_name) {
        std::cerr << "Ensure that the two picked meshes are the same"
                  << std::endl;
        return false;
    }

    auto mesh = params.get_input<Geometry>("Picked Mesh")
                    .get_component<MeshComponent>();
    auto vertices = mesh->get_vertices();
    auto face_vertex_counts = mesh->get_face_vertex_counts();
    auto face_vertex_indices = mesh->get_face_vertex_indices();

    // Convert the mesh to OpenMesh format
    MyMesh omesh;

    // Add vertices
    std::vector<OpenMesh::VertexHandle> vhandles;
    vhandles.reserve(vertices.size());
    for (auto vertex : vertices) {
        omesh.add_vertex(OpenMesh::Vec3f(vertex[0], vertex[1], vertex[2]));
    }

    // Add faces
    size_t start = 0;
    for (int face_vertex_count : face_vertex_counts) {
        std::vector<OpenMesh::VertexHandle> face;
        face.reserve(face_vertex_count);
        for (int j = 0; j < face_vertex_count; j++) {
            face.push_back(
                OpenMesh::VertexHandle(face_vertex_indices[start + j]));
        }
        omesh.add_face(face);
        start += face_vertex_count;
    }

    auto start_vertex_index =
        params.get_input<size_t>("Picked Vertex [0] Index");
    auto end_vertex_index = params.get_input<size_t>("Picked Vertex [1] Index");

    // Get OpenMesh vertex handles from indices
    OpenMesh::VertexHandle start_vertex_handle(start_vertex_index);
    OpenMesh::VertexHandle end_vertex_handle(end_vertex_index);

    // Stores the result: vertex indices on the shortest path
    std::list<size_t> shortest_path_vertex_indices;

    // Stores the result: total distance
    float distance = 0.0f;

    if (find_shortest_path(
            start_vertex_handle,
            end_vertex_handle,
            omesh,
            shortest_path_vertex_indices,
            distance)) {
        params.set_output(
            "Shortest Path Vertex Indices", shortest_path_vertex_indices);
        params.set_output("Shortest Path Distance", distance);
        return true;
    }
    else {
        params.set_output("Shortest Path Vertex Indices", std::list<size_t>());
        params.set_output("Shortest Path Distance", 0.0f);
        return false;
    }

    return true;
}

NODE_DECLARATION_UI(shortest_path);
NODE_DECLARATION_REQUIRED(shortest_path);

NODE_DEF_CLOSE_SCOPE
