#include <Eigen/Sparse>
#include <cmath>
#include <limits>

#include "GCore/Components/MeshOperand.h"
#include "OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh"
#include "geom_node_base.h"

typedef OpenMesh::TriMesh_ArrayKernelT<> MyMesh;

void tutte_embedding(MyMesh& omesh)
{
    int n_vertices = omesh.n_vertices();
    std::vector<MyMesh::VertexHandle> interior_vertices;
    std::vector<MyMesh::VertexHandle> boundary_vertices;

    std::map<int, int> handle_to_interior_index;
    int interior_count = 0;

    for (auto vh : omesh.vertices()) {
        if (omesh.is_boundary(vh)) {
            boundary_vertices.push_back(vh);
        }
        else {
            interior_vertices.push_back(vh);
            handle_to_interior_index[vh.idx()] = interior_count++;
        }
    }

    int n_interior = interior_vertices.size();
    if (n_interior == 0) {
        return;
    }
    Eigen::SparseMatrix<double> L_II(n_interior, n_interior);
    std::vector<Eigen::Triplet<double>> L_triplets;

    Eigen::VectorXd RHS_x = Eigen::VectorXd::Zero(n_interior);
    Eigen::VectorXd RHS_y = Eigen::VectorXd::Zero(n_interior);
    Eigen::VectorXd RHS_z = Eigen::VectorXd::Zero(n_interior);

    for (int i = 0; i < n_interior; ++i) {
        auto vh_i = interior_vertices[i];
        int degree = 0;

        for (auto vj_it = omesh.vv_iter(vh_i); vj_it.is_valid(); ++vj_it) {
            auto vh_j = *vj_it;
            degree++;

            if (omesh.is_boundary(vh_j)) {
                const auto& pos = omesh.point(vh_j);
                RHS_x(i) += pos[0];
                RHS_y(i) += pos[1];
                RHS_z(i) += pos[2];
            }
            else {
                int j_interior_idx = handle_to_interior_index[vh_j.idx()];
                L_triplets.push_back(
                    Eigen::Triplet<double>(i, j_interior_idx, -1.0));
            }
        }
        L_triplets.push_back(
            Eigen::Triplet<double>(i, i, static_cast<double>(degree)));
    }

    L_II.setFromTriplets(L_triplets.begin(), L_triplets.end());
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    solver.compute(L_II);

    if (solver.info() != Eigen::Success) {
        std::cerr << "Tutte Embedding: Matrix decomposition failed."
                  << std::endl;
        return;
    }

    Eigen::VectorXd P_I_x = solver.solve(RHS_x);
    Eigen::VectorXd P_I_y = solver.solve(RHS_y);
    Eigen::VectorXd P_I_z = solver.solve(RHS_z);

    if (solver.info() != Eigen::Success) {
        std::cerr << "Tutte Embedding: Linear solve failed." << std::endl;
        return;
    }

    for (int i = 0; i < n_interior; ++i) {
        auto vh_i = interior_vertices[i];
        MyMesh::Point new_pos(P_I_x(i), P_I_y(i), P_I_z(i));
        omesh.set_point(vh_i, new_pos);
    }
}

void normalize_uvs(MyMesh& uv_mesh)
{
    if (uv_mesh.n_vertices() == 0) {
        return;
    }

    float u_min = std::numeric_limits<float>::max();
    float u_max = std::numeric_limits<float>::lowest();
    float v_min = std::numeric_limits<float>::max();
    float v_max = std::numeric_limits<float>::lowest();

    for (const auto& vh : uv_mesh.vertices()) {
        const auto& p = uv_mesh.point(vh);
        u_min = std::min(u_min, p[0]);
        u_max = std::max(u_max, p[0]);
        v_min = std::min(v_min, p[1]);
        v_max = std::max(v_max, p[1]);
    }
    float u_range = u_max - u_min;
    float v_range = v_max - v_min;
    float max_range = std::max(u_range, v_range);

    if (max_range < 1e-6) {
        return;
    }

    for (auto vh : uv_mesh.vertices()) {
        auto p = uv_mesh.point(vh);
        float normalized_u = (p[0] - u_min) / max_range;
        float normalized_v = (p[1] - v_min) / max_range;
        uv_mesh.set_point(vh, MyMesh::Point(normalized_u, normalized_v, 0.0f));
    }
}
NODE_DEF_OPEN_SCOPE
NODE_DECLARATION_FUNCTION(tutte)
{
    // Function content omitted
    b.add_input<Geometry>("Input");

    b.add_output<Geometry>("Output");
}

NODE_EXECUTION_FUNCTION(tutte)
{
    auto input = params.get_input<Geometry>("Input");

    // Avoid processing the node when there is no input
    if (!input.get_component<MeshComponent>()) {
        std::cerr << "Tutte Parameterization: Need Geometry Input."
                  << std::endl;
        return false;
    }

    auto mesh = input.get_component<MeshComponent>();
    auto vertices = mesh->get_vertices();
    auto face_vertex_indices = mesh->get_face_vertex_indices();
    auto face_vertex_counts = mesh->get_face_vertex_counts();

    // Convert the mesh to OpenMesh
    MyMesh omesh;
    for (size_t i = 0; i < vertices.size(); i++) {
        omesh.add_vertex(
            OpenMesh::Vec3f(vertices[i][0], vertices[i][1], vertices[i][2]));
    }

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

    tutte_embedding(omesh);
    //    normalize_uvs(omesh);
    pxr::VtArray<pxr::GfVec3f> final_vertices;
    final_vertices.reserve(omesh.n_vertices());
    for (const auto& v : omesh.vertices()) {
        const auto& p = omesh.point(v);
        final_vertices.push_back(pxr::GfVec3f(p[0], p[1], p[2]));
    }

    pxr::VtArray<int> final_faceVertexIndices = face_vertex_indices;
    pxr::VtArray<int> final_faceVertexCounts = face_vertex_counts;

    Geometry final_geometry;
    auto final_mesh = std::make_shared<MeshComponent>(&final_geometry);

    final_mesh->set_vertices(final_vertices);
    final_mesh->set_face_vertex_indices(final_faceVertexIndices);
    final_mesh->set_face_vertex_counts(final_faceVertexCounts);

    final_geometry.attach_component(final_mesh);

    params.set_output("Output", final_geometry);

    return true;
}

NODE_DECLARATION_UI(tutte);
NODE_DEF_CLOSE_SCOPE
