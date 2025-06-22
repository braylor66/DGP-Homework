#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <cmath>
#include <limits>
#include <memory>

#include "GCore/Components/MeshOperand.h"
#include "GCore/util_openmesh_bind.h"
#include "geom_node_base.h"

typedef OpenMesh::PolyMesh_ArrayKernelT<> MyMesh;
void arap(
    std::shared_ptr<MyMesh> halfedge_mesh,
    std::shared_ptr<MyMesh> iter_mesh)
{
    int n_faces = halfedge_mesh->n_faces();
    int n_vertices = halfedge_mesh->n_vertices();

    // Construct a set of new triangles
    std::vector<std::vector<Eigen::Vector2d>> edges(n_faces);
    Eigen::SparseMatrix<double> cotangents(n_vertices, n_vertices);

    std::vector<Eigen::Triplet<double>> triple;
    // Use vertex 0 as the fixed vertex
    triple.push_back(Eigen::Triplet<double>(0, 0, 1));

    for (auto const& face_handle : halfedge_mesh->faces()) {
        int face_idx = face_handle.idx();
        std::vector<int> vertex_idx(3);
        std::vector<double> edge_length(3);
        int i = 0;
        for (const auto& vertex_handle : face_handle.vertices()) {
            vertex_idx[i++] = vertex_handle.idx();
        }

        for (int i = 0; i < 3; i++) {
            edge_length[i] =
                (halfedge_mesh->point(
                     halfedge_mesh->vertex_handle(vertex_idx[(i + 1) % 3])) -
                 halfedge_mesh->point(
                     halfedge_mesh->vertex_handle(vertex_idx[(i + 2) % 3])))
                    .length();
        }

        edges[face_idx].resize(3);
        double cos_angle =
            (edge_length[1] * edge_length[1] + edge_length[2] * edge_length[2] -
             edge_length[0] * edge_length[0]) /
            (2 * edge_length[1] * edge_length[2]);
        double sin_angle = sqrt(1 - cos_angle * cos_angle);
        edges[face_idx][1] << -edge_length[1] * cos_angle,
            -edge_length[1] * sin_angle;
        edges[face_idx][2] << edge_length[2], 0;
        edges[face_idx][0] = -edges[face_idx][1] - edges[face_idx][2];

        for (int i = 0; i < 3; i++) {
            double cos_value =
                edges[face_idx][i].dot(edges[face_idx][(i + 1) % 3]) /
                (edges[face_idx][i].norm() *
                 edges[face_idx][(i + 1) % 3].norm());
            double sin_value = sqrt(1 - cos_value * cos_value);
            double cot_value = cos_value / sin_value;
            cotangents.coeffRef(vertex_idx[i], vertex_idx[(i + 1) % 3]) =
                cot_value;

            // Use vertex 0 as the fixed vertex
            if (vertex_idx[i] == 0) {
                triple.push_back(Eigen::Triplet<double>(
                    vertex_idx[(i + 1) % 3],
                    vertex_idx[(i + 1) % 3],
                    cot_value));
            }
            else if (vertex_idx[(i + 1) % 3] == 0) {
                triple.push_back(Eigen::Triplet<double>(
                    vertex_idx[i], vertex_idx[i], cot_value));
            }
            else {
                triple.push_back(Eigen::Triplet<double>(
                    vertex_idx[i], vertex_idx[(i + 1) % 3], -cot_value));
                triple.push_back(Eigen::Triplet<double>(
                    vertex_idx[(i + 1) % 3], vertex_idx[i], -cot_value));
                triple.push_back(Eigen::Triplet<double>(
                    vertex_idx[i], vertex_idx[i], cot_value));
                triple.push_back(Eigen::Triplet<double>(
                    vertex_idx[(i + 1) % 3],
                    vertex_idx[(i + 1) % 3],
                    cot_value));
            }
        }
    }

    Eigen::SparseMatrix<double> A(n_vertices, n_vertices);
    A.setFromTriplets(triple.begin(), triple.end());

    // Precompute the matrix
    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    solver.compute(A);

    std::vector<Eigen::MatrixXd> b_pre(n_faces);
    std::vector<Eigen::MatrixXd> Jacobi_pre(n_faces);
    Eigen::MatrixXd Edges(3, 2);
    Eigen::MatrixXd Cotangents = Eigen::MatrixXd::Identity(3, 3);
    Eigen::MatrixXd transform_matrix(3, 3);
    transform_matrix << 1, -1, 0, 0, 1, -1, -1, 0, 1;
    for (const auto& face_handle : halfedge_mesh->faces()) {
        int face_idx = face_handle.idx();
        std::vector<int> vertex_idx(3);
        int i = 0;
        for (const auto& vertex_handle : face_handle.vertices())
            vertex_idx[i++] = vertex_handle.idx();
        for (int i = 0; i < 3; i++)
            Edges.row(i) = edges[face_idx][(i + 2) % 3];
        for (int i = 0; i < 3; i++)
            Cotangents(i, i) =
                cotangents.coeffRef(vertex_idx[i], vertex_idx[(i + 1) % 3]);
        Jacobi_pre[face_idx] = Cotangents * Edges;
        b_pre[face_idx] = -Jacobi_pre[face_idx].transpose() * transform_matrix;
    }

    int max_iter = 300;
    int now_iter = 0;
    double err_pre = -1e9;
    double err = 1e9;
    Eigen::VectorXd bx(n_vertices);
    Eigen::VectorXd by(n_vertices);
    std::vector<Eigen::Matrix2d> Jacobi(n_faces);
    // Begin iteration
    do {
        bx.setZero();
        by.setZero();
        for (const auto& face_handle : iter_mesh->faces()) {
            int face_idx = face_handle.idx();

            // Calculate the Jacobian matrix of each faces
            std::vector<int> vertex_idx(3);
            int i = 0;
            for (const auto& vertex_handle : face_handle.vertices())
                vertex_idx[i++] = vertex_handle.idx();
            Eigen::MatrixXd U(2, 3);
            for (int i = 0; i < 3; i++) {
                const auto& v0 =
                    iter_mesh->point(iter_mesh->vertex_handle(vertex_idx[i]));
                const auto& v1 = iter_mesh->point(
                    iter_mesh->vertex_handle(vertex_idx[(i + 1) % 3]));
                U(0, i) = (v1 - v0)[0];
                U(1, i) = (v1 - v0)[1];
            }
            Jacobi[face_idx] = U * Jacobi_pre[face_idx];

            // Use SVD deformation to determine whether there is a flip, and set
            // the sigular values of the Jacobian matrix 1
            Eigen::JacobiSVD<Eigen::MatrixXd> svd(
                Jacobi[face_idx],
                Eigen::DecompositionOptions::ComputeThinU |
                    Eigen::DecompositionOptions::ComputeThinV);
            Eigen::MatrixXd svd_u = svd.matrixU();
            Eigen::MatrixXd svd_v = svd.matrixV();
            Eigen::MatrixXd S = Eigen::MatrixXd::Identity(2, 2);
            if (Jacobi[face_idx].determinant() < 0) {
                // If there is a flip, set the fliped sigular values 1 instead
                // of -1
                if (svd.singularValues()[0] < svd.singularValues()[1])
                    S(0, 0) = -1;
                else
                    S(1, 1) = -1;
            }
            Jacobi[face_idx] = svd_u * S * svd_v.transpose();

            // Calculate bx and by by matrix multilplication
            Eigen::MatrixXd b = Jacobi[face_idx] * b_pre[face_idx];
            for (int i = 0; i < 3; i++) {
                if (vertex_idx[i] != 0) {
                    bx(vertex_idx[i]) += b(0, i);
                    by(vertex_idx[i]) += b(1, i);
                }
            }
        }
        bx(0) = 0;
        by(0) = 0;
        // Solve the linear equations
        Eigen::VectorXd ux = bx;
        ux = solver.solve(ux);
        Eigen::VectorXd uy = by;
        uy = solver.solve(uy);

        // Set the answers back to iter mesh
        for (const auto& vertex_handle : iter_mesh->vertices()) {
            int vertex_idx = vertex_handle.idx();
            iter_mesh->point(vertex_handle)[0] = ux(vertex_idx);
            iter_mesh->point(vertex_handle)[1] = uy(vertex_idx);
            iter_mesh->point(vertex_handle)[2] = 0;
        }

        // Calculate the error
        err_pre = err;
        err = 0;
        for (const auto& face_handle : iter_mesh->faces()) {
            int face_idx = face_handle.idx();
            std::vector<int> vertex_idx(3);
            int i = 0;
            for (const auto& vertex_handle : face_handle.vertices())
                vertex_idx[i++] = vertex_handle.idx();
            for (int i = 0; i < 3; i++) {
                int idx0 = vertex_idx[(i + 1) % 3];
                int idx1 = vertex_idx[(i + 2) % 3];
                const auto& tmp_edge =
                    iter_mesh->point(iter_mesh->vertex_handle(idx1)) -
                    iter_mesh->point(iter_mesh->vertex_handle(idx0));
                Eigen::Vector2d iter_edge;
                iter_edge << tmp_edge[0], tmp_edge[1];
                const auto& ori_edge = edges[face_idx][i];
                err += 0.5 * abs(cotangents.coeffRef(idx0, idx1)) *
                       (iter_edge - Jacobi[face_idx] * ori_edge).squaredNorm();
            }
        }
        now_iter++;
        // std::cout << now_iter << "\t" << err << std::endl;
    } while (now_iter < max_iter && abs(err - err_pre) > 1e-7);
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
NODE_DECLARATION_FUNCTION(arap_parameterization)
{
    // Input-1: Original 3D mesh with boundary
    b.add_input<Geometry>("Input");

    // Input-2: An embedding result of the mesh. Use the XY coordinates of the
    // embedding as the initialization of the ARAP algorithm
    //
    // Here we use **the result of Assignment 4** as the initialization
    b.add_input<Geometry>("Initialization");

    // Output-1: Like the result of Assignment 4, output the 2D embedding of the
    // mesh
    b.add_output<Geometry>("Output");
}

NODE_EXECUTION_FUNCTION(arap_parameterization)
{
    // Get the input from params
    auto input = params.get_input<Geometry>("Input");
    auto iters = params.get_input<Geometry>("Initialization");

    // Avoid processing the node when there is no input
    if (!input.get_component<MeshComponent>() ||
        !iters.get_component<MeshComponent>()) {
        std::cerr << "ARAP Parameterization: Need Geometry Input." << std::endl;
    }

    /* ----------------------------- Preprocess -------------------------------
    ** Create a halfedge structure (using OpenMesh) for the input mesh. The
    ** half-edge data structure is a widely used data structure in geometric
    ** processing, offering convenient operations for traversing and modifying
    ** mesh elements.
    */

    // Initialization
    auto halfedge_mesh = operand_to_openmesh(&input);
    auto iter_mesh = operand_to_openmesh(&iters);

    // ARAP parameterization
    arap(halfedge_mesh, iter_mesh);

    normalize_uvs(*iter_mesh);
    auto geometry = openmesh_to_operand(iter_mesh.get());

    // Set the output of the nodes
    params.set_output("Output", std::move(*geometry));
    return true;
}

NODE_DECLARATION_UI(arap_parameterization);
NODE_DEF_CLOSE_SCOPE
