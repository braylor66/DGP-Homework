#include <igl/arap.h>
#include <igl/cotmatrix.h>
#include <igl/min_quad_with_fixed.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <cmath>
#include <cstddef>
#include <memory>
#include <vector>

#include "GCore/Components/MeshOperand.h"
#include "GCore/util_openmesh_bind.h"
#include "geom_node_base.h"
#include "igl/ARAPEnergyType.h"
#include "nodes/core/def/node_def.hpp"

typedef OpenMesh::PolyMesh_ArrayKernelT<> MyMesh;

void arap_deformation(
    std::shared_ptr<MyMesh> halfedge_mesh,
    std::vector<size_t> indices,
    std::vector<std::array<float, 3>> new_positions)
{
    // TODO: Implement ARAP Deformation Algorithm.
    const int n_vertices = halfedge_mesh->n_vertices();
    if (n_vertices == 0)
        return;

    // Step 1: Initial Setup - Build Cotangent Laplacian L and apply
    // constraints.
    std::vector<Eigen::Triplet<double>> L_triplets;
    L_triplets.reserve(n_vertices * 7);

    std::vector<bool> is_control(n_vertices, false);
    for (const auto& index : indices) {
        if (index < n_vertices) {
            is_control[index] = true;
        }
    }
    for (const auto& index : indices) {
        if (index < n_vertices) {
            L_triplets.emplace_back(index, index, 1.0);
        }
    }

    for (const auto& eh : halfedge_mesh->edges()) {
        const auto heh0 = halfedge_mesh->halfedge_handle(eh, 0);
        const auto heh1 = halfedge_mesh->halfedge_handle(eh, 1);
        const int v0_idx = halfedge_mesh->from_vertex_handle(heh0).idx();
        const int v1_idx = halfedge_mesh->to_vertex_handle(heh0).idx();

        double cotan_sum = 0.0;
        // Calculate cotangent weight from adjacent faces.
        for (const auto& heh : { heh0, heh1 }) {
            if (!halfedge_mesh->is_boundary(heh)) {
                const auto opp_vh = halfedge_mesh->to_vertex_handle(
                    halfedge_mesh->next_halfedge_handle(heh));
                const auto& p0 = halfedge_mesh->point(
                    halfedge_mesh->from_vertex_handle(heh));
                const auto& p1 =
                    halfedge_mesh->point(halfedge_mesh->to_vertex_handle(heh));
                const auto& p_opp = halfedge_mesh->point(opp_vh);

                const auto vec_a = (p0 - p_opp).normalized();
                const auto vec_b = (p1 - p_opp).normalized();

                double dot_val = vec_a | vec_b;
                dot_val = std::max(-1.0, std::min(1.0, dot_val));
                cotan_sum += 1.0 / std::tan(std::acos(dot_val));
            }
        }
        const double weight = 0.5 * std::max(0.0, cotan_sum);

        if (!is_control[v0_idx] && !is_control[v1_idx]) {
            L_triplets.emplace_back(v0_idx, v0_idx, weight);
            L_triplets.emplace_back(v1_idx, v1_idx, weight);
            L_triplets.emplace_back(v0_idx, v1_idx, -weight);
            L_triplets.emplace_back(v1_idx, v0_idx, -weight);
        }
        else if (!is_control[v0_idx]) {  // v1 is control point
            L_triplets.emplace_back(v0_idx, v0_idx, weight);
            L_triplets.emplace_back(v0_idx, v1_idx, -weight);
        }
        else if (!is_control[v1_idx]) {  // v0 is control point
            L_triplets.emplace_back(v1_idx, v1_idx, weight);
            L_triplets.emplace_back(v1_idx, v0_idx, -weight);
        }
    }

    Eigen::SparseMatrix<double> L(n_vertices, n_vertices);
    L.setFromTriplets(L_triplets.begin(), L_triplets.end());

    // Step 2: Pre-Factorization
    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    solver.compute(L);
    if (solver.info() != Eigen::Success) {
        std::cerr << "ARAP Deformation: SparseLU decomposition failed."
                  << std::endl;
        return;
    }

    // Step 3: Initialization
    Eigen::MatrixXd P0(n_vertices, 3), P(n_vertices, 3);
    for (const auto& vh : halfedge_mesh->vertices()) {
        const int idx = vh.idx();
        const auto& pt = halfedge_mesh->point(vh);
        P0.row(idx) << pt[0], pt[1], pt[2];
        P.row(idx) << pt[0], pt[1], pt[2];
    }

    for (size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] < n_vertices) {
            P.row(indices[i]) << new_positions[i][0], new_positions[i][1],
                new_positions[i][2];
        }
    }

    // Step 4: Iteration (Local-Global)
    const int max_iter = 10;
    std::vector<Eigen::Matrix3d> R(n_vertices);

    for (int iter = 0; iter < max_iter; ++iter) {
        // Local Phase
        for (const auto& vh : halfedge_mesh->vertices()) {
            const int i = vh.idx();
            if (is_control[i]) {
                R[i].setIdentity();
                continue;
            }

            Eigen::Matrix3d Ci = Eigen::Matrix3d::Zero();
            for (auto it = halfedge_mesh->vv_iter(vh); it.is_valid(); ++it) {
                const int j = it->idx();
                // L(i, j) = -w_ij.
                const double w_ij = -L.coeff(i, j);
                if (w_ij > 0) {  // Consider only neighbors with positive weight
                    Ci += w_ij * (P.row(i) - P.row(j)).transpose() *
                          (P0.row(i) - P0.row(j));
                }
            }

            Eigen::JacobiSVD<Eigen::Matrix3d> svd(
                Ci, Eigen::ComputeFullU | Eigen::ComputeFullV);
            const Eigen::Matrix3d& U = svd.matrixU();
            const Eigen::Matrix3d& V = svd.matrixV();
            Eigen::Matrix3d Ri = V * U.transpose();

            if (Ri.determinant() < 0.0) {
                Eigen::Matrix3d V_prime = V;
                V_prime.col(2) *= -1.0;
                Ri = V_prime * U.transpose();
            }
            R[i] = Ri;
        }

        // Global Phase
        Eigen::MatrixXd b = Eigen::MatrixXd::Zero(n_vertices, 3);

        for (const auto& vh : halfedge_mesh->vertices()) {
            const int i = vh.idx();
            if (is_control[i]) {
                b.row(i) = P.row(i);
            }
            else {
                Eigen::Vector3d b_i = Eigen::Vector3d::Zero();
                for (auto it = halfedge_mesh->vv_iter(vh); it.is_valid();
                     ++it) {
                    const int j = it->idx();
                    const double w_ij = -L.coeff(i, j);
                    if (w_ij > 0) {
                        b_i += w_ij * 0.5 * (R[i] + R[j]) *
                               (P0.row(i) - P0.row(j)).transpose();
                    }
                }
                b.row(i) = b_i;
            }
        }

        // Solve
        P.col(0) = solver.solve(b.col(0));
        P.col(1) = solver.solve(b.col(1));
        P.col(2) = solver.solve(b.col(2));

        if (solver.info() != Eigen::Success) {
            std::cerr << "ARAP Deformation: Sparse solve failed in iteration "
                      << iter << std::endl;
            for (const auto& vh : halfedge_mesh->vertices()) {
                const int idx = vh.idx();
                halfedge_mesh->point(vh) =
                    MyMesh::Point(P0(idx, 0), P0(idx, 1), P0(idx, 2));
            }
            return;
        }
    }

    // Step 5: Finalization
    for (const auto& vh : halfedge_mesh->vertices()) {
        const int idx = vh.idx();
        halfedge_mesh->point(vh) = MyMesh::Point(
            static_cast<float>(P(idx, 0)),
            static_cast<float>(P(idx, 1)),
            static_cast<float>(P(idx, 2)));
    }
}

NODE_DEF_OPEN_SCOPE
NODE_DECLARATION_FUNCTION(arap_deformation)
{
    // Input-1: Original 3D mesh with boundary
    b.add_input<Geometry>("Input");

    // Input-2: Indices of the control vertices
    b.add_input<std::vector<size_t>>("Indices");

    // Input-3: New positions for the control vertices
    b.add_input<std::vector<std::array<float, 3>>>("New Positions");

    // Output-1: Deformed mesh
    b.add_output<Geometry>("Output");
}

NODE_EXECUTION_FUNCTION(arap_deformation)
{
    // Get the input from params
    auto input = params.get_input<Geometry>("Input");
    auto indices = params.get_input<std::vector<size_t>>("Indices");
    auto new_positions =
        params.get_input<std::vector<std::array<float, 3>>>("New Positions");

    if (indices.empty() || new_positions.empty()) {
        std::cerr << "ARAP Deformation: Please set control points."
                  << std::endl;
        return false;
    }

    if (indices.size() != new_positions.size()) {
        std::cerr << "ARAP Deformation: The size of indices and new positions "
                     "should be the same."
                  << std::endl;
        return false;
    }

    /* ----------------------------- Preprocess -------------------------------
     ** Create a halfedge structure (using OpenMesh) for the input mesh. The
     ** half-edge data structure is a widely used data structure in geometric
     ** processing, offering convenient operations for traversing and modifying
     ** mesh elements.
     */

    // Initialization
    auto halfedge_mesh = operand_to_openmesh(&input);

    // ARAP deformation
    arap_deformation(halfedge_mesh, indices, new_positions);

    auto geometry = openmesh_to_operand(halfedge_mesh.get());

    // Set the output of the nodes
    params.set_output("Output", std::move(*geometry));
    return true;
}

NODE_DECLARATION_UI(arap_deformation);
NODE_DEF_CLOSE_SCOPE
