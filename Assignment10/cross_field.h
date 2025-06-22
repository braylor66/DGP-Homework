#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <complex>
#include <vector>

#include "Eigen/Eigen"
#include "Eigen/src/Core/Matrix.h"
#include "Eigen/src/SparseCore/SparseUtil.h"
#include "GCore/api.h"
#include "GCore/util_openmesh_bind.h"
#include "OpenMesh/Core/Geometry/EigenVectorT.hh"
#include "OpenMesh/Core/Mesh/Traits.hh"
#include "cmath"

// Define the traits for OpenMesh to use Eigen types
struct EigenTraits : OpenMesh::DefaultTraits {
    using Point = Eigen::Vector3d;
    using Normal = Eigen::Vector3d;
    using TexCoord2D = Eigen::Vector2d;
};

typedef OpenMesh::PolyMesh_ArrayKernelT<EigenTraits> EigenPolyMesh;

class CrossField {
   public:
    CrossField(EigenPolyMesh& _mesh);

    // Create constraints for the frame field
    // Since manually setting constraints is difficult,
    // we will generate constraints based on the mesh boundary
    void generateBoundaryConstrain();

    // Generate the corss field
    void generateCrossField();

    // Get the frame field for a specific face
    std::vector<std::vector<Eigen::Vector3d>> getCrossFields();
    std::vector<Eigen::Vector3d> getCrossFields(OpenMesh::FaceHandle fh);
    std::vector<Eigen::Vector3d> getCrossFields(int _fid);

   private:
    // Openmesh polymesh
    EigenPolyMesh* mesh;

    // Local coordinates for each face
    std::vector<std::array<Eigen::Vector3d, 2>> localCoords;

    // P(z) = z^4 - u^4

    // For each face, we store u
    std::vector<std::complex<double>> allComplex;

    // Constraints for the frame field
    std::unordered_map<OpenMesh::FaceHandle, std::complex<double>> constrains;

    // Generate local coordinates
    void generateLocalCoordinates();
};

typedef Eigen::SparseMatrix<std::complex<double>> SpMat;
typedef Eigen::Triplet<std::complex<double>> T;

CrossField::CrossField(EigenPolyMesh& _mesh) : mesh(&_mesh)
{
    allComplex.resize(mesh->n_faces());
    localCoords.resize(mesh->n_faces());

    mesh->request_vertex_status();
    mesh->request_edge_status();
    mesh->request_face_status();
    mesh->request_halfedge_status();
    mesh->request_face_normals();
    mesh->request_vertex_normals();

    generateLocalCoordinates();
}

void CrossField::generateBoundaryConstrain()
{
    // Set up constraints for the frame field
    // Prependicular to the boundary edges
    for (auto it = mesh->edges_sbegin(); it != mesh->edges_end(); it++) {
        if (!it->is_boundary()) {
            continue;
        }
        auto eh = *it;
        OpenMesh::SmartFaceHandle fh;
        if (eh.h0().face().is_valid())
            fh = eh.h0().face();
        else
            fh = eh.h1().face();
        // assert(fh.is_valid());
        if (!fh.is_valid()) {
            continue;
        }

        Eigen::Vector3d N = mesh->calc_face_normal(fh);
        Eigen::Vector3d p0 = mesh->point(eh.v0());
        Eigen::Vector3d p1 = mesh->point(eh.v1());
        Eigen::Vector3d dir = (p0 - p1).normalized();
        // Vector perpendicular to the boundary
        Eigen::Vector3d pp = dir.cross(N);

        // Set constraints
        double cos = pp.dot(localCoords[fh.idx()][0]);
        double sin = pp.dot(localCoords[fh.idx()][1]);

        // u = cos + i sin
        allComplex[fh.idx()] = std::complex<double>(cos, sin);
        constrains[fh] = allComplex[fh.idx()];
    }
}

void CrossField::generateCrossField()
{
    // TODO: Generate the cross field for the mesh

    int nFaces = mesh->n_faces();
    if (nFaces == 0) {
        return;
    }


    std::vector<T> triplets; 

    int nInteriorEdges = 0;
    for (auto eh : mesh->edges()) {
        if (!eh.is_boundary()) {
            nInteriorEdges++;
        }
    }
    int nConstraints = constrains.size();
    int nRows = nInteriorEdges + nConstraints; 

    Eigen::VectorXcd b = Eigen::VectorXcd::Zero(nRows); 
    int currentRow = 0;

    for (auto eh : mesh->edges()) {
        if (eh.is_boundary()) {
            continue;
        }

        auto heh0 = mesh->halfedge_handle(eh, 0);
        auto heh1 = mesh->halfedge_handle(eh, 1);
        auto f0 = mesh->face_handle(heh0);
        auto f1 = mesh->face_handle(heh1);

        if (!f0.is_valid() || !f1.is_valid())
            continue;

        int idx0 = f0.idx();
        int idx1 = f1.idx();

        const auto& basis0 = localCoords[idx0];  
        const auto& basis1 = localCoords[idx1];  

        std::complex<double> rho(
            basis1[0].dot(basis0[0]), basis1[0].dot(basis0[1]));

        std::complex<double> rho4 = std::pow(rho, 4.0);

        triplets.emplace_back(currentRow, idx0, 1.0);
        triplets.emplace_back(currentRow, idx1, -rho4);

        currentRow++;
    }

    double weight = 100.0; 
    for (auto const& [fh, u_val] : constrains) {
        int idx = fh.idx();

        triplets.emplace_back(currentRow, idx, weight);

        std::complex<double> c4 = std::pow(u_val, 4.0);
        b(currentRow) = weight * c4;

        currentRow++;
    }

    SpMat A(nRows, nFaces);
    A.setFromTriplets(triplets.begin(), triplets.end());

    SpMat At = A.transpose();
    SpMat AtA = At * A;
    Eigen::VectorXcd Atb = At * b;

    Eigen::SimplicialLDLT<SpMat> solver;
    solver.compute(AtA);

    if (solver.info() != Eigen::Success) {
        throw("CrossField generation: LDLT decomposition failed.");
        return;
    }

    Eigen::VectorXcd x = solver.solve(Atb);

    if (solver.info() != Eigen::Success) {
        throw("CrossField generation: Linear solve failed.");
        return;
    }

    for (int i = 0; i < nFaces; ++i) {
        allComplex[i] = std::pow(x(i), 1.0 / 4.0);
    }
}

std::vector<std::vector<Eigen::Vector3d>> CrossField::getCrossFields()
{
    std::vector<std::vector<Eigen::Vector3d>> ret;
    for (int i = 0; i < mesh->n_faces(); i++) {
        ret.push_back(getCrossFields(i));
    }
    return ret;
}

std::vector<Eigen::Vector3d> CrossField::getCrossFields(OpenMesh::FaceHandle fh)
{
    return getCrossFields(fh.idx());
}

std::vector<Eigen::Vector3d> CrossField::getCrossFields(int _fid)
{
    std::vector<Eigen::Vector3d> ret;
    if (allComplex.size() <= _fid || _fid < 0)
        return ret;
    auto c = allComplex[_fid];
    auto& localCoord = localCoords[_fid];
    // 其中一条向量
    Eigen::Vector3d dir = c.real() * localCoord[0] + c.imag() * localCoord[1];
    auto N = mesh->calc_face_normal(mesh->face_handle(_fid));
    // 第一条向量
    ret.push_back(dir);
    for (int i = 0; i < 3; i++) {
        Eigen::Vector3d d = N.cross(dir);
        dir = d;
        ret.push_back(dir);
    }
    return ret;
}

void CrossField::generateLocalCoordinates()
{
    // TODO: Generate local coordinates for each face, fill in localCoords
    for (auto fh : mesh->faces()) {
        Eigen::Vector3d N = mesh->normal(fh);
        N.normalize();

        auto heh = mesh->halfedge_handle(fh);
        auto v_from = mesh->from_vertex_handle(heh);
        auto v_to = mesh->to_vertex_handle(heh);
        Eigen::Vector3d p_from = mesh->point(v_from);
        Eigen::Vector3d p_to = mesh->point(v_to);
        Eigen::Vector3d e1 = (p_to - p_from).normalized();
        Eigen::Vector3d e2 = N.cross(e1).normalized();
        localCoords[fh.idx()][0] = e1;
        localCoords[fh.idx()][1] = e2;
    }
}
