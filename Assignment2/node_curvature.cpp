#include <pxr/base/vt/array.h>

#include <limits>
#include <vector>

#include "GCore/Components/MeshOperand.h"
#include "OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh"
#include "geom_node_base.h"
#include "nodes/core/def/node_def.hpp"

typedef OpenMesh::TriMesh_ArrayKernelT<> MyMesh;

using Vec = MyMesh::Point;
constexpr float EPS = 1e-12f;

static inline float cotangent(const Vec& a, const Vec& b)
{
    Vec c = OpenMesh::cross(a, b);
    float n = c.norm();
    if (n < EPS)
        n = EPS;  // Avoid division by zero
    return OpenMesh::dot(a, b) / n;
}

// ---- Mean Curvature ----
void compute_mean_curvature(
    const MyMesh& mesh,
    pxr::VtArray<float>& mean_curvature)
{
    mean_curvature.assign(mesh.n_vertices(), 0.f);

    for (auto v_it = mesh.vertices_begin(); v_it != mesh.vertices_end();
         ++v_it) {
        auto vh = *v_it;
        Vec xi = mesh.point(vh);
        Vec lap(0, 0, 0);

        // Iterate over outgoing half-edges from vh
        for (auto voh_it = mesh.cvoh_iter(vh); voh_it.is_valid(); ++voh_it) {
            auto heh = *voh_it;
            auto vj = mesh.to_vertex_handle(heh);
            Vec xj = mesh.point(vj);

            float w = 0.f;

            // First adjacent face
            if (!mesh.is_boundary(heh)) {
                auto vk = mesh.to_vertex_handle(mesh.next_halfedge_handle(heh));
                w += cotangent(mesh.point(vk) - xi, mesh.point(vk) - xj);
            }
            // Opposite adjacent face
            auto heh_op = mesh.opposite_halfedge_handle(heh);
            if (!mesh.is_boundary(heh_op)) {
                auto vk =
                    mesh.to_vertex_handle(mesh.next_halfedge_handle(heh_op));
                w += cotangent(mesh.point(vk) - xi, mesh.point(vk) - xj);
            }

            lap += w * (xj - xi);
        }

        // Calculate one-ring area
        float area = 0.f;
        for (auto vf_it = mesh.cvf_iter(vh); vf_it.is_valid(); ++vf_it)
            area += mesh.calc_face_area(*vf_it);
        area *= 1.f / 3.f;
        if (area < EPS)
            continue;

        Vec Hvec = lap / (2.f * area);
        float H = 0.5f * Hvec.norm();

        mean_curvature[vh.idx()] = H;
    }
}

// ---- Gaussian Curvature ----
void compute_gaussian_curvature(
    const MyMesh& mesh,
    pxr::VtArray<float>& gaussian_curvature)
{
    gaussian_curvature.assign(mesh.n_vertices(), 0.f);

    for (auto v_it = mesh.vertices_begin(); v_it != mesh.vertices_end();
         ++v_it) {
        auto vh = *v_it;
        float angle_sum = 0.f;

        // Sum angles of surrounding faces
        for (auto vf_it = mesh.cvf_iter(vh); vf_it.is_valid(); ++vf_it) {
            auto fh = *vf_it;

            // Get triangle vertices
            MyMesh::ConstFaceVertexIter fv_it = mesh.cfv_iter(fh);
            MyMesh::VertexHandle v0 = *fv_it;
            ++fv_it;
            MyMesh::VertexHandle v1 = *fv_it;
            ++fv_it;
            MyMesh::VertexHandle v2 = *fv_it;

            // Find which vertex of the face is vh
            int idx;
            if (vh == v0)
                idx = 0;
            else if (vh == v1)
                idx = 1;
            else
                idx = 2;

            const Vec& p0 = mesh.point(v0);
            const Vec& p1 = mesh.point(v1);
            const Vec& p2 = mesh.point(v2);

            // Calculate angle at vh
            Vec a, b;
            if (idx == 0) {
                a = p1 - p0;
                b = p2 - p0;
            }
            else if (idx == 1) {
                a = p0 - p1;
                b = p2 - p1;
            }
            else {
                a = p0 - p2;
                b = p1 - p2;
            }

            float denom = a.norm() * b.norm();
            float cosang = (denom < EPS) ? 1.f : OpenMesh::dot(a, b) / denom;
            cosang = std::clamp(cosang, -1.f, 1.f);
            angle_sum += std::acos(cosang);
        }

        float theta0 = mesh.is_boundary(vh) ? static_cast<float>(M_PI)
                                            : static_cast<float>(2.0 * M_PI);

        // Calculate one-ring area
        float area = 0.f;
        for (auto vf_it = mesh.cvf_iter(vh); vf_it.is_valid(); ++vf_it)
            area += mesh.calc_face_area(*vf_it);
        area *= 1.f / 3.f;
        if (area < EPS)
            continue;

        // K = (2*PI - sum of angles) / Area
        gaussian_curvature[vh.idx()] = (theta0 - angle_sum) / area;
    }
}

NODE_DEF_OPEN_SCOPE

NODE_DECLARATION_FUNCTION(mean_curvature)
{
    b.add_input<Geometry>("Mesh");
    b.add_output<pxr::VtArray<float>>("Mean Curvature");
}

NODE_EXECUTION_FUNCTION(mean_curvature)
{
    auto geometry = params.get_input<Geometry>("Mesh");
    auto mesh = geometry.get_component<MeshComponent>();
    auto vertices = mesh->get_vertices();
    auto face_vertex_indices = mesh->get_face_vertex_indices();
    auto face_vertex_counts = mesh->get_face_vertex_counts();

    // Convert the mesh to OpenMesh
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

    // Compute mean curvature
    pxr::VtArray<float> mean_curvature_values;
    mean_curvature_values.reserve(omesh.n_vertices());
    compute_mean_curvature(omesh, mean_curvature_values);
    params.set_output("Mean Curvature", mean_curvature_values);

    return true;
}

NODE_DECLARATION_UI(mean_curvature);

NODE_DECLARATION_FUNCTION(gaussian_curvature)
{
    b.add_input<Geometry>("Mesh");
    b.add_output<pxr::VtArray<float>>("Gaussian Curvature");
}

NODE_EXECUTION_FUNCTION(gaussian_curvature)
{
    auto geometry = params.get_input<Geometry>("Mesh");
    auto mesh = geometry.get_component<MeshComponent>();
    auto vertices = mesh->get_vertices();
    auto face_vertex_indices = mesh->get_face_vertex_indices();
    auto face_vertex_counts = mesh->get_face_vertex_counts();

    // Convert the mesh to OpenMesh
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

    // Compute Gaussian curvature
    pxr::VtArray<float> gaussian_curvature_values;
    gaussian_curvature_values.reserve(omesh.n_vertices());
    compute_gaussian_curvature(omesh, gaussian_curvature_values);
    params.set_output("Gaussian Curvature", gaussian_curvature_values);

    return true;
}

NODE_DECLARATION_UI(gaussian_curvature);

NODE_DEF_CLOSE_SCOPE
