#include <functional>

#include "GCore/Components/MeshOperand.h"
#include "geom_node_base.h"
#include "nodes/core/def/node_def.hpp"

NODE_DEF_OPEN_SCOPE
NODE_DECLARATION_FUNCTION(mvc)
{
    b.add_input<Geometry>("Mesh");
    b.add_output<std::function<std::vector<float>(float, float)>>(
        "Mean Value Coordinates");
}

NODE_EXECUTION_FUNCTION(mvc)
{
    // Get the input mesh
    auto geometry = params.get_input<Geometry>("Mesh");
    auto mesh = geometry.get_component<MeshComponent>();

    if (!mesh) {
        std::cerr
            << "MVC Node: Failed to get MeshComponent from input geometry."
            << std::endl;
        return false;
    }

    auto vertices = mesh->get_vertices();
    auto face_vertex_counts = mesh->get_face_vertex_counts();
    auto face_vertex_indices = mesh->get_face_vertex_indices();

    if (vertices.size() < 3 || face_vertex_counts.size() != 1 ||
        face_vertex_counts[0] != vertices.size()) {
        std::cerr << "MVC Node: Input mesh must be a single polygon with at "
                     "least 3 vertices. "
                  << "Provided: " << vertices.size() << " vertices, "
                  << face_vertex_counts.size() << " faces. "
                  << "First face has "
                  << (face_vertex_counts.empty() ? 0 : face_vertex_counts[0])
                  << " vertices." << std::endl;

        return false;
    }

    for (const auto& vertex : vertices) {
        if (std::abs(vertex[2]) > 1e-5) {
            std::cerr << "MVC Node: Input mesh must be a 2D polygon on the XY "
                         "plane. Found vertex with Z-coordinate: "
                      << vertex[2] << std::endl;
            return false;
        }
    }

    std::vector<std::array<float, 2>> polygon_vertices;
    for (int i = 0; i < face_vertex_counts[0]; i++) {
        auto vertex = vertices[face_vertex_indices[i]];
        polygon_vertices.push_back({ vertex[0], vertex[1] });
    }

    auto mvc_function = [captured_vertices = polygon_vertices](
                            float p_x, float p_y) -> std::vector<float> {
        const size_t num_vertices = captured_vertices.size();
        if (num_vertices == 0) {
            return {};
        }

        const float epsilon = 1e-6f;
        std::vector<float> distances(num_vertices);

        // Step 1: Compute distances and check for vertex coincidence.
        for (size_t i = 0; i < num_vertices; ++i) {
            const float dx = captured_vertices[i][0] - p_x;
            const float dy = captured_vertices[i][1] - p_y;
            distances[i] = std::sqrt(dx * dx + dy * dy);

            if (distances[i] < epsilon) {
                std::vector<float> result_weights(num_vertices, 0.0f);
                result_weights[i] = 1.0f;
                return result_weights;
            }
        }

        // Step 2: Compute tan(alpha_i / 2), and handle boundary cases.
        std::vector<float> tan_half_angles(num_vertices);
        for (size_t i = 0; i < num_vertices; ++i) {
            const size_t i_next = (i + 1) % num_vertices;

            const float dist_i = distances[i];
            const float dist_i_next = distances[i_next];

            const float edge_dx =
                captured_vertices[i_next][0] - captured_vertices[i][0];
            const float edge_dy =
                captured_vertices[i_next][1] - captured_vertices[i][1];
            const float edge_len_squared =
                edge_dx * edge_dx + edge_dy * edge_dy;

            double cos_alpha = (dist_i * dist_i + dist_i_next * dist_i_next -
                                edge_len_squared) /
                               (2.0 * dist_i * dist_i_next);

            if (std::abs(cos_alpha + 1.0) < epsilon) {
                std::vector<float> result_weights(num_vertices, 0.0f);
                float edge_len = std::sqrt(edge_len_squared);
                if (edge_len > epsilon) {
                    // Linear interpolation based on distance
                    result_weights[i] = dist_i_next / edge_len;
                    result_weights[i_next] = dist_i / edge_len;
                }
                return result_weights;
            }

            cos_alpha = std::max(-1.0, std::min(1.0, cos_alpha));

            tan_half_angles[i] = static_cast<float>(
                std::sqrt((1.0 - cos_alpha) / (1.0 + cos_alpha)));
        }

        std::vector<float> unnormalized_weights(num_vertices);
        float weight_sum = 0.0f;
        for (size_t i = 0; i < num_vertices; ++i) {
            const size_t i_prev = (i + num_vertices - 1) % num_vertices;
            unnormalized_weights[i] =
                (tan_half_angles[i_prev] + tan_half_angles[i]) / distances[i];
            weight_sum += unnormalized_weights[i];
        }

        if (std::abs(weight_sum) > epsilon) {
            for (size_t i = 0; i < num_vertices; ++i) {
                unnormalized_weights[i] /= weight_sum;
            }
        }
        else {
            float uniform_weight = 1.0f / static_cast<float>(num_vertices);
            std::fill(
                unnormalized_weights.begin(),
                unnormalized_weights.end(),
                uniform_weight);
        }

        return unnormalized_weights;
    };

    // Set the output of the node
    params.set_output(
        "Mean Value Coordinates",
        std::function<std::vector<float>(float, float)>(mvc_function));
    return true;
}

NODE_DECLARATION_UI(mvc);

NODE_DEF_CLOSE_SCOPE
