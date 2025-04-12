#include "minimal_solver.h"
#include "utility.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <random>
#include <tuple>
#include <vector>

// Generates random 3D points in front of the camera.
std::vector<Eigen::Vector3d> Generate3DPoints(int num_points)
{
    std::random_device               rd;
    std::mt19937                     gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    std::vector<Eigen::Vector3d> points;
    points.reserve(num_points);

    for (int i = 0; i < num_points; ++i)
    {
        double x = dis(gen);
        double y = dis(gen);
        double z = dis(gen) + 3.0; // ensure positive depth
        points.emplace_back(x, y, z);
    }
    return points;
}

// Projects 3D points into 2D using intrinsic and extrinsic parameters.
std::vector<Eigen::Vector2d> ProjectPoints(const std::vector<Eigen::Vector3d>& points,
                                           const Eigen::Matrix3d& k, const Eigen::Matrix3d& r,
                                           const Eigen::Vector3d& t)
{
    std::vector<Eigen::Vector2d> image_points;
    image_points.reserve(points.size());

    for (const auto& point : points)
    {
        Eigen::Vector3d x_cam = r * point + t;
        Eigen::Vector3d x_img = k * x_cam;

        double u = x_img.x() / x_img.z();
        double v = x_img.y() / x_img.z();

        image_points.emplace_back(u, v);
    }

    return image_points;
}

std::vector<Eigen::Vector2d> NormalizePoints(const std::vector<Eigen::Vector2d>& image_points,
                                             const Eigen::Matrix3d&              K)
{
    Eigen::Matrix3d              K_inv = K.inverse();
    std::vector<Eigen::Vector2d> normalized;
    normalized.reserve(image_points.size());

    for (const auto& pt : image_points)
    {
        Eigen::Vector3d pt_h(pt.x(), pt.y(), 1.0); // homogeneous coordinates
        Eigen::Vector3d norm_pt = K_inv * pt_h;
        normalized.emplace_back(norm_pt.x(), norm_pt.y());
    }

    return normalized;
}

struct Pose
{
    Eigen::Matrix3d rotation;
    Eigen::Vector3d translation;
};

// Convert degrees to radians
inline double deg2rad(double degrees)
{
    return degrees * M_PI / 180.0;
}

// Generate pose with rotations around X, Y, Z and translation tx, ty, tz
Pose GeneratePose(double roll_deg, double pitch_deg, double yaw_deg, double tx, double ty,
                  double tz)
{
    double roll  = deg2rad(roll_deg);  // rotation around X
    double pitch = deg2rad(pitch_deg); // rotation around Y
    double yaw   = deg2rad(yaw_deg);   // rotation around Z

    // Rotation matrices around each axis
    Eigen::Matrix3d Rx;
    Rx << 1, 0, 0, 0, cos(roll), -sin(roll), 0, sin(roll), cos(roll);

    Eigen::Matrix3d Ry;
    Ry << cos(pitch), 0, sin(pitch), 0, 1, 0, -sin(pitch), 0, cos(pitch);

    Eigen::Matrix3d Rz;
    Rz << cos(yaw), -sin(yaw), 0, sin(yaw), cos(yaw), 0, 0, 0, 1;

    // Combined rotation: R = Rz * Ry * Rx
    Eigen::Matrix3d R = Rz * Ry * Rx;
    Eigen::Vector3d t(tx, ty, tz);

    return {R, t};
}

double PoseError(const Eigen::Matrix3d& R1, const Eigen::Vector3d& t1, const Eigen::Matrix3d& R2,
                 const Eigen::Vector3d& t2)
{
    Eigen::Matrix3d R_rel       = R1.transpose() * R2;
    double          cos_angle_R = (R_rel.trace() - 1.0) / 2.0;
    cos_angle_R                 = std::clamp(cos_angle_R, -1.0, 1.0); // Ensure numerical stability
    double rot_error_deg        = std::acos(cos_angle_R) * 180.0 / M_PI;

    Eigen::Vector3d t1_norm     = t1.normalized();
    Eigen::Vector3d t2_norm     = t2.normalized();
    double          cos_angle_t = t1_norm.dot(t2_norm);
    cos_angle_t                 = std::clamp(cos_angle_t, -1.0, 1.0);
    double trans_error_deg      = std::acos(cos_angle_t) * 180.0 / M_PI;

    return rot_error_deg + trans_error_deg;
}

int main()
{
    Eigen::Matrix3d k;
    k << 800, 0, 320, 0, 800, 240, 0, 0, 1;

    // Set up random generators
    std::random_device               rd;
    std::mt19937                     gen(rd());
    std::uniform_real_distribution<> rot_dist_deg(-10.0, 10.0); // in degrees
    std::uniform_real_distribution<> trans_dist(-1.0, 1.0);     // for t

    // Generate rotation angles in deg
    const double rx = rot_dist_deg(gen);
    const double ry = rot_dist_deg(gen);
    const double rz = rot_dist_deg(gen);

    // Generate and normalize translation vector
    Eigen::Vector3d position(trans_dist(gen), trans_dist(gen), trans_dist(gen));
    position.normalize();

    const double tx = position.x();
    const double ty = position.y();
    const double tz = position.z();

    auto pose = GeneratePose(rx, ry, rz, tx, ty, tz);

    Eigen::Matrix3d r = pose.rotation;
    Eigen::Vector3d t = pose.translation;
    std::cout << "R (gt) = \n" << r << std::endl;
    std::cout << "t (gt) = \t" << t.transpose() << std::endl;

    const auto points_3d_all = Generate3DPoints(100);

    const auto points_cam1_all =
        ProjectPoints(points_3d_all, k, Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());
    const auto points_cam2_all = ProjectPoints(points_3d_all, k, r, t);

    const auto points_cam1_all_normalized = NormalizePoints(points_cam1_all, k);
    const auto points_cam2_all_normalized = NormalizePoints(points_cam2_all, k);

    // Randomly pick 5 indices
    std::vector<int> indices(100);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), std::mt19937{std::random_device{}()});
    std::vector<int> sample_indices(indices.begin(), indices.begin() + 5);

    std::vector<Eigen::Vector2d> points_cam1_sampled, points_cam2_sampled;
    for (int idx : sample_indices)
    {
        points_cam1_sampled.push_back(points_cam1_all_normalized[idx]);
        points_cam2_sampled.push_back(points_cam2_all_normalized[idx]);
    }

    const auto [essential_matrices, rotation_matrices, translation_vectors] =
        minimal_solver::FindEssentialMatMinimalSolver(points_cam1_sampled, points_cam2_sampled);

    double min_sampson = std::numeric_limits<double>::max();
    int    best_index  = -1;

    for (size_t i = 0; i < essential_matrices.size(); ++i)
    {
        const auto&  essential_matrix = essential_matrices[i];
        const double sampson          = std::fabs(ComputeEpipolarConstraint(
                     points_cam1_all_normalized, points_cam2_all_normalized, essential_matrix));
        if (sampson < min_sampson)
        {
            min_sampson = sampson;
            best_index  = static_cast<int>(i);
        }
    }
    std::cout << "Picking the Essential Matrix that yields the minimum absolute Sampson distance."
              << std::endl;
    Eigen::Matrix3d E_best = essential_matrices[best_index];
    Eigen::Matrix3d R_best = rotation_matrices[best_index];
    Eigen::Vector3d t_best = translation_vectors[best_index];

    std::cout << "E (best) = \n" << E_best << std::endl;
    std::cout << "R (best) = \n" << R_best << std::endl;
    std::cout << "t (best) = \t" << t_best.transpose() << std::endl;

    std::cout << "Pose error: " << PoseError(r, t, R_best, t_best) << std::endl;
    return 0;
}
