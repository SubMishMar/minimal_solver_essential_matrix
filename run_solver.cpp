#include "minimal_solver.h"

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

int main()
{
    Eigen::Matrix3d k;
    k << 800, 0, 320, 0, 800, 240, 0, 0, 1;

    auto pose = GeneratePose(-5.0, 5.0, 10.0, 0.1, -0.1, -0.5);

    Eigen::Matrix3d r = pose.rotation;
    Eigen::Vector3d t = pose.translation;

    const auto points_3d = Generate3DPoints(5);
    const auto points_cam1 =
        ProjectPoints(points_3d, k, Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());
    const auto points_cam1_normalized = NormalizePoints(points_cam1, k);
    const auto points_cam2            = ProjectPoints(points_3d, k, r, t);
    const auto points_cam2_normalized = NormalizePoints(points_cam2, k);

    const auto [essential_matrices, rotation_matrices, translation_vectors] =
        minimal_solver::FindEssentialMatMinimalSolver(points_cam1_normalized,
                                                      points_cam2_normalized, k);

    for (const auto essential_matrix : essential_matrices)
    {
        std::cout << "E: \n" << essential_matrix << std::endl;
    }

    for (const auto rotation_matrix : rotation_matrices)
    {
        std::cout << "R: \n" << rotation_matrix << std::endl;
    }

    for (const auto translation_vector : translation_vectors)
    {
        std::cout << "t: \t" << translation_vector.transpose() << std::endl;
    }
    return 0;
}
