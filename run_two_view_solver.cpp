#include "utility.h"

#include <iostream>
#include <random>
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

    constexpr int num_points{100};
    const auto    points_3d_all_1 = Generate3DPoints(num_points);
    const auto    points_3d_all_2 = TransformPointsFromWorldToCamera(points_3d_all_1, r, t);

    const auto points_cam1_all = ProjectPoints(points_3d_all_1, k);
    const auto points_cam2_all = ProjectPoints(points_3d_all_2, k);

    const auto points_cam1_all_normalized = NormalizePoints(points_cam1_all, k);
    const auto points_cam2_all_normalized = NormalizePoints(points_cam2_all, k);

    // Randomly pick 2 indices
    std::vector<int> indices(num_points);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), std::mt19937{std::random_device{}()});
    std::vector<int> sample_indices(indices.begin(), indices.begin() + 2);
}