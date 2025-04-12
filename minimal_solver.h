#pragma once

#include <Eigen/Core>

// Implementation inspired from - https://github.com/SergioRAgostinho/five_point_algorithm
namespace minimal_solver
{
std::tuple<std::vector<Eigen::Matrix3d>, std::vector<Eigen::Matrix3d>, std::vector<Eigen::Vector3d>>
FindEssentialMatMinimalSolver(const std::vector<Eigen::Vector2d>& points_1,
                              const std::vector<Eigen::Vector2d>& points_2,
                              const Eigen::Matrix3d&              k);
}