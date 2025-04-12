#pragma once

#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <random>
#include <vector>
#include <stdexcept>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <tuple>
#include <limits>
#include <armadillo>

using cv::Mat;
using cv::Mat_;
using cv::Point2d;
using cv::Point3d;
using std::cout;
using std::endl;
using std::vector;

namespace minimal_solver {
    cv::Mat FindEssentialMatMinimalSolver(const std::vector<Eigen::Vector2d>& points_1,
        const std::vector<Eigen::Vector2d>& points_2,
        const Eigen::Matrix3d& k);
}