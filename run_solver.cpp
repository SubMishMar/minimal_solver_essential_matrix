#include <opencv2/opencv.hpp>

#include "minimal_solver.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <armadillo>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <tuple>
#include <vector>

using cv::Mat;
using cv::Mat_;
using cv::Point2d;
using cv::Point3d;
using std::cout;
using std::endl;
using std::vector;

// Generates random 3D points in front of the camera.
vector<Point3d> Generate3DPoints(int num_points)
{
    std::random_device               rd;
    std::mt19937                     gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    vector<Point3d> points;
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
std::vector<Eigen::Vector2d> ProjectPoints(const vector<Point3d>& points, const Mat& k,
                                           const Mat& r, const Mat& t)
{
    std::vector<Eigen::Vector2d> image_points;
    image_points.reserve(points.size());

    for (const auto& point : points)
    {
        Mat x     = (Mat_<double>(3, 1) << point.x, point.y, point.z);
        Mat x_cam = r * x + t;
        Mat x_img = k * x_cam;

        double u = x_img.at<double>(0, 0) / x_img.at<double>(2, 0);
        double v = x_img.at<double>(1, 0) / x_img.at<double>(2, 0);

        image_points.emplace_back(Eigen::Vector2d(u, v));
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

Eigen::Matrix3d cvToEigen3x3(const cv::Mat& mat)
{
    assert(mat.rows == 3 && mat.cols == 3 && mat.type() == CV_64F);
    Eigen::Matrix3d m;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) m(i, j) = mat.at<double>(i, j);
    return m;
}

int main()
{
    const Mat k = (Mat_<double>(3, 3) << 800, 0, 320, 0, 800, 240, 0, 0, 1);

    const Eigen::Matrix3d k_eig = cvToEigen3x3(k);

    const double theta = 10 * CV_PI / 180.0;
    const Mat    r =
        (Mat_<double>(3, 3) << cos(theta), -sin(theta), 0, sin(theta), cos(theta), 0, 0, 0, 1);
    const Mat  t         = (Mat_<double>(3, 1) << 0.1, 0, 0);
    const auto points_3d = Generate3DPoints(5);
    const auto points_cam1 =
        ProjectPoints(points_3d, k, Mat::eye(3, 3, CV_64F), Mat::zeros(3, 1, CV_64F));
    const auto points_cam1_normalized = NormalizePoints(points_cam1, k_eig);
    const auto points_cam2            = ProjectPoints(points_3d, k, r, t);
    const auto points_cam2_normalized = NormalizePoints(points_cam2, k_eig);

    minimal_solver::FindEssentialMatMinimalSolver(points_cam1_normalized, points_cam2_normalized,
                                                  k_eig);

    return 0;
}