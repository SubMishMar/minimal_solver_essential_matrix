#pragma once

#include <opencv2/opencv.hpp>

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

Eigen::RowVectorXd EpipolarConstraintRow(const Eigen::Vector2d& x1, const Eigen::Vector2d& x2);

std::vector<Eigen::Matrix3d> ComputeNullspaceEssentialCandidates(
    const Eigen::Matrix<double, 5, 9>& A_eigen);

// Multiply two first-order polynomials with structure [ax, ay, az, aw]
// Returns a second-order polynomial with 10 terms
Eigen::VectorXd P1P1(const Eigen::Vector4d& p1, const Eigen::Vector4d& p2);

// Multiply a second-order polynomial (10 terms) with a first-order polynomial (4 terms)
// Returns a third-order polynomial with 20 terms
Eigen::VectorXd P2P1(const Eigen::VectorXd& p2, const Eigen::Vector4d& p1);

// Multiply two 3rd-order z polynomials
// p1, p2: length-4 vectors representing z³ → z⁰
// Returns: length-7 vector representing z⁶ → z⁰
Eigen::RowVectorXd PZ3PZ3(const Eigen::RowVectorXd& p1, const Eigen::RowVectorXd& p2);

// Multiply a 4th-order z polynomial by a 3rd-order z polynomial
// p1: coefficients of z⁴ → z⁰ (length 5)
// p2: coefficients of z³ → z⁰ (length 4)
// Returns: coefficients of z⁷ → z⁰ (length 8)
Eigen::RowVectorXd PZ4PZ3(const Eigen::RowVectorXd& p1, const Eigen::RowVectorXd& p2);

// Multiply a 6th-order polynomial with a 4th-order polynomial in z
// p1: [z^6, ..., z^0] (length 7)
// p2: [z^4, ..., z^0] (length 5)
// returns: [z^10, ..., z^0] (length 11)
Eigen::RowVectorXd PZ6PZ4(const Eigen::RowVectorXd& p1, const Eigen::RowVectorXd& p2);

// Multiply a 7th-order polynomial with a 3rd-order polynomial in z
// p1: [z^7, ..., z^0] (length 8)
// p2: [z^3, ..., z^0] (length 4)
// returns: [z^10, ..., z^0] (length 11)
Eigen::RowVectorXd PZ7PZ3(const Eigen::RowVectorXd& p1, const Eigen::RowVectorXd& p2);

// Hybrid: Uses Armadillo internally but exposes Eigen interface
Eigen::MatrixXd GaussJordanEliminationWithPartialPivoting(const Eigen::MatrixXd& A_eigen);

// Computes po = p1 - z * p2, following a specific polynomial ordering
Eigen::RowVectorXd PartialSubtrc(const Eigen::RowVectorXd& p1, const Eigen::RowVectorXd& p2);

// Compute roots of a 10th-degree polynomial using the companion matrix
std::vector<std::complex<double>> ComputeRootsFromPolynomial(const Eigen::RowVectorXd& coeffs);

std::tuple<std::vector<Eigen::Matrix3d>, std::vector<Eigen::Matrix3d>, std::vector<Eigen::Vector3d>>
EssentialMatricesFromComplexRoots(const std::vector<std::complex<double>>& roots,
                                  const Eigen::RowVectorXd& p_1, const Eigen::RowVectorXd& p_2,
                                  const Eigen::RowVectorXd& p_3, const Eigen::Matrix3d& essential_x,
                                  const Eigen::Matrix3d& essential_y,
                                  const Eigen::Matrix3d& essential_z,
                                  const Eigen::Matrix3d& essential_w, const Eigen::Vector3d& q_1,
                                  const Eigen::Vector3d& q_2);
