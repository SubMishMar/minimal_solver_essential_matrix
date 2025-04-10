#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <random>
#include <vector>
#include <stdexcept>
#include <Eigen/Core>
#include <Eigen/SVD>
#include <tuple>

using cv::Mat;
using cv::Mat_;
using cv::Point2d;
using cv::Point3d;
using std::cout;
using std::endl;
using std::vector;

// Generates random 3D points in front of the camera.
vector<Point3d> Generate3DPoints(int num_points) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-1.0, 1.0);

  vector<Point3d> points;
  points.reserve(num_points);

  for (int i = 0; i < num_points; ++i) {
    double x = dis(gen);
    double y = dis(gen);
    double z = dis(gen) + 3.0;  // ensure positive depth
    points.emplace_back(x, y, z);
  }
  return points;
}

// Projects 3D points into 2D using intrinsic and extrinsic parameters.
vector<Point2d> ProjectPoints(const vector<Point3d>& points, const Mat& k,
                              const Mat& r, const Mat& t) {
  vector<Point2d> image_points;
  image_points.reserve(points.size());

  for (const auto& point : points) {
    Mat x = (Mat_<double>(3, 1) << point.x, point.y, point.z);
    Mat x_cam = r * x + t;
    Mat x_img = k * x_cam;

    double u = x_img.at<double>(0, 0) / x_img.at<double>(2, 0);
    double v = x_img.at<double>(1, 0) / x_img.at<double>(2, 0);

    image_points.emplace_back(u, v);
  }

  return image_points;
}

Eigen::RowVectorXd EpipolarConstraintRow(const Eigen::Vector2d& x1,
    const Eigen::Vector2d& x2) {
    double u1 = x1(0);
    double v1 = x1(1);
    double u2 = x2(0);
    double v2 = x2(1);

    Eigen::RowVectorXd row(9);
    row << u2 * u1, u2 * v1, u2,
    v2 * u1, v2 * v1, v2,
    u1,      v1,      1.0;

    return row;
}

std::vector<Eigen::Matrix3d> ComputeNullspaceEssentialCandidates(const Eigen::Matrix<double, 5, 9>& A) {
    Eigen::JacobiSVD<Eigen::Matrix<double, 5, 9>> svd(A, Eigen::ComputeFullV);
    const auto& V = svd.matrixV();  // V is 9x9
  
    std::vector<Eigen::Matrix3d> essential_matrices;
    essential_matrices.reserve(4);
  
    // Last 4 columns of V form the 4D null space of A
    for (int i = 5; i < 9; ++i) {
      Eigen::VectorXd e = V.col(i);  // 9x1
      Eigen::Matrix3d E;
      E << e(0), e(1), e(2),
           e(3), e(4), e(5),
           e(6), e(7), e(8);
      essential_matrices.push_back(E);
    }
  
    return essential_matrices;
}
  
// Multiply two first-order polynomials with structure [ax, ay, az, aw]
// Returns a second-order polynomial with 10 terms
Eigen::VectorXd P1P1(const Eigen::Vector4d& p1, const Eigen::Vector4d& p2) {
    Eigen::VectorXd pout(10);
  
    pout(0) = p1(0) * p2(0);                          // x^2
    pout(1) = p1(1) * p2(1);                          // y^2
    pout(2) = p1(2) * p2(2);                          // z^2
    pout(3) = p1(0) * p2(1) + p1(1) * p2(0);          // xy
    pout(4) = p1(0) * p2(2) + p1(2) * p2(0);          // xz
    pout(5) = p1(1) * p2(2) + p1(2) * p2(1);          // yz
    pout(6) = p1(0) * p2(3) + p1(3) * p2(0);          // x
    pout(7) = p1(1) * p2(3) + p1(3) * p2(1);          // y
    pout(8) = p1(2) * p2(3) + p1(3) * p2(2);          // z
    pout(9) = p1(3) * p2(3);                          // constant term (1)
  
    return pout;
}

// Multiply a second-order polynomial (10 terms) with a first-order polynomial (4 terms)
// Returns a third-order polynomial with 20 terms
Eigen::VectorXd P2P1(const Eigen::VectorXd& p2, const Eigen::Vector4d& p1) {
    Eigen::VectorXd pout(20);
  
    // p2 is assumed to have structure: [x² y² z² xy xz yz x y z 1]
    pout(0) = p2(0) * p1(0);                                  // x^3
    pout(1) = p2(1) * p1(1);                                  // y^3
    pout(2) = p2(2) * p1(2);                                  // z^3
    pout(3) = p2(0) * p1(1) + p2(3) * p1(0);                  // x^2y
    pout(4) = p2(1) * p1(0) + p2(3) * p1(1);                  // xy^2
    pout(5) = p2(0) * p1(2) + p2(4) * p1(0);                  // x^2z
    pout(6) = p2(2) * p1(0) + p2(4) * p1(2);                  // xz^2
    pout(7) = p2(1) * p1(2) + p2(5) * p1(1);                  // y^2z
    pout(8) = p2(2) * p1(1) + p2(5) * p1(2);                  // yz^2
    pout(9) = p2(3) * p1(2) + p2(4) * p1(1) + p2(5) * p1(0);  // xyz
  
    pout(10) = p2(0) * p1(3) + p2(6) * p1(0);                 // x^2
    pout(11) = p2(1) * p1(3) + p2(7) * p1(1);                 // y^2
    pout(12) = p2(2) * p1(3) + p2(8) * p1(2);                 // z^2
    pout(13) = p2(3) * p1(3) + p2(6) * p1(1) + p2(7) * p1(0); // xy
    pout(14) = p2(4) * p1(3) + p2(6) * p1(2) + p2(8) * p1(0); // xz
    pout(15) = p2(5) * p1(3) + p2(7) * p1(2) + p2(8) * p1(1); // yz
  
    pout(16) = p2(6) * p1(3) + p2(9) * p1(0);                 // x
    pout(17) = p2(7) * p1(3) + p2(9) * p1(1);                 // y
    pout(18) = p2(8) * p1(3) + p2(9) * p1(2);                 // z
    pout(19) = p2(9) * p1(3);                                // 1
  
    return pout;
}

#include <Eigen/Dense>
#include <stdexcept>

// Multiply two 3rd-order z polynomials: p1 and p2
// p1, p2: [z³ z² z¹ 1] → size 4
// Result: [z⁶ ... 1]   → size 7
Eigen::RowVectorXd PZ3PZ3(const Eigen::RowVectorXd& p1, const Eigen::RowVectorXd& p2) {
    if (p1.size() != 4 || p2.size() != 4) {
        throw std::invalid_argument("Both p1 and p2 must be size 4 (z^3 to 1)");
    }

    Eigen::RowVectorXd po(7);
    po.setZero();

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
        int deg = i + j;
        if (deg <= 6) {
            po(6 - deg) += p1(i) * p2(j);  // store in decreasing degree order
        }
        }
    }

    return po;
}

// Multiply a 4th-order z polynomial by a 3rd-order z polynomial
// p1: [z⁴, z³, z², z, 1]  → size 5
// p2: [z³, z², z, 1]      → size 4
// po: [z⁷, z⁶, ..., z, 1] → size 8
Eigen::RowVectorXd PZ4PZ3(const Eigen::RowVectorXd& p1, const Eigen::RowVectorXd& p2) {
    if (p1.size() != 5 || p2.size() != 4) {
      throw std::invalid_argument("p1 must be size 5 (z^4 to 1), p2 must be size 4 (z^3 to 1)");
    }
  
    Eigen::RowVectorXd po(8);
    po.setZero();
  
    for (int i = 0; i < 5; ++i) {
      for (int j = 0; j < 4; ++j) {
        int deg = i + j;  // total degree of term
        po(7 - deg) += p1(i) * p2(j);  // store in decreasing degree order
      }
    }
  
    return po;
}

// Multiply a 6th-order z polynomial (p1) by a 4th-order z polynomial (p2)
// p1: z^6 to 1 → size 7
// p2: z^4 to 1 → size 5
// Result: z^10 to 1 → size 11
Eigen::RowVectorXd PZ6PZ4(const Eigen::RowVectorXd& p1, const Eigen::RowVectorXd& p2) {
    if (p1.size() != 7 || p2.size() != 5) {
      throw std::invalid_argument("p1 must be size 7 (z^6 to 1), p2 must be size 5 (z^4 to 1)");
    }
  
    Eigen::RowVectorXd po(11);
    po.setZero();
  
    // Perform the convolution manually according to the structure
    for (int i = 0; i < 7; ++i) {
      for (int j = 0; j < 5; ++j) {
        int deg = i + j;           // z^(6-i) * z^(4-j) = z^(10 - (i + j))
        po(10 - deg) += p1(i) * p2(j);
      }
    }
  
    return po;
}

// Multiply a 7th-order z polynomial (p1) by a 3rd-order z polynomial (p2)
// p1: [z⁷ z⁶ z⁵ z⁴ z³ z² z¹ 1] → size 8
// p2: [z³ z² z¹ 1]             → size 4
// Result: [z¹⁰ ... 1]          → size 11
Eigen::RowVectorXd PZ7PZ3(const Eigen::RowVectorXd& p1, const Eigen::RowVectorXd& p2) {
    if (p1.size() != 8 || p2.size() != 4) {
      throw std::invalid_argument("p1 must be size 8 (z^7 to 1), p2 must be size 4 (z^3 to 1)");
    }
  
    Eigen::RowVectorXd po(11);
    po.setZero();
  
    for (int i = 0; i < 8; ++i) {
      for (int j = 0; j < 4; ++j) {
        int deg = i + j;
        if (deg <= 10) {
          po(10 - deg) += p1(i) * p2(j);  // store in decreasing degree order
        }
      }
    }
  
    return po;
}

// Computes the Reduced Row Echelon Form (RREF) of a matrix using partial pivoting.
Eigen::MatrixXd RREF(Eigen::MatrixXd mat, double tol = 1e-12) {
    const int rows = mat.rows();
    const int cols = mat.cols();
    int lead = 0;
  
    for (int r = 0; r < rows; ++r) {
      if (lead >= cols)
        break;
  
      // Find row with max absolute value in column 'lead'
      int i_max = r;
      for (int i = r + 1; i < rows; ++i) {
        if (std::abs(mat(i, lead)) > std::abs(mat(i_max, lead))) {
          i_max = i;
        }
      }
  
      // If pivot is too small, move to next column
      if (std::abs(mat(i_max, lead)) < tol) {
        ++lead;
        --r;  // Retry same row with next column
        continue;
      }
  
      // Swap to bring pivot row to position r
      mat.row(r).swap(mat.row(i_max));
  
      // Normalize pivot row
      mat.row(r) /= mat(r, lead);
  
      // Eliminate all other entries in this column
      for (int i = 0; i < rows; ++i) {
        if (i != r) {
          double factor = mat(i, lead);
          mat.row(i) -= factor * mat.row(r);
        }
      }
  
      ++lead;
    }
    return mat;
}

// Computes po = p1 - z * p2, following a specific polynomial ordering
Eigen::RowVectorXd PartialSubtrc(const Eigen::RowVectorXd& p1, const Eigen::RowVectorXd& p2) {
    if (p1.size() != 10 || p2.size() != 10) {
      throw std::invalid_argument("Both p1 and p2 must be row vectors of size 11.");
    }
  
    Eigen::RowVectorXd po(13);
  
    po(0)  = -p2(0);                           // -xz^2 → xz^3
    po(1)  = p1(0) - p2(1);                    // xz - z*xz → xz^2
    po(2)  = p1(1) - p2(2);                    // x - z*x → xz
    po(3)  = p1(2);                            // x
    po(4)  = -p2(3);                           // -yz^2 → yz^3
    po(5)  = p1(3) - p2(4);                    // yz - z*yz → yz^2
    po(6)  = p1(4) - p2(5);                    // y - z*y → yz
    po(7)  = p1(5);                            // y
    po(8)  = -p2(6);                           // -z^3 → z^4
    po(9)  = p1(6) - p2(7);                    // z^2 - z*z^2 → z^3
    po(10) = p1(7) - p2(8);                    // z - z*z → z^2
    po(11) = p1(8) - p2(9);                    // 1*z - z → z
    po(12) = p1(9);                            // constant (1)
  
    return po;
}

// Compute roots of a 10th-degree polynomial using the companion matrix
std::vector<std::complex<double>> ComputeRootsFromPolynomial(const Eigen::RowVectorXd& coeffs) {
    if (coeffs.size() != 11) {
      throw std::invalid_argument("Polynomial must be degree 10 (11 coefficients from z^10 to 1)");
    }
  
    Eigen::RowVectorXd poly = coeffs;
    
    // Normalize the polynomial (make leading coefficient 1)
    if (std::abs(poly(0)) < 1e-12) {
      throw std::runtime_error("Leading coefficient is zero or nearly zero. Cannot normalize.");
    }
    poly /= poly(0);
  
    // Create companion matrix
    Eigen::MatrixXd companion = Eigen::MatrixXd::Zero(10, 10);
    companion.block(1, 0, 9, 9) = Eigen::MatrixXd::Identity(9, 9);  // identity below diagonal
    companion.col(9) = -poly.segment(1, 10).transpose();           // -coefficients from z^9 to 1
  
    // Compute eigenvalues (roots)
    Eigen::EigenSolver<Eigen::MatrixXd> solver(companion);
    Eigen::VectorXcd roots = solver.eigenvalues();
  
    // Convert to std::vector
    std::vector<std::complex<double>> result(roots.size());
    for (int i = 0; i < roots.size(); ++i) {
      result[i] = roots(i);
    }
  
    return result;
}

bool IsReal(const std::complex<double>& c, double tol = 1e-12) {
    return std::abs(c.imag()) < tol;
}

std::vector<Eigen::Matrix3d> EssentialMatricesFromComplexRoots(const std::vector<std::complex<double>>& roots,
                                                               const Eigen::RowVectorXd& p_1,
                                                               const Eigen::RowVectorXd& p_2,
                                                               const Eigen::RowVectorXd& p_3,
                                                               const Eigen::Matrix3d essential_x,
                                                               const Eigen::Matrix3d essential_y,
                                                               const Eigen::Matrix3d essential_z,
                                                               const Eigen::Matrix3d essential_w) {
    if (p_1.size() != 8) {
        throw std::invalid_argument("Polynomial p_1 must be degree 7");
    }
    if (p_2.size() != 8) {
        throw std::invalid_argument("Polynomial p_2 must be degree 7");
    }
    if (p_3.size() != 7) {
        throw std::invalid_argument("Polynomial p_2 must be degree 6");
    }
    std::vector<Eigen::Matrix3d> essential_matrices;
    for(const auto& root : roots) {
        if(!IsReal(root)) {
            continue;
        }
        const double z = root.real();
        double z2 = z*z;
        double z3 = z2*z;
        double z4 = z3*z;
        double z5 = z4*z;
        double z6 = z5*z;
        double z7 = z6 * z;
        Eigen::RowVectorXd p_z6(7);
        p_z6 << z6, z5, z4, z3, z2, z, 1.0;
        Eigen::RowVectorXd p_z7(8);
        p_z7(0) = z7;
        p_z7.tail(7) = p_z6;
        const double x = p_1.dot(p_z7) / p_3.dot(p_z6);
        const double y = p_2.dot(p_z7) / p_3.dot(p_z6);
        Eigen::Matrix3d essential_matrix = x*essential_x + y*essential_y + z*essential_z + essential_w;
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(essential_matrix, Eigen::ComputeFullU | Eigen::ComputeFullV);
        const Eigen::MatrixXd U = svd.matrixU();  // Left singular vectors
        const Eigen::MatrixXd V = svd.matrixV(); 
        const Eigen::Matrix3d D = Eigen::Vector3d(1.0, 1.0, 0.0).asDiagonal();
        essential_matrix = U*D*V.transpose();
        essential_matrices.emplace_back(essential_matrix);
    }

    return essential_matrices;
}

cv::Mat FindEssentialMatMinimalSolver(const std::vector<cv::Point2d>& points_1,
    const std::vector<cv::Point2d>& points_2) {
    if (points_1.size() != points_2.size()) {
    throw std::invalid_argument("Point vectors must be of the same size.");
    }

    const int num_points = static_cast<int>(points_1.size());
    if (num_points < 5) {
    throw std::invalid_argument("Need at least 5 point correspondences.");
    }

    // Shuffle indices for random sampling
    std::vector<int> indices(num_points);
    std::iota(indices.begin(), indices.end(), 0);

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    Eigen::Matrix<double, 5, 9> A;

    for (int i = 0; i < 5; ++i) {
        const auto& pt1 = Eigen::Vector2d{points_1[indices[i]].x, points_1[indices[i]].y};
        const auto& pt2 = Eigen::Vector2d{points_2[indices[i]].x, points_2[indices[i]].y};
        A.row(i) = EpipolarConstraintRow(pt1, pt2);
    }

    const auto essential_matrices_basis = ComputeNullspaceEssentialCandidates(A);
    const Eigen::Matrix3d essential_x = essential_matrices_basis.at(0);
    const Eigen::Matrix3d essential_y = essential_matrices_basis.at(1);
    const Eigen::Matrix3d essential_z = essential_matrices_basis.at(2);
    const Eigen::Matrix3d essential_w = essential_matrices_basis.at(3);
    const Eigen::Vector4d e_00{essential_x(0, 0), essential_y(0, 0), essential_z(0, 0), essential_w(0, 0)};
    const Eigen::Vector4d e_01{essential_x(0, 1), essential_y(0, 1), essential_z(0, 1), essential_w(0, 1)};
    const Eigen::Vector4d e_02{essential_x(0, 2), essential_y(0, 2), essential_z(0, 2), essential_w(0, 2)};

    const Eigen::Vector4d e_10{essential_x(1, 0), essential_y(1, 0), essential_z(1, 0), essential_w(1, 0)};
    const Eigen::Vector4d e_11{essential_x(1, 1), essential_y(1, 1), essential_z(1, 1), essential_w(1, 1)};
    const Eigen::Vector4d e_12{essential_x(1, 2), essential_y(1, 2), essential_z(1, 2), essential_w(1, 2)};

    const Eigen::Vector4d e_20{essential_x(2, 0), essential_y(2, 0), essential_z(2, 0), essential_w(2, 0)};
    const Eigen::Vector4d e_21{essential_x(2, 1), essential_y(2, 1), essential_z(2, 1), essential_w(2, 1)};
    const Eigen::Vector4d e_22{essential_x(2, 2), essential_y(2, 2), essential_z(2, 2), essential_w(2, 2)};

    const Eigen::VectorXd det_essential = P2P1(P1P1(e_11, e_22) - P1P1(e_21, e_12), e_00)
                              -P2P1(P1P1(e_10, e_22) - P1P1(e_20, e_21), e_01)
                              +P2P1(P1P1(e_10, e_21) - P1P1(e_11, e_20), e_02);
    const Eigen::VectorXd ee_t00 = P1P1(e_00, e_00) + P1P1(e_01, e_01) + P1P1(e_02, e_02);

    const Eigen::VectorXd ee_t01 = P1P1(e_00, e_10) + P1P1(e_01, e_11) + P1P1(e_02, e_12);
    const Eigen::VectorXd ee_t02 = P1P1(e_00, e_20) + P1P1(e_01, e_21) + P1P1(e_02, e_22);
    const Eigen::VectorXd ee_t11 = P1P1(e_10, e_10) + P1P1(e_11, e_11) + P1P1(e_12, e_12);
    const Eigen::VectorXd ee_t12 = P1P1(e_10, e_20) + P1P1(e_11, e_21) + P1P1(e_12, e_22);
    const Eigen::VectorXd ee_t22 = P1P1(e_20, e_20) + P1P1(e_21, e_21) + P1P1(e_22, e_22);

    const Eigen::VectorXd trace_eet = (ee_t00 + ee_t11 + ee_t22);    
    const Eigen::VectorXd a_00 = ee_t00 - 0.5*trace_eet;
    const Eigen::VectorXd a_01 = ee_t01;
    const Eigen::VectorXd a_02 = ee_t02;
    const Eigen::VectorXd a_10 = a_01;
    const Eigen::VectorXd a_11 = ee_t11 - 0.5*trace_eet;
    const Eigen::VectorXd a_12 = ee_t12;
    const Eigen::VectorXd a_20 = a_02;
    const Eigen::VectorXd a_21 = a_12;
    const Eigen::VectorXd a_22 = ee_t22 - 0.5*trace_eet;

    const Eigen::VectorXd ae_00 = P2P1(a_00, e_00) + P2P1(a_01, e_10) + P2P1(a_02, e_20);
    const Eigen::VectorXd ae_01 = P2P1(a_00, e_01) + P2P1(a_01, e_11) + P2P1(a_02, e_21);
    const Eigen::VectorXd ae_02 = P2P1(a_00, e_02) + P2P1(a_01, e_12) + P2P1(a_02, e_22);

    const Eigen::VectorXd ae_10 = P2P1(a_10, e_00) + P2P1(a_11, e_10) + P2P1(a_12, e_20);
    const Eigen::VectorXd ae_11 = P2P1(a_10, e_01) + P2P1(a_11, e_11) + P2P1(a_12, e_21);
    const Eigen::VectorXd ae_12 = P2P1(a_10, e_02) + P2P1(a_11, e_12) + P2P1(a_12, e_22);

    const Eigen::VectorXd ae_20 = P2P1(a_20, e_00) + P2P1(a_21, e_10) + P2P1(a_22, e_20);
    const Eigen::VectorXd ae_21 = P2P1(a_20, e_01) + P2P1(a_21, e_11) + P2P1(a_22, e_21);
    const Eigen::VectorXd ae_22 = P2P1(a_20, e_02) + P2P1(a_21, e_12) + P2P1(a_22, e_22);

    Eigen::MatrixXd a(10, 20);
    a.row(0) = det_essential;
    a.row(1) = ae_00;
    a.row(2) = ae_01;
    a.row(3) = ae_02;
    a.row(4) = ae_10;
    a.row(5) = ae_11;
    a.row(6) = ae_12;
    a.row(7) = ae_20;
    a.row(8) = ae_21;
    a.row(9) = ae_22;

    std::vector<int> col_order = {0, 1, 3, 4, 5, 10, 7, 11, 9, 13, 6, 14, 16, 8, 15, 17, 2, 12, 18, 19};

    Eigen::MatrixXd a_permuted(10, 20);
    for(size_t i = 0; i < 20; ++i) {
        a_permuted.col(i) = a.col(col_order[i]);
    }
    
    const Eigen::MatrixXd a_el = RREF(a_permuted);

    Eigen::RowVectorXd k_row = PartialSubtrc(a_el.row(4).segment(10, 10), a_el.row(5).segment(10, 10));
    Eigen::RowVectorXd l_row = PartialSubtrc(a_el.row(6).segment(10, 10), a_el.row(7).segment(10, 10));
    Eigen::RowVectorXd m_row = PartialSubtrc(a_el.row(8).segment(10, 10), a_el.row(9).segment(10, 10));

    Eigen::RowVectorXd b_11 = k_row.row(0).segment(0, 4);
    Eigen::RowVectorXd b_12 = k_row.row(0).segment(4, 4);
    Eigen::RowVectorXd b_13 = k_row.row(0).segment(8, 5);
    Eigen::RowVectorXd b_21 = l_row.row(0).segment(0, 4);
    Eigen::RowVectorXd b_22 = l_row.row(0).segment(4, 4);
    Eigen::RowVectorXd b_23 = l_row.row(0).segment(8, 5);
    Eigen::RowVectorXd b_31 = m_row.row(0).segment(0, 4);
    Eigen::RowVectorXd b_32 = m_row.row(0).segment(4, 4);
    Eigen::RowVectorXd b_33 = m_row.row(0).segment(8, 5);

    const Eigen::RowVectorXd p_1 = PZ4PZ3(b_23, b_12) - PZ4PZ3(b_13, b_22);
    const Eigen::RowVectorXd p_2 = PZ4PZ3(b_13, b_21) - PZ4PZ3(b_23, b_11);
    const Eigen::RowVectorXd p_3 = PZ3PZ3(b_11, b_22) - PZ3PZ3(b_12, b_21);

    const Eigen::RowVectorXd n_row = PZ7PZ3(p_1, b_31) + PZ7PZ3(p_2, b_32) + PZ6PZ4(p_3, b_33);

    Eigen::RowVectorXd n_row_scaled;

    if (std::abs(n_row(0)) > 1e-12) {
      n_row_scaled = n_row / n_row(0);  // normalize safely
    } else {
      std::cerr << "Warning: n_row(0) is too close to zero. Skipping normalization.\n";
      n_row_scaled = n_row;  // or set to zero, or handle differently
    }

    std::cout << "n_row: \n" << n_row_scaled << std::endl;
    std::vector<std::complex<double>> all_roots = ComputeRootsFromPolynomial(n_row_scaled);
    const std::vector<Eigen::Matrix3d> essential_matrices = EssentialMatricesFromComplexRoots(all_roots, p_1, p_2, p_3, essential_x, essential_y, essential_z, essential_w);
    for(const auto& essential_matrix : essential_matrices) {
        std::cout << "E: \n" << essential_matrix << std::endl; 
    }
    return cv::Mat();  
}

// Creates a skew-symmetric matrix from a translation vector.
Mat Skew(const Mat& t) {
  return (Mat_<double>(3, 3) << 
    0, -t.at<double>(2), t.at<double>(1),
    t.at<double>(2), 0, -t.at<double>(0),
    -t.at<double>(1), t.at<double>(0), 0);
}

std::vector<cv::Point2d> NormalizePoints(const std::vector<cv::Point2d>& image_points,
    const cv::Mat& k) {
    cv::Mat k_inv = k.inv();
    std::vector<cv::Point2d> normalized;

    normalized.reserve(image_points.size());

    for (const auto& pt : image_points) {
    cv::Mat pt_h = (cv::Mat_<double>(3, 1) << pt.x, pt.y, 1.0);
    cv::Mat norm_pt = k_inv * pt_h;
    normalized.emplace_back(norm_pt.at<double>(0), norm_pt.at<double>(1));
    }

    return normalized;
}


int main() {
  const Mat k = (Mat_<double>(3, 3) << 
    800, 0, 320,
    0, 800, 240,
    0, 0, 1);

  const double theta = 10 * CV_PI / 180.0;
  const Mat r = (Mat_<double>(3, 3) <<
    cos(theta), -sin(theta), 0,
    sin(theta),  cos(theta), 0,
    0, 0, 1);

  const Mat t = (Mat_<double>(3, 1) << 0.1, 0, 0);

  const auto points_3d = Generate3DPoints(10);

  const auto points_cam1 = ProjectPoints(points_3d, k, Mat::eye(3, 3, CV_64F),
                                         Mat::zeros(3, 1, CV_64F));
  const auto points_cam1_normalized = NormalizePoints(points_cam1, k);
  const auto points_cam2 = ProjectPoints(points_3d, k, r, t);
  const auto points_cam2_normalized = NormalizePoints(points_cam2, k);

  FindEssentialMatMinimalSolver(points_cam1_normalized, points_cam2_normalized);
  Mat e_est = findEssentialMat(points_cam1_normalized, points_cam2_normalized, 1.0, cv::Point2d(0,0), cv::RANSAC, 0.999, 1.0);

  Mat e_gt = Skew(t) * r;
  e_gt /= cv::norm(e_gt);
  e_est /= cv::norm(e_est);

  cout << "Estimated Essential Matrix:\n" << e_est << endl;
  cout << "Ground Truth Essential Matrix:\n" << e_gt << endl;

  double frobenius_error = cv::norm(e_est - e_gt);
  cout << "Frobenius Error: " << frobenius_error << endl;

  return 0;
}