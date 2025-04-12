#include "minimal_solver.h"

#include "utility.h"

#include <iostream>
namespace minimal_solver
{
std::tuple<std::vector<Eigen::Matrix3d>, std::vector<Eigen::Matrix3d>, std::vector<Eigen::Vector3d>>
FindEssentialMatMinimalSolver(const std::vector<Eigen::Vector2d>& points_1,
                              const std::vector<Eigen::Vector2d>& points_2,
                              const Eigen::Matrix3d&              k)
{
    if (points_1.size() != points_2.size())
    {
        throw std::invalid_argument("Point vectors must be of the same size.");
    }

    const int num_points = static_cast<int>(points_1.size());
    if (num_points < 5)
    {
        throw std::invalid_argument("Need at least 5 point correspondences.");
    }

    const Eigen::Matrix3d k_inv           = k.inverse();
    const Eigen::Matrix3d k_inv_transpose = k_inv.transpose();

    Eigen::Matrix<double, 5, 9> A;

    for (int i = 0; i < 5; ++i)
    {
        const auto& pt1 = points_1[i];
        const auto& pt2 = points_2[i];
        A.row(i)        = EpipolarConstraintRow(pt1, pt2);
    }

    const auto            essential_matrices_basis = ComputeNullspaceEssentialCandidates(A);
    const Eigen::Matrix3d essential_x              = essential_matrices_basis.at(0);
    const Eigen::Matrix3d essential_y              = essential_matrices_basis.at(1);
    const Eigen::Matrix3d essential_z              = essential_matrices_basis.at(2);
    const Eigen::Matrix3d essential_w              = essential_matrices_basis.at(3);
    const Eigen::Vector4d e_00{essential_x(0, 0), essential_y(0, 0), essential_z(0, 0),
                               essential_w(0, 0)};
    const Eigen::Vector4d e_01{essential_x(0, 1), essential_y(0, 1), essential_z(0, 1),
                               essential_w(0, 1)};
    const Eigen::Vector4d e_02{essential_x(0, 2), essential_y(0, 2), essential_z(0, 2),
                               essential_w(0, 2)};
    const Eigen::Vector4d e_10{essential_x(1, 0), essential_y(1, 0), essential_z(1, 0),
                               essential_w(1, 0)};
    const Eigen::Vector4d e_11{essential_x(1, 1), essential_y(1, 1), essential_z(1, 1),
                               essential_w(1, 1)};
    const Eigen::Vector4d e_12{essential_x(1, 2), essential_y(1, 2), essential_z(1, 2),
                               essential_w(1, 2)};
    const Eigen::Vector4d e_20{essential_x(2, 0), essential_y(2, 0), essential_z(2, 0),
                               essential_w(2, 0)};
    const Eigen::Vector4d e_21{essential_x(2, 1), essential_y(2, 1), essential_z(2, 1),
                               essential_w(2, 1)};
    const Eigen::Vector4d e_22{essential_x(2, 2), essential_y(2, 2), essential_z(2, 2),
                               essential_w(2, 2)};

    const Eigen::Matrix3d essential_x_k = k_inv_transpose * essential_matrices_basis.at(0) * k_inv;
    const Eigen::Matrix3d essential_y_k = k_inv_transpose * essential_matrices_basis.at(1) * k_inv;
    const Eigen::Matrix3d essential_z_k = k_inv_transpose * essential_matrices_basis.at(2) * k_inv;
    const Eigen::Matrix3d essential_w_k = k_inv_transpose * essential_matrices_basis.at(3) * k_inv;
    const Eigen::Vector4d e_00_k{essential_x_k(0, 0), essential_y_k(0, 0), essential_z_k(0, 0),
                                 essential_w_k(0, 0)};
    const Eigen::Vector4d e_01_k{essential_x_k(0, 1), essential_y_k(0, 1), essential_z_k(0, 1),
                                 essential_w_k(0, 1)};
    const Eigen::Vector4d e_02_k{essential_x_k(0, 2), essential_y_k(0, 2), essential_z_k(0, 2),
                                 essential_w_k(0, 2)};
    const Eigen::Vector4d e_10_k{essential_x_k(1, 0), essential_y_k(1, 0), essential_z_k(1, 0),
                                 essential_w_k(1, 0)};
    const Eigen::Vector4d e_11_k{essential_x_k(1, 1), essential_y_k(1, 1), essential_z_k(1, 1),
                                 essential_w_k(1, 1)};
    const Eigen::Vector4d e_12_k{essential_x_k(1, 2), essential_y_k(1, 2), essential_z_k(1, 2),
                                 essential_w_k(1, 2)};
    const Eigen::Vector4d e_20_k{essential_x_k(2, 0), essential_y_k(2, 0), essential_z_k(2, 0),
                                 essential_w_k(2, 0)};
    const Eigen::Vector4d e_21_k{essential_x_k(2, 1), essential_y_k(2, 1), essential_z_k(2, 1),
                                 essential_w_k(2, 1)};
    const Eigen::Vector4d e_22_k{essential_x_k(2, 2), essential_y_k(2, 2), essential_z_k(2, 2),
                                 essential_w_k(2, 2)};

    const Eigen::VectorXd det_essential_k =
        P2P1(P1P1(e_01_k, e_12_k) - P1P1(e_02_k, e_11_k), e_20_k) +
        P2P1(P1P1(e_02_k, e_10_k) - P1P1(e_00_k, e_12_k), e_21_k) +
        P2P1(P1P1(e_00_k, e_11_k) - P1P1(e_01_k, e_10_k), e_22_k);

    const Eigen::VectorXd ee_t00 = P1P1(e_00, e_00) + P1P1(e_01, e_01) + P1P1(e_02, e_02);
    const Eigen::VectorXd ee_t01 = P1P1(e_00, e_10) + P1P1(e_01, e_11) + P1P1(e_02, e_12);
    const Eigen::VectorXd ee_t02 = P1P1(e_00, e_20) + P1P1(e_01, e_21) + P1P1(e_02, e_22);
    const Eigen::VectorXd ee_t11 = P1P1(e_10, e_10) + P1P1(e_11, e_11) + P1P1(e_12, e_12);
    const Eigen::VectorXd ee_t12 = P1P1(e_10, e_20) + P1P1(e_11, e_21) + P1P1(e_12, e_22);
    const Eigen::VectorXd ee_t22 = P1P1(e_20, e_20) + P1P1(e_21, e_21) + P1P1(e_22, e_22);

    const Eigen::VectorXd trace_eet = (ee_t00 + ee_t11 + ee_t22);
    const Eigen::VectorXd a_00      = ee_t00 - 0.5 * trace_eet;
    const Eigen::VectorXd a_01      = ee_t01;
    const Eigen::VectorXd a_02      = ee_t02;
    const Eigen::VectorXd a_10      = a_01;
    const Eigen::VectorXd a_11      = ee_t11 - 0.5 * trace_eet;
    const Eigen::VectorXd a_12      = ee_t12;
    const Eigen::VectorXd a_20      = a_02;
    const Eigen::VectorXd a_21      = a_12;
    const Eigen::VectorXd a_22      = ee_t22 - 0.5 * trace_eet;

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
    a.row(0) = det_essential_k;
    a.row(1) = ae_00;
    a.row(2) = ae_01;
    a.row(3) = ae_02;
    a.row(4) = ae_10;
    a.row(5) = ae_11;
    a.row(6) = ae_12;
    a.row(7) = ae_20;
    a.row(8) = ae_21;
    a.row(9) = ae_22;

    std::vector<int> col_order = {0, 1,  3,  4, 5,  10, 7, 11, 9,  13,
                                  6, 14, 16, 8, 15, 17, 2, 12, 18, 19};

    Eigen::MatrixXd a_permuted(10, 20);
    for (size_t i = 0; i < 20; ++i)
    {
        a_permuted.col(i) = a.col(col_order[i]);
    }

    const Eigen::MatrixXd a_el = GaussJordanEliminationWithPartialPivoting(a_permuted);

    Eigen::RowVectorXd k_row =
        PartialSubtrc(a_el.row(4).segment(10, 10), a_el.row(5).segment(10, 10));
    Eigen::RowVectorXd l_row =
        PartialSubtrc(a_el.row(6).segment(10, 10), a_el.row(7).segment(10, 10));
    Eigen::RowVectorXd m_row =
        PartialSubtrc(a_el.row(8).segment(10, 10), a_el.row(9).segment(10, 10));

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

    if (std::abs(n_row(0)) > 1e-12)
    {
        n_row_scaled = n_row / n_row(0); // normalize safely
    }
    else
    {
        std::cerr << "Warning: n_row(0) is too close to zero. Skipping normalization.\n";
        n_row_scaled = n_row; // or set to zero, or handle differently
    }
    std::vector<std::complex<double>> all_roots = ComputeRootsFromPolynomial(n_row_scaled);
    Eigen::Vector3d                   q_1;
    q_1 << points_1[0].x(), points_1[0].y(), 1.0;
    Eigen::Vector3d q_2;
    q_2 << points_2[0].x(), points_2[0].y(), 1.0;
    return EstimateMotionFromComplexRoots(all_roots, p_1, p_2, p_3, essential_x, essential_y,
                                          essential_z, essential_w, q_1, q_2);
}
} // namespace minimal_solver