#include "utility.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <armadillo>

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

std::vector<Eigen::Vector2d> ProjectPoints(const std::vector<Eigen::Vector3d>& points,
                                           const Eigen::Matrix3d&              k)
{
    std::vector<Eigen::Vector2d> image_points;
    image_points.reserve(points.size());

    for (const auto& point : points)
    {
        Eigen::Vector3d x_img = k * point;

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

double deg2rad(double degrees)
{
    return degrees * M_PI / 180.0;
}

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

Eigen::Vector3d TransformPointFromWorldToCamera(const Eigen::Vector3d& world_point,
                                                const Eigen::Matrix3d& R, const Eigen::Vector3d& t)
{
    return R * world_point + t;
}

std::vector<Eigen::Vector3d> TransformPointsFromWorldToCamera(
    const std::vector<Eigen::Vector3d>& world_points, const Eigen::Matrix3d& R,
    const Eigen::Vector3d& t)
{
    std::vector<Eigen::Vector3d> camera_points;
    camera_points.reserve(world_points.size());
    for (const auto& world_point : world_points)
    {
        camera_points.emplace_back(TransformPointFromWorldToCamera(world_point, R, t));
    }
    return camera_points;
}

Eigen::RowVectorXd EpipolarConstraintRow(const Eigen::Vector2d& x1, const Eigen::Vector2d& x2)
{
    double u1 = x1(0);
    double v1 = x1(1);
    double u2 = x2(0);
    double v2 = x2(1);

    Eigen::RowVectorXd row(9);
    row << u2 * u1, u2 * v1, u2, v2 * u1, v2 * v1, v2, u1, v1, 1.0;

    return row;
}

std::vector<Eigen::Matrix3d> ComputeNullspaceEssentialCandidates(
    const Eigen::Matrix<double, 5, 9>& A_eigen)
{
    // Convert Eigen input to Armadillo
    Eigen::Matrix<double, 5, 9> A_eigen_nonconst = A_eigen;
    arma::mat                   A_arma(A_eigen_nonconst.data(), 5, 9, false, true);

    // Compute null space (each column is a basis vector)
    arma::mat N = arma::null(A_arma); // N is 9 x 4 if nullity is 4

    std::vector<Eigen::Matrix3d> essential_matrices;
    essential_matrices.reserve(N.n_cols); // adapt to actual number of null vectors

    for (arma::uword i = 0; i < N.n_cols; ++i)
    {
        // Extract the i-th 9x1 null vector from Armadillo
        arma::vec e_arma = N.col(i);

        // Copy into Eigen vector
        Eigen::VectorXd e = Eigen::Map<Eigen::VectorXd>(e_arma.memptr(), 9);

        // Reshape into 3x3 matrix
        Eigen::Matrix3d E;
        E << e(0), e(1), e(2), e(3), e(4), e(5), e(6), e(7), e(8);

        essential_matrices.push_back(E);
    }

    return essential_matrices;
}

Eigen::VectorXd P1P1(const Eigen::Vector4d& p1, const Eigen::Vector4d& p2)
{
    Eigen::VectorXd pout(10);

    pout(0) = p1(0) * p2(0);                 // x^2
    pout(1) = p1(1) * p2(1);                 // y^2
    pout(2) = p1(2) * p2(2);                 // z^2
    pout(3) = p1(0) * p2(1) + p1(1) * p2(0); // xy
    pout(4) = p1(0) * p2(2) + p1(2) * p2(0); // xz
    pout(5) = p1(1) * p2(2) + p1(2) * p2(1); // yz
    pout(6) = p1(0) * p2(3) + p1(3) * p2(0); // x
    pout(7) = p1(1) * p2(3) + p1(3) * p2(1); // y
    pout(8) = p1(2) * p2(3) + p1(3) * p2(2); // z
    pout(9) = p1(3) * p2(3);                 // constant term (1)

    return pout;
}

Eigen::VectorXd P2P1(const Eigen::VectorXd& p2, const Eigen::Vector4d& p1)
{
    Eigen::VectorXd pout(20);

    // p2 is assumed to have structure: [x² y² z² xy xz yz x y z 1]
    pout(0) = p2(0) * p1(0);                                 // x^3
    pout(1) = p2(1) * p1(1);                                 // y^3
    pout(2) = p2(2) * p1(2);                                 // z^3
    pout(3) = p2(0) * p1(1) + p2(3) * p1(0);                 // x^2y
    pout(4) = p2(1) * p1(0) + p2(3) * p1(1);                 // xy^2
    pout(5) = p2(0) * p1(2) + p2(4) * p1(0);                 // x^2z
    pout(6) = p2(2) * p1(0) + p2(4) * p1(2);                 // xz^2
    pout(7) = p2(1) * p1(2) + p2(5) * p1(1);                 // y^2z
    pout(8) = p2(2) * p1(1) + p2(5) * p1(2);                 // yz^2
    pout(9) = p2(3) * p1(2) + p2(4) * p1(1) + p2(5) * p1(0); // xyz

    pout(10) = p2(0) * p1(3) + p2(6) * p1(0);                 // x^2
    pout(11) = p2(1) * p1(3) + p2(7) * p1(1);                 // y^2
    pout(12) = p2(2) * p1(3) + p2(8) * p1(2);                 // z^2
    pout(13) = p2(3) * p1(3) + p2(6) * p1(1) + p2(7) * p1(0); // xy
    pout(14) = p2(4) * p1(3) + p2(6) * p1(2) + p2(8) * p1(0); // xz
    pout(15) = p2(5) * p1(3) + p2(7) * p1(2) + p2(8) * p1(1); // yz

    pout(16) = p2(6) * p1(3) + p2(9) * p1(0); // x
    pout(17) = p2(7) * p1(3) + p2(9) * p1(1); // y
    pout(18) = p2(8) * p1(3) + p2(9) * p1(2); // z
    pout(19) = p2(9) * p1(3);                 // 1

    return pout;
}

Eigen::RowVectorXd PZ3PZ3(const Eigen::RowVectorXd& p1, const Eigen::RowVectorXd& p2)
{
    if (p1.size() != 4 || p2.size() != 4)
        throw std::invalid_argument("Both polynomials must be of length 4 (z³ → z⁰)");

    Eigen::RowVectorXd po(7);

    po(0) = p1(0) * p2(0);                                                 // z^6
    po(1) = p1(0) * p2(1) + p1(1) * p2(0);                                 // z^5
    po(2) = p1(0) * p2(2) + p1(1) * p2(1) + p1(2) * p2(0);                 // z^4
    po(3) = p1(0) * p2(3) + p1(1) * p2(2) + p1(2) * p2(1) + p1(3) * p2(0); // z^3
    po(4) = p1(1) * p2(3) + p1(2) * p2(2) + p1(3) * p2(1);                 // z^2
    po(5) = p1(2) * p2(3) + p1(3) * p2(2);                                 // z^1
    po(6) = p1(3) * p2(3);                                                 // z^0

    return po;
}

Eigen::RowVectorXd PZ4PZ3(const Eigen::RowVectorXd& p1, const Eigen::RowVectorXd& p2)
{
    if (p1.size() != 5 || p2.size() != 4)
        throw std::invalid_argument("p1 must be length 5 and p2 must be length 4.");

    Eigen::RowVectorXd po(8);

    po(0) = p1(0) * p2(0);                                                 // z^7
    po(1) = p1(1) * p2(0) + p1(0) * p2(1);                                 // z^6
    po(2) = p1(2) * p2(0) + p1(1) * p2(1) + p1(0) * p2(2);                 // z^5
    po(3) = p1(3) * p2(0) + p1(2) * p2(1) + p1(1) * p2(2) + p1(0) * p2(3); // z^4
    po(4) = p1(4) * p2(0) + p1(3) * p2(1) + p1(2) * p2(2) + p1(1) * p2(3); // z^3
    po(5) = p1(4) * p2(1) + p1(3) * p2(2) + p1(2) * p2(3);                 // z^2
    po(6) = p1(4) * p2(2) + p1(3) * p2(3);                                 // z^1
    po(7) = p1(4) * p2(3);                                                 // z^0

    return po;
}

Eigen::RowVectorXd PZ6PZ4(const Eigen::RowVectorXd& p1, const Eigen::RowVectorXd& p2)
{
    if (p1.size() != 7 || p2.size() != 5)
        throw std::invalid_argument("p1 must be length 7 and p2 must be length 5");

    Eigen::RowVectorXd po(11);

    po(0)  = p1(0) * p2(0);                                                                 // z^10
    po(1)  = p1(1) * p2(0) + p1(0) * p2(1);                                                 // z^9
    po(2)  = p1(2) * p2(0) + p1(1) * p2(1) + p1(0) * p2(2);                                 // z^8
    po(3)  = p1(3) * p2(0) + p1(2) * p2(1) + p1(1) * p2(2) + p1(0) * p2(3);                 // z^7
    po(4)  = p1(4) * p2(0) + p1(3) * p2(1) + p1(2) * p2(2) + p1(1) * p2(3) + p1(0) * p2(4); // z^6
    po(5)  = p1(5) * p2(0) + p1(4) * p2(1) + p1(3) * p2(2) + p1(2) * p2(3) + p1(1) * p2(4); // z^5
    po(6)  = p1(6) * p2(0) + p1(5) * p2(1) + p1(4) * p2(2) + p1(3) * p2(3) + p1(2) * p2(4); // z^4
    po(7)  = p1(6) * p2(1) + p1(5) * p2(2) + p1(4) * p2(3) + p1(3) * p2(4);                 // z^3
    po(8)  = p1(6) * p2(2) + p1(5) * p2(3) + p1(4) * p2(4);                                 // z^2
    po(9)  = p1(6) * p2(3) + p1(5) * p2(4);                                                 // z^1
    po(10) = p1(6) * p2(4);                                                                 // z^0

    return po;
}

Eigen::RowVectorXd PZ7PZ3(const Eigen::RowVectorXd& p1, const Eigen::RowVectorXd& p2)
{
    if (p1.size() != 8 || p2.size() != 4)
        throw std::invalid_argument("p1 must be of length 8 and p2 of length 4");

    Eigen::RowVectorXd po(11);

    po(0)  = p1(0) * p2(0);                                                 // z^10
    po(1)  = p1(1) * p2(0) + p1(0) * p2(1);                                 // z^9
    po(2)  = p1(2) * p2(0) + p1(1) * p2(1) + p1(0) * p2(2);                 // z^8
    po(3)  = p1(3) * p2(0) + p1(2) * p2(1) + p1(1) * p2(2) + p1(0) * p2(3); // z^7
    po(4)  = p1(4) * p2(0) + p1(3) * p2(1) + p1(2) * p2(2) + p1(1) * p2(3); // z^6
    po(5)  = p1(5) * p2(0) + p1(4) * p2(1) + p1(3) * p2(2) + p1(2) * p2(3); // z^5
    po(6)  = p1(6) * p2(0) + p1(5) * p2(1) + p1(4) * p2(2) + p1(3) * p2(3); // z^4
    po(7)  = p1(7) * p2(0) + p1(6) * p2(1) + p1(5) * p2(2) + p1(4) * p2(3); // z^3
    po(8)  = p1(7) * p2(1) + p1(6) * p2(2) + p1(5) * p2(3);                 // z^2
    po(9)  = p1(7) * p2(2) + p1(6) * p2(3);                                 // z^1
    po(10) = p1(7) * p2(3);                                                 // z^0

    return po;
}

Eigen::MatrixXd GaussJordanEliminationWithPartialPivoting(const Eigen::MatrixXd& A_eigen)
{
    if (A_eigen.rows() != 10 || A_eigen.cols() != 20)
    {
        throw std::runtime_error("Matrix must be 10x20");
    }
    Eigen::MatrixXd A_eigen_nonconst = A_eigen;
    // Convert Eigen to Armadillo
    arma::mat A(A_eigen_nonconst.data(), A_eigen_nonconst.rows(), A_eigen_nonconst.cols(), false,
                true);

    arma::mat L, U, P;
    arma::lu(L, U, P, A); // MATLAB-style LU: P*A = L*U
    // U.print("U:");
    arma::mat B(10, 20, arma::fill::zeros);
    B.rows(0, 3) = U.rows(0, 3); // First 4 rows directly from U

    // Manual back-substitution
    B.row(9) = U.row(9) / U(9, 9);
    B.row(8) = (U.row(8) - U(8, 9) * B.row(9)) / U(8, 8);
    B.row(7) = (U.row(7) - U(7, 8) * B.row(8) - U(7, 9) * B.row(9)) / U(7, 7);
    B.row(6) = (U.row(6) - U(6, 7) * B.row(7) - U(6, 8) * B.row(8) - U(6, 9) * B.row(9)) / U(6, 6);
    B.row(5) = (U.row(5) - U(5, 6) * B.row(6) - U(5, 7) * B.row(7) - U(5, 8) * B.row(8) -
                U(5, 9) * B.row(9)) /
        U(5, 5);
    B.row(4) = (U.row(4) - U(4, 5) * B.row(5) - U(4, 6) * B.row(6) - U(4, 7) * B.row(7) -
                U(4, 8) * B.row(8) - U(4, 9) * B.row(9)) /
        U(4, 4);

    // Convert back to Eigen
    Eigen::MatrixXd B_eigen(10, 20);
    for (int i = 0; i < 10; ++i)
        for (int j = 0; j < 20; ++j) B_eigen(i, j) = B(i, j);

    return B_eigen;
}

Eigen::RowVectorXd PartialSubtrc(const Eigen::RowVectorXd& p1, const Eigen::RowVectorXd& p2)
{
    if (p1.size() != 10 || p2.size() != 10)
    {
        throw std::invalid_argument("Both p1 and p2 must be row vectors of size 11.");
    }

    Eigen::RowVectorXd po(13);

    po(0)  = -p2(0);        // -xz^2 → xz^3
    po(1)  = p1(0) - p2(1); // xz - z*xz → xz^2
    po(2)  = p1(1) - p2(2); // x - z*x → xz
    po(3)  = p1(2);         // x
    po(4)  = -p2(3);        // -yz^2 → yz^3
    po(5)  = p1(3) - p2(4); // yz - z*yz → yz^2
    po(6)  = p1(4) - p2(5); // y - z*y → yz
    po(7)  = p1(5);         // y
    po(8)  = -p2(6);        // -z^3 → z^4
    po(9)  = p1(6) - p2(7); // z^2 - z*z^2 → z^3
    po(10) = p1(7) - p2(8); // z - z*z → z^2
    po(11) = p1(8) - p2(9); // 1*z - z → z
    po(12) = p1(9);         // constant (1)

    return po;
}

std::vector<std::complex<double>> ComputeRootsFromPolynomial(const Eigen::RowVectorXd& coeffs)
{
    if (coeffs.size() != 11)
    {
        throw std::invalid_argument(
            "Polynomial must be degree 10 (11 coefficients from z^10 to 1)");
    }

    Eigen::RowVectorXd poly = coeffs;

    // Normalize the polynomial (make leading coefficient 1)
    if (std::abs(poly(0)) < 1e-12)
    {
        throw std::runtime_error("Leading coefficient is zero or nearly zero. Cannot normalize.");
    }
    poly /= poly(0);

    // Create companion matrix
    Eigen::MatrixXd companion   = Eigen::MatrixXd::Zero(10, 10);
    companion.block(1, 0, 9, 9) = Eigen::MatrixXd::Identity(9, 9); // identity below diagonal
    companion.row(0)            = -poly.segment(1, 10);            // -coefficients from z^9 to 1

    // Compute eigenvalues (roots)
    Eigen::EigenSolver<Eigen::MatrixXd> solver(companion);
    Eigen::VectorXcd                    roots = solver.eigenvalues();

    // Convert to std::vector
    std::vector<std::complex<double>> result(roots.size());
    for (int i = 0; i < roots.size(); ++i)
    {
        result[i] = roots(i);
    }

    return result;
}

bool IsReal(const std::complex<double>& c, double tol = 1e-12)
{
    return std::abs(c.imag()) < tol;
}

double ComputeEpipolarConstraint(const Eigen::Vector2d& point_1, const Eigen::Vector2d& point_2,
                                 const Eigen::Matrix3d& essential_matrix)
{
    return Eigen::Vector3d(point_2.x(), point_2.y(), 1.0).transpose() * essential_matrix *
        Eigen::Vector3d(point_1.x(), point_1.y(), 1.0);
}

double ComputeEpipolarConstraint(const std::vector<Eigen::Vector2d>& points_1,
                                 const std::vector<Eigen::Vector2d>& points_2,
                                 const Eigen::Matrix3d&              essential_matrix)
{
    if (points_1.size() != points_2.size())
    {
        throw std::invalid_argument("Both points vectors must be of same size");
    }

    double sum = 0.0;
    for (size_t i = 0; i < points_1.size(); ++i)
    {
        sum += ComputeEpipolarConstraint(points_1[i], points_2[i], essential_matrix);
    }

    sum /= points_1.size();

    return sum;
}
std::tuple<std::vector<Eigen::Matrix3d>, std::vector<Eigen::Matrix3d>, std::vector<Eigen::Vector3d>>
EstimateMotionFromComplexRoots(const std::vector<std::complex<double>>& roots,
                               const Eigen::RowVectorXd& p_1, const Eigen::RowVectorXd& p_2,
                               const Eigen::RowVectorXd& p_3, const Eigen::Matrix3d& essential_x,
                               const Eigen::Matrix3d& essential_y,
                               const Eigen::Matrix3d& essential_z,
                               const Eigen::Matrix3d& essential_w, const Eigen::Vector3d& q_1,
                               const Eigen::Vector3d& q_2)
{
    if (p_1.size() != 8)
    {
        throw std::invalid_argument("Polynomial p_1 must be degree 7");
    }
    if (p_2.size() != 8)
    {
        throw std::invalid_argument("Polynomial p_2 must be degree 7");
    }
    if (p_3.size() != 7)
    {
        throw std::invalid_argument("Polynomial p_2 must be degree 6");
    }
    std::vector<Eigen::Matrix3d> essential_matrices;
    std::vector<Eigen::Matrix3d> rotation_matrices;
    std::vector<Eigen::Vector3d> translation_vectors;
    for (const auto& root : roots)
    {
        if (!IsReal(root))
        {
            continue;
        }
        const double       z  = root.real();
        double             z2 = z * z;
        double             z3 = z2 * z;
        double             z4 = z3 * z;
        double             z5 = z4 * z;
        double             z6 = z5 * z;
        double             z7 = z6 * z;
        Eigen::RowVectorXd p_z6(7);
        p_z6 << z6, z5, z4, z3, z2, z, 1.0;
        Eigen::RowVectorXd p_z7(8);
        p_z7(0)           = z7;
        p_z7.tail(7)      = p_z6;
        const double    x = p_1.dot(p_z7) / p_3.dot(p_z6);
        const double    y = p_2.dot(p_z7) / p_3.dot(p_z6);
        Eigen::Matrix3d essential_matrix =
            x * essential_x + y * essential_y + z * essential_z + essential_w;
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(essential_matrix,
                                              Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix3d                   U = svd.matrixU(); // Left singular vectors
        Eigen::Matrix3d                   V = svd.matrixV();
        Eigen::Matrix3d                   D = Eigen::Vector3d(1.0, 1.0, 0.0).asDiagonal();
        essential_matrix                    = U * D * V.transpose();
        essential_matrices.emplace_back(essential_matrix);
        if (U.determinant() < 0)
        {
            U.col(2) = -U.col(2);
        }

        if (V.determinant() < 0)
        {
            V.col(2) = -V.col(2);
        }
        D << 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0;
        Eigen::Vector3d t;
        Eigen::Matrix3d R;
        for (size_t i = 0; i < 4; ++i)
        {
            switch (i)
            {
                case 0:
                    t = U.col(2);
                    R = U * D * V.transpose();
                    break;
                case 1:
                    t = -U.col(2);
                    R = U * D * V.transpose();
                    break;
                case 2:
                    t = U.col(2);
                    R = U * D.transpose() * V.transpose();
                    break;
                case 3:
                    t = -U.col(2);
                    R = U * D.transpose() * V.transpose();
                    break;
            }
            const Eigen::Vector3d a = essential_matrix.transpose() * q_2;
            const Eigen::Vector3d b = q_1.cross(Eigen::Vector3d(a.x(), a.y(), 0.0));
            Eigen::Matrix3d       diagonal_mat_temp;
            diagonal_mat_temp << 1, 0, 0, 0, 1, 0, 0, 0, 0;
            const Eigen::Vector3d c = q_2.cross(diagonal_mat_temp * essential_matrix * q_1);
            const Eigen::Vector3d d = a.cross(b);

            Eigen::Matrix<double, 3, 4> P;
            P << R, t;
            Eigen::Vector4d C = P.transpose() * c;
            Eigen::Vector4d Q;
            Q.head<3>() = d * C(3);
            Q(3)        = -d.dot(C.head<3>());

            if (Q(2) * Q(3) < 0.0)
            {
                continue;
            }

            Eigen::Vector3d c_2 = P * Q;
            if (c_2(2) * Q(3) < 0.0)
            {
                continue;
            }
            rotation_matrices.emplace_back(R);
            translation_vectors.emplace_back(t);
            break;
        }
    }

    return {essential_matrices, rotation_matrices, translation_vectors};
}
