#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <armadillo>

// Returns a 9xN matrix whose columns are an orthonormal basis for the null space of A
Eigen::MatrixXd nullSpaceBasisMatrix(const Eigen::MatrixXd& A, double tol = 1e-10) {
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
    const Eigen::VectorXd& singularValues = svd.singularValues();
    const Eigen::MatrixXd& V = svd.matrixV(); // V is n x n, where n = A.cols()

    // Determine rank based on tolerance
    int rank = 0;
    for (int i = 0; i < singularValues.size(); ++i) {
        if (singularValues(i) > tol)
            ++rank;
    }

    int nullity = V.cols() - rank;

    // Extract the last 'nullity' columns of V
    Eigen::MatrixXd nullSpace = V.rightCols(nullity); // size: A.cols() x nullity

    return nullSpace;
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
    companion.row(0) = -poly.segment(1, 10);           // -coefficients from z^9 to 1
  
    std::cout << "Companion Matrix: \n" << companion << std::endl;

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

int main() {
    arma::mat A = arma::randu<arma::mat>(3, 3);
    arma::mat L, U, P;
    arma::lu(L, U, P, A);
    A.print("Original A:");
    L.print("L:");
    U.print("U:");
    P.print("P:");
    return 0;
}