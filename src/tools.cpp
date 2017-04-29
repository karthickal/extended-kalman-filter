#include <iostream>
#include "tools.h"

using namespace std;
using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {

    // Initialize the rmse values
    VectorXd rmse(4);
    rmse << 0, 0, 0, 0;

    // check if values are valid
    if (estimations.size() == 0 || estimations.size() != ground_truth.size()) {
        cout << "The estimation values are incorrect." << endl;
        return rmse;
    }

    // Calculate the root mean squared error
    for (int i = 0; i < estimations.size(); ++i) {
        VectorXd residual = estimations[i] - ground_truth[i];
        residual = residual.array() * residual.array();
        rmse += residual;
    }
    rmse = rmse / estimations.size();
    rmse = rmse.array().sqrt();

    return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd &x_state) {

    // Initialize the jacobian matrix
    MatrixXd hj(3, 4);
    hj.setZero();

    // Store all reusable calculations and values
    float px = x_state(0);
    float py = x_state(1);
    float vx = x_state(2);
    float vy = x_state(3);
    float px2 = px * px;
    float py2 = py * py;
    float sum_p2 = px2 + py2;
    float sqrt_p2 = sqrt(sum_p2);
    float ohrt_p2 = sum_p2 * sqrt_p2;
    float vxpy = vx * py;
    float vypx = vy * px;

    // check if division by zero is a possibility
    if (fabs(sum_p2) > 0.00001f) {
        // create the jacobian and return
        hj << px / sqrt_p2, py / sqrt_p2, 0, 0,
                -py / sum_p2, px / sum_p2, 0, 0,
                (py * (vxpy - vypx)) / ohrt_p2, (px * (vypx - vxpy)) / ohrt_p2, px / sqrt_p2, py / sqrt_p2;
    }

    return hj;

}
