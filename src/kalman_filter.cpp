#include "kalman_filter.h"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
    x_ = x_in;
    P_ = P_in;
    F_ = F_in;
    H_ = H_in;
    R_ = R_in;
    Q_ = Q_in;
}

void KalmanFilter::Predict() {

    // predict the state using the state function matrix
    x_ = F_ * x_;
    P_ = (F_ * P_ * F_.transpose()) + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {

    // use the general kalman filter algorithm to update
    VectorXd y = z - (H_ * x_);
    MatrixXd Ht = H_.transpose();
    MatrixXd S = H_ * P_ * Ht + R_;
    MatrixXd K = P_ * Ht * S.inverse();
    x_ = x_ + (K * y);

    MatrixXd I = MatrixXd::Identity(x_.size(), x_.size());
    P_ = (I - (K * H_)) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {

    // use the extended kalman filter algorithm to update
    // store the calculation fields
    float px = x_[0];
    float py = x_[1];
    float vx = x_[2];
    float vy = x_[3];
    float sqrtp2 = sqrt((px * px) + (py * py));
    float pi = M_PI;

    float rho;
    float phi;
    float rho_dot;

    // calculate the  polar coordinates
    if (fabs(sqrtp2) < 0.0001) {
        rho = phi = rho_dot = 0;
    } else {
        rho = sqrtp2;
        phi = atan2(py, px);
        rho_dot = ((px * vx) + (py * vy)) / sqrtp2;
    }

    VectorXd hx(3);
    hx << rho, phi, rho_dot;

    VectorXd y = z - hx;
    // normalize the phi value to be between -pi to pi
    while ((y[1] < -pi) || (y[1] > pi)) {
        if (y[1] < -pi) {
            y[1] = y[1] + (2 * pi);
        }

        if (y[1] > pi) {
            y[1] = y[1] - (2 * pi);
        }
    }

    // use the kalman filter algorithm to update
    MatrixXd Ht = H_.transpose();
    MatrixXd S = H_ * P_ * Ht + R_;
    MatrixXd K = P_ * Ht * S.inverse();
    x_ = x_ + (K * y);

    MatrixXd I = MatrixXd::Identity(x_.size(), x_.size());
    P_ = (I - (K * H_)) * P_;
}
