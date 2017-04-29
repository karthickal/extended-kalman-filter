#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

const int NOISE_AX = 9;
const int NOISE_AY = 9;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {

    // initialize object properties
    is_initialized_ = false;
    previous_timestamp_ = 0.0f;

    // initializing matrices
    R_laser_ = MatrixXd(2, 2);
    R_radar_ = MatrixXd(3, 3);
    H_laser_ = MatrixXd(2, 4);
    Hj_ = MatrixXd(3, 4);

    //measurement covariance matrix - laser
    R_laser_ << 0.0225, 0.0,
            0.0, 0.0225f;

    //measurement covariance matrix - radar
    R_radar_ << 0.09, 0.0, 0.0,
            0.0, 0.0009, 0.0,
            0.0, 0.0, 0.09f;

    // initialize the state transition matrix
    ekf_.F_ = MatrixXd(4, 4);
    ekf_.F_ << 1, 0, 1, 0,
            0, 1, 0, 1,
            0, 0, 1, 0,
            0, 0, 0, 1;

    // initialize the observation matrix
    H_laser_ << 1, 0, 0, 0,
            0, 1, 0, 0;

    // initialize the covariance matrix
    ekf_.P_ = MatrixXd(4, 4);
    ekf_.P_ << 1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1000, 0,
            0, 0, 0, 1000;

    // initialize the noise covariance matrix
    ekf_.Q_ = MatrixXd(4, 4);
}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}


void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {

    /*****************************************************************************
     *  Initialization
     ****************************************************************************/
    /* If this is the first time we are measuring */
    if (!is_initialized_) {

        // first measurement
        cout << "EKF: " << endl;
        ekf_.x_ = VectorXd(4);
        ekf_.x_ << 1, 1, 1, 1;

        if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {

            /**
            Convert radar from polar to cartesian coordinates and initialize state.
            */
            float rho = measurement_pack.raw_measurements_[0];
            float phi = measurement_pack.raw_measurements_[1];
            float rho_dot = measurement_pack.raw_measurements_[2];

            // set the cartesian coordinates to the state
            ekf_.x_ << rho * cos(phi), rho * sin(phi), rho_dot * cos(phi), rho_dot * sin(phi);
        } else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {

            /**
            Initialize state by using the measurements.
            */
            ekf_.x_ << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1], 0, 0;
        }

        // done initializing, no need to predict or update
        // store the timestamp to be used later
        previous_timestamp_ = measurement_pack.timestamp_;
        is_initialized_ = true;

        return;
    }

    // calculate the timestamp difference and modify the state transition matrix
    float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
    previous_timestamp_ = measurement_pack.timestamp_;
    ekf_.F_(0, 2) = dt;
    ekf_.F_(1, 3) = dt;

    float dt2 = dt * dt;
    float dt3 = dt2 * dt;
    float dt4 = dt3 * dt;

    // modify the process noise covariance matrix.
    ekf_.Q_ << (dt4 / 4) * NOISE_AX, 0, (dt3 / 2) * NOISE_AX, 0,
            0, (dt4 / 4) * NOISE_AY, 0, (dt3 / 2) * NOISE_AY,
            (dt3 / 2) * NOISE_AX, 0, (dt2) * NOISE_AX, 0,
            0, (dt3 / 2) * NOISE_AY, 0, (dt2) * NOISE_AY;

    // predict the new state
    ekf_.Predict();

    /*****************************************************************************
     *  Update
     ****************************************************************************/
    /* Update the radar measurement and sensor covariance matrices */
    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {

        /* Compute the jacobian, which will become the measurement matrix */
        ekf_.H_ = tools.CalculateJacobian(ekf_.x_);
        // update only if we got a valid Jacobian
        if (!ekf_.H_.isZero()) {
            ekf_.R_ = R_radar_;
            ekf_.UpdateEKF(measurement_pack.raw_measurements_);
        }

    }/* Update the laser measurement and sensor covariance matrices */
    else {
        ekf_.H_ = H_laser_;
        ekf_.R_ = R_laser_;
        ekf_.Update(measurement_pack.raw_measurements_);
    }

    /* Print out the state and covariance */
    //cout << "x_ = " << ekf_.x_ << endl;
    //cout << "P_ = " << ekf_.P_ << endl;
    //cin >> dt;
}
