//
// Created by TESTER on 03/11/2023.
//

#ifndef TFG_FUNCTIONS_H
#define TFG_FUNCTIONS_H
#include <complex>
#include <vector>
#include <chrono>
#include<unordered_map>

using namespace std;

void setX(int npoints, string xmode);

void setR(double r);

void setSNR(double snr);

void setMod(int mod, string xmode);

string complextostr(complex<double> x);

void initQ();

double Q(int a);

unsigned long long factorial(unsigned int n);

double abs_sq(std::complex<double> a);

vector<double> Hroots(int n);

inline double hermite_w(int n, double root, unsigned long long fact);

vector<double> Hweights(int my_n);

inline double cost(complex<double> current_x);

inline double W(complex<double> y, complex<double> x);

inline double G(double alpha, complex<double> xhat, complex<double> y, double rho);

inline double G_co(double r, complex<double> xhat, complex<double> y, double rho);

inline double H_co(double r, complex<double> x, complex<double> y, double rho);

inline double H(double alpha, complex<double> x, complex<double> y, double rho);

double fa(complex<double> x, complex<double> y, vector<double> alphas, double rho, int xind);

double fa_co(complex<double> x, complex<double> y, double r, double rho);

double E_0(double rho, vector<double> alphas, int n);

double E_0_co(double r, double rho, double& grad_rho, double& E0);

double E_0_co(double r, double rho, double& grad_rho, double& grad_2_rho, double& E0, int n, vector<double> hweights, vector<double> multhweights, vector<double> roots);

double E_0_co(double r, double rho, double& grad_rho, double& E0, int n, vector<double> hweights, vector<double> multhweights, vector<double> roots);

inline void gradient_f(complex<double> x, complex<double> y, vector<double> alphas, double rho, vector<double>& grads_alpha, double& grad_rho, int xindex);

double e02(int n);

inline void gradient_e0(vector<double> alphas, double rho, vector<double>& grads_alpha, double& grad_rho, int my_n, vector<double> hweights, vector<double> mult, vector<double> roots);

inline void gradient_e0_co(double r, double rho, double& grad_r, double& grad_rho, int my_n, vector<double> hweights, vector<double> mult, vector<double> roots);

inline vector<double> mult_newhweights(vector<double> hweights, int my_n);

void compute_hweights(int n, int num_iterations);

double GD_ccomp(vector<double>& alphas, double& rho, double learning_rate, int num_iterations, int n);

double GD_co(double& r, double& rho, double learning_rate, int num_iterations, int n, bool updateR);

double GD_iid(double& r, double& rho, double& rho_interpolated, int num_iterations, int n);

double GD_cc(double& r, double& rho, double learning_rate, int num_iterations, int n);

void NAG_update(double &x_t, double &y_t, double &x_tp1, double &y_tp1, double beta, double grad, double kaux);

double NAG(vector<double>& alphas, double& rho, double learning_rate, int num_iterations, int n, double k);

double NAG_co(double& r, double& rho, double learning_rate, int num_iterations, int n, double k, bool updateR);

double NAG_iid(double& r, double& rho, double learning_rate, int num_iterations, int n, double k);

double NAG_cc(double& r, double& rho, double learning_rate, int num_iterations, int n, double k);

double NM_co(double& r, double& rho, int num_iterations, int n, bool updateR);

vector<chrono::microseconds> getE0_times();

void test();

void setQ();

void setPI();

void setW();

unordered_map<string, vector<chrono::microseconds>> get_times();

void setN(int n);

vector<double> getAllHweights();

vector<double> getAllRoots();

vector<double> getAllMultHweights();

double E_0_1_co_old();

double E_0_1_co();

double E_0_2_co();

void setA(vector<int> mods);

double initial_guess(double r, double E0_0, double E0_1, double E0_0_der, double E0_1_der, double& max_g);

#endif //TFG_FUNCTIONS_H
