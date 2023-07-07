#include <iostream>
#include <fstream>
#include <string>
#include <Eigen/Dense>
#include <cmath>
#include <nlopt.h>

using namespace std;
using namespace Eigen;

#define MAXBUFSIZE  ((int) 1e6)
//#define MAXBUFSIZE  ((int) 1e9)

template <typename MatrixType>
inline typename MatrixType::Scalar logdet(const MatrixType& M, bool use_cholesky = false) {
  using namespace Eigen;
  using std::log;
  typedef typename MatrixType::Scalar Scalar;
  Scalar ld = 0;
  if (use_cholesky) {
    LLT<Matrix<Scalar,Dynamic,Dynamic>> chol(M);
    auto& U = chol.matrixL();
    for (unsigned i = 0; i < M.rows(); ++i)
      ld += log(U(i,i));
    ld *= 2;
  } else {
    PartialPivLU<Matrix<Scalar,Dynamic,Dynamic>> lu(M);
    auto& LU = lu.matrixLU();
    Scalar c = lu.permutationP().determinant(); // -1 or 1
    for (unsigned i = 0; i < LU.rows(); ++i) {
      const auto& lii = LU(i,i);
      if (lii < Scalar(0)) c *= -1;
      ld += log(abs(lii));
    }
    ld += log(c);
  }
  return ld;
}

MatrixXd readMatrix(const char *filename)
    {
    int cols = 0, rows = 0;
    double *buff = new double[MAXBUFSIZE];

    // Read numbers from file into buffer.
    ifstream infile;
    infile.open(filename);
    while (!infile.eof())
        {
        string line;
        getline(infile, line,'\n');

        int temp_cols = 0;
        stringstream stream(line);
        while(!stream.eof())
            stream >> buff[cols*rows+temp_cols++];

        if (temp_cols == 0)
            continue;

        if (cols == 0)
            cols = temp_cols;

        rows++;
        }

    infile.close();

    rows--;

    // Populate matrix with numbers.
    MatrixXd result(rows,cols);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            result(i,j) = buff[ cols*i+j ];

    return result;
    };

typedef struct{
    MatrixXd y;
    MatrixXd H;
    MatrixXd x_prior;
    MatrixXd shape;
}myData;

double norm(int i, int j, int lon)
{
    int x1 = i / lon;
    int y1 = i % lon;
    int x2 = j / lon;
    int y2 = j % lon;

    return sqrt(pow(x1-x2,2)+pow(y1-y2,2));
}

double likehood(unsigned n, const double *x, double *grad, void *data)
{
    myData *md = (myData *)data;

    double sigma_o = x[0];
    double sigma_b = x[1];
    double l = x[2];
    cout << "x0=" << x[0] << " x1=" << x[1] << " x2=" << x[2] << endl;
    
    MatrixXd H = md->H;
    MatrixXd y = md->y;
    MatrixXd x_prior = md->x_prior;
    MatrixXd shape = md->shape;
    int obs_num = md->y.rows();
    int grids_num = md->x_prior.rows();
    int lon = shape(0,0);
    int lat = shape(1,0);
    MatrixXd R_theta = MatrixXd::Identity(obs_num,obs_num);
    R_theta = pow(sigma_o,2) * R_theta;

    MatrixXd B_theta = MatrixXd::Zero(grids_num,grids_num);
    for (int i = 0; i < grids_num; ++i) {
        for (int j = i;  j < grids_num; ++j) {
            if(i == j)
            {
                B_theta(i,j) = pow(sigma_b,2);
            }
            else
            {
                double r = norm(i,j,lon);
                B_theta(i,j) = pow(sigma_b,2) * exp(-r/l);
                B_theta(j,i) = B_theta(i,j);
                //cout << "norm= " << norm(i,j,lon) << " i=" << i << " j=" << j << endl;
            }
        }
    }

    MatrixXd R_sigmao = MatrixXd::Identity(obs_num,obs_num);
    R_sigmao = sigma_o * 2 * R_sigmao;

    MatrixXd B_sigmab = MatrixXd::Zero(grids_num,grids_num);
    for (int i = 0; i < grids_num; ++i) {
        for (int j = i;  j < grids_num; ++j) {
            if(i == j)
            {
                B_sigmab(i,j) = sigma_b * 2;
            }
            else
            {
                double r = norm(i,j,lon);
                B_sigmab(i,j) = sigma_b * 2 * exp(-r/l);
                B_sigmab(j,i) = B_theta(i,j);
            }
        }
    }

    MatrixXd B_l = MatrixXd::Zero(grids_num,grids_num);
    for (int i = 0; i < grids_num; ++i) {
        for (int j = i;  j < grids_num; ++j) {
            if(i == j)
            {
                B_l(i,j) = pow(sigma_b, 2);
            }
            else
            {
                double r = norm(i,j,lon);
                B_l(i,j) = pow(sigma_b ,2) * exp(-r/l) * r * pow(l,-2);
                B_l(j,i) = B_theta(i,j);
            }
        }
    }
    
    MatrixXd D_theta = R_theta + H * B_theta * H.transpose();
    MatrixXd D_inv = D_theta.inverse();
    MatrixXd alpha = D_inv * y;
    MatrixXd p1 = D_inv - alpha * alpha.transpose();

    if (grad) {
    //cout << "test1" << endl;
        grad[0] = 0.5 * (p1 * R_sigmao).trace();
        grad[1] = 0.5 * (p1 * H * B_sigmab * H.transpose()).trace();
        grad[2] = 0.5 * (p1 * H * B_l * H.transpose()).trace();
    }	

    MatrixXd y_Hxp = y - H * x_prior;

    auto ret1 = logdet(D_theta);
    auto ret2 = y_Hxp.transpose() * D_inv * y_Hxp;
    double ret = ret1 + ret2(0,0);
    cout << "ret1 = " << ret1 << " ret2 = " << ret2 << endl;
    cout << "ret = " << ret << endl;
    return ret;
    
}


int main(int argc, char *argv[])
{
    MatrixXd y = readMatrix("./data/hhsd/y.txt");
    MatrixXd H = readMatrix("./data/hhsd/H.txt");
    MatrixXd x_prior = readMatrix("./data/hhsd/x_prior.txt");
    MatrixXd shape = readMatrix("./data/hhsd/shape.txt");
    cout << "read data from file" << endl;
    myData md = {y,H,x_prior,shape};
    myData *md_p = &md; 
    double lb[3] = { 0.0, 0.0, 0.0 }; //lower bounds
    double ub[3] = {5.0 , 5.0, 50.0 }; //upper bounds
     
    // create the optimization problem
    // opaque pointer type
    nlopt_opt opt;
    opt = nlopt_create(NLOPT_GD_STOGO, 3);
     
    nlopt_set_lower_bounds(opt,lb);
    nlopt_set_upper_bounds(opt,ub);
     
    nlopt_set_min_objective(opt, likehood,md_p);
     
    //nlopt_set_xtol_rel(opt, 1e-4);
    nlopt_set_maxtime(opt, 60);
     
    // initial guess
    double x[3]={2.5,2.5,25.0};
    double minf;
     
    nlopt_result res=nlopt_optimize(opt, x,&minf);
     
     
    if (res < 0) {
        printf("nlopt failed!\n");
    }
    else {
        printf("found minimum at f(%g,%g,%g) = %0.10g\n", x[0], x[1],x[2], minf);
    }

        return 0;
}
