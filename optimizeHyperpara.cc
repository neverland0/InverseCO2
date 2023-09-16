#include <iostream>
#include <fstream>
#include <string>
#include <Eigen/Dense>
#include <cmath>
#include <ctime>
#include <nlopt.h>
#include <math.h>
#include <assert.h>

using namespace std;
using namespace Eigen;

//#define MAXBUFSIZE  ((int) 1e6)
#define MAXBUFSIZE  ((int) 1e9)
static int lon;
static int lat;
static int counts = 0;

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

void writeMatrix(MatrixXd &Mat, const char *filename)
{
    ofstream outfile;
    outfile.open(filename, ios::trunc);

    outfile << Mat;
    outfile.close();
    return;
}

MatrixXd readMatrix(const string &filename)
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

double exponential_cf(double dist, double sigma, double l)
{
    return pow(sigma,2) * exp(-dist/l);
}

double exponential_cf_l(double dist, double sigma, double l)
{
    return pow(sigma,2) * exp(-dist/l) * dist * pow(l,-2);
}

double exponential_cf_sigma(double dist, double sigma, double l)
{
    return sigma * 2 * exp(-dist/l);
}

typedef double (*cf_pointer)(double,double,double);

void gen_cov_matrix(MatrixXd &Mat, double sigma, double l,cf_pointer cfp, bool isR, bool prime = false)
{
    assert(Mat.rows() == Mat.cols());
    for (int i = 0; i < Mat.rows(); ++i) {
        for (int j = i;  j < Mat.cols(); ++j) {
            if(i == j)
            {
                if(!prime)
                {
                    Mat(i,j) = pow(sigma,2);
                }
                else
                {
                    Mat(i,j) = sigma * 2;
                }
                //cout << "r" << i << ' ' << j << endl;
            }
            else
            {
                double r;
                if(isR)
                {
                    r = abs(i-j);
                //cerr << "R" << i << ' ' << j << endl;
                }
                else
                {
                    r = norm(i,j,lon);
                //cerr << "B" << i << ' ' << j << endl;
                }
                Mat(i,j) = cfp(r,sigma,l);
                Mat(j,i) = Mat(i,j);
            }
        }
    }
    
}

double likehood(unsigned n, const double *x, double *grad, void *data)
{
    myData *md = (myData *)data;

    double sigma_o = x[0];
    double r_l = x[1];
    double sigma_b = x[2];
    double b_l = x[3];

    cout << "x1,x2,x3,x4=" << x[0] <<" " << x[1] << " " << x[2] << " " << x[3] << endl; 
    
    MatrixXd H = md->H;
    MatrixXd y = md->y;
    MatrixXd x_prior = md->x_prior;
    MatrixXd shape = md->shape;
    int obs_num = md->y.rows();
    int grids_num = md->x_prior.rows();
    lon = shape(0,0);
    lat = shape(1,0);
    
    cout << "1" << endl;
    MatrixXd R_theta = MatrixXd::Zero(obs_num,obs_num);
    gen_cov_matrix(R_theta,sigma_o,r_l,exponential_cf,true,false);
    cout << "2" << endl;

    MatrixXd B_theta = MatrixXd::Zero(grids_num,grids_num);
    gen_cov_matrix(B_theta,sigma_b,b_l,exponential_cf,false,false);
    cout << "3" << endl;

    MatrixXd R_o = MatrixXd::Zero(obs_num,obs_num);
    gen_cov_matrix(R_o,sigma_o,r_l,exponential_cf_sigma,true,true);
    cout << "4" << endl;

    MatrixXd R_l = MatrixXd::Zero(obs_num,obs_num);
    gen_cov_matrix(R_l,sigma_o,r_l,exponential_cf_l,true,false);
    cout << "5" << endl;

    MatrixXd B_b = MatrixXd::Zero(grids_num,grids_num);
    gen_cov_matrix(B_b,sigma_b,b_l,exponential_cf_sigma,false,true);
    cout << "6" << endl;

    MatrixXd B_l = MatrixXd::Zero(grids_num,grids_num);
    gen_cov_matrix(B_l,sigma_b,b_l,exponential_cf_l,false,false);
    cout << "7" << endl;
    
    clock_t start = clock();
    MatrixXd D_theta = R_theta + H * B_theta * H.transpose();
    clock_t end = clock();
    cout << "乘法花费了" << (double)(end - start) / CLOCKS_PER_SEC << "秒" << endl;
    cout << "8" << endl;
    start = clock();
    MatrixXd D_inv = D_theta.inverse();
    end = clock();
    cout << "求逆花费了" << (double)(end - start) / CLOCKS_PER_SEC << "秒" << endl;
    cout << "9" << endl;
    MatrixXd alpha = D_inv * y;
    cout << "10" << endl;
    MatrixXd p1 = D_inv - alpha * alpha.transpose();
    cout << "11" << endl;

    if (grad) {
    //cout << "test1" << endl;
        grad[0] = 0.5 * (p1 * R_o).trace();
    cout << "12" << endl;
        grad[1] = 0.5 * (p1 * R_l).trace();
    cout << "13" << endl;
        grad[2] = 0.5 * (p1 * H * B_b * H.transpose()).trace();
    cout << "14" << endl;
        grad[3] = 0.5 * (p1 * H * B_l * H.transpose()).trace();
    }	

    cout << "15" << endl;
    MatrixXd y_Hxp = y - H * x_prior;

    cout << "16" << endl;
    auto ret1 = logdet(D_theta);
    cout << "17" << endl;
    auto ret2 = y_Hxp.transpose() * D_inv * y_Hxp;
    double ret = ret1 + ret2(0,0);
    cout << "ret1 = " << ret1 << " ret2 = " << ret2 << endl;
    cout << "ret = " << ret << endl;
    counts++;
    return ret;
    
}


int main(int argc, char *argv[])
{
    string pre = "/home/akagi/InverseCO2/data/hhsd/";
    MatrixXd y = readMatrix(pre + "y.txt");
    MatrixXd H = readMatrix(pre + "H.txt");
    MatrixXd x_prior = readMatrix(pre + "x_prior.txt");
    MatrixXd flux = readMatrix(pre + "flux.txt");
    MatrixXd shape = readMatrix(pre + "shape.txt");
    cout << "read data from file" << endl;
    myData md = {y,H,x_prior,shape};
    myData *md_p = &md; 
    double lb[4] = { 0.0, 0.0, 0.0, 0.0 }; //lower bounds
    double ub[4] = {20.0 , 40.0, 20.0, 40.0 }; //upper bounds
     
    // create the optimization problem
    // opaque pointer type
    nlopt_opt opt;
    opt = nlopt_create(NLOPT_GN_ISRES, 4);
    //opt = nlopt_create(NLOPT_GD_STOGO, 4);
     
    nlopt_set_lower_bounds(opt,lb);
    nlopt_set_upper_bounds(opt,ub);
     
    nlopt_set_min_objective(opt, likehood,md_p);
     
    //nlopt_set_xtol_rel(opt, 1e-4);
    nlopt_set_maxtime(opt, 180);
     
    // initial guess
    double x[4]={10.0,20.0,10.0,20.0};
    double minf;
     
    nlopt_result res=nlopt_optimize(opt, x,&minf);
     
     
    if (res < 0) {
        printf("nlopt failed!\n");
    }
    else {
        printf("found minimum at f(%g,%g,%g,%g) = %0.10g\n", x[0], x[1],x[2],x[3], minf);
        cout << "optim times :" << counts << endl;
    }
    double sigma_o = x[0];
    double r_l = x[1];
    double sigma_b = x[2];
    double b_l = x[3];
    int obs_num = y.rows();
    int grids_num = x_prior.rows();
    MatrixXd R_theta = MatrixXd::Zero(obs_num,obs_num);
    gen_cov_matrix(R_theta,sigma_o,r_l,exponential_cf,true,false);

    MatrixXd B_theta = MatrixXd::Zero(grids_num,grids_num);
    gen_cov_matrix(B_theta,sigma_b,b_l,exponential_cf,false,false);

    MatrixXd D_theta = R_theta + H * B_theta * H.transpose();
    MatrixXd D_inv = D_theta.inverse();

    MatrixXd posterior = x_prior + B_theta * H.transpose() * D_inv * (y - H * x_prior);

    VectorXd diff = posterior - flux;
    float se = diff.dot(diff);
    float rmse = sqrt(se/grids_num);


    writeMatrix(posterior,"posterior.txt");
    writeMatrix(x_prior,"x_prior.txt");
    writeMatrix(flux,"flux.txt");
    cout << "posterior sum = " << posterior.sum() << endl;
    cout << "prior sum = " << x_prior.sum() << endl;
    cout << "true flux sum = " << flux.sum() << endl;
    cout << "rmse = " << rmse << endl;
        return 0;
}
