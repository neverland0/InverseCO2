#include <iostream>
#include <fstream>
#include <string>
#include <Eigen/Dense>
#include <cmath>
#include <ctime>
#include <nlopt.h>
#include <nlopt.hpp>
#include <math.h>
#include <assert.h>
#include <torch/torch.h>
#include <ATen/ATen.h>
#include <sys/stat.h>
#include <unistd.h>

using namespace std;
using namespace Eigen;

//#define MAXBUFSIZE  ((int) 1e6)
#define MAXBUFSIZE  ((int) 1e9)
#define R_COV exp_Covfun
#define B_COV balgo_Covfun
#define STR1(R)  #R
#define STR(R)  STR1(R)
#define PI 3.1415926
#define OBS 744
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXdr;
static int lon;
static int lat;
static int counts = 0;
static int p1_num;
static int p2_num;

static int obs_num;
static int grids_num;
MatrixXdr y;
MatrixXdr H;
MatrixXdr H_v;
MatrixXdr obs_v;
MatrixXdr x_prior;
MatrixXdr flux;

bool folderExists(const char* folderPath) {
    struct stat info;
    return (stat(folderPath, &info) == 0 );
}

bool createFolder(const char* folderPath) {
    int status = mkdir(folderPath, S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH);
    if (status == 0) {
        std::cout << "文件夹创建成功" << std::endl;
        return true;
    } else {
        std::cerr << "无法创建文件夹" << std::endl;
        return false;
    }
}


class CovfunBase
{
    protected:
        vector<double> para;
        int dim;
        vector<MatrixXdr*> vec_mat;
        bool isR;
    public:
        CovfunBase(vector<double> x, int n,bool isR):para(x),dim(n),isR(isR)
        {
            for(int i=0;i<para.size()+1;i++)
            {
                vec_mat.push_back(new MatrixXdr(dim,dim));
            }
        }
        double norm(int i, int j, int lon)
        {
            int x1 = i / lon;
            int y1 = i % lon;
            int x2 = j / lon;
            int y2 = j % lon;

            return sqrt(pow(x1-x2,2)+pow(y1-y2,2));
        }
        void gen_matrix(MatrixXdr &Mat, int index)
        {
            for (int i = 0; i < dim; ++i) {
                for (int j = i;  j < dim; ++j) {
                    if(i == j)
                    {
                        Mat(i,j) = cov_pd(index,0,para);
                    }
                    else
                    {
                        double r;
                        if(isR)
                        {
                            r = abs(i-j);
                            if(i/OBS == j/OBS)
                            {
                                Mat(i,j) = cov_pd(index,r,para);
                                Mat(j,i) = Mat(i,j);
                            }
                            else
                            {
                                Mat(i,j) = 0;
                                Mat(j,i) = Mat(i,j);
                            }
                        }
                        else
                        {
                            r = norm(i,j,lon);
                            Mat(i,j) = cov_pd(index,r,para);
                            Mat(j,i) = Mat(i,j);
                        }
                    }
                }
            }
        }
        MatrixXdr* gen_covpd_fun(int i)
        {
            MatrixXdr &mat = *vec_mat[i];
            gen_matrix(mat,i);
            return &mat;
        }


        virtual double cov_pd(int i,double r,vector<double> para)=0;

        //virtual double pd(int i,double r,vector<double> para)=0;

        virtual ~CovfunBase()
        {
            for(MatrixXdr* b : vec_mat)
            {
                delete b;
            }
        }
    private:
};

class exp_Covfun : public CovfunBase
{
    public:
    exp_Covfun(vector<double> x, int n,bool isR):CovfunBase(x,n,isR)
    {}
    double cov_pd(int i,double r, vector<double> para) override
    {
        double sigma = para[0];
        double l = para[1];
        if(0==i)//cov fun
        {
            if(0==r)
            {
                return pow(sigma,2);
            }
            return pow(sigma,2) * exp(-r/l);
        }
        else if(1==i)//cov pd to para 1 -> sigma
        {
            if(0==r)
            {
                return sigma*2;
            }
            return sigma * 2 * exp(-r/l);
        }else if(2==i)//cov pd to para 2 -> l
        {
            if(0==r)
            {
               return pow(sigma,2);
            }
            return pow(sigma,2) * exp(-r/l) * r * pow(l,-2);
        }
        return 0;
    }

};
class balgo_Covfun : public CovfunBase
{
    public:
    balgo_Covfun(vector<double> x, int n,bool isR):CovfunBase(x,n,isR)
    {}
    double cov_pd(int i,double r, vector<double> para) override
    {
        double sigma = para[0];
        double l = para[1];
        if(0==i)//cov fun
        {
            if(0==r)
            {
                return pow(sigma,2);
            }
            return pow(sigma,2)* (1+r/l) * exp(-r/l);
        }
        else if(1==i)//cov pd to para 1 -> sigma
        {
            if(0==r)
            {
                return sigma*2;
            }
            return sigma * 2 * (1+r/l) * exp(-r/l);
        }else if(2==i)//cov pd to para 2 -> l
        {
            if(0==r)
            {
               return pow(sigma,2);
            }
            return pow(sigma,2) *(-r*pow(l,-2) * (1+r/l) * exp(-r/l) + (1+r/l)* exp(-r/l) * r * pow(l,-2));
        }
        return 0;
    }

};

class sqr_exp_Covfun : public CovfunBase
{
    public:
    sqr_exp_Covfun(vector<double> x, int n,bool isR):CovfunBase(x,n,isR)
    {}
    double cov_pd(int i,double r, vector<double> para) override
    {
        double sigma = para[0];
        double l = para[1];
        if(0==i)//cov fun
        {
            if(0==r)
            {
                return pow(sigma,2);
            }
            return pow(sigma,2) * exp(-pow(r,2)/(2*pow(l,2)));
        }
        else if(1==i)//cov pd to para 1 -> sigma
        {
            if(0==r)
            {
                return sigma*2;
            }
            return sigma*2 * exp(-pow(r,2)/(2*pow(l,2)));
        }else if(2==i)//cov pd to para 2 -> l
        {
            if(0==r)
            {
               return pow(sigma,2);
            }
            return pow(sigma,2) * exp(-pow(r,2)/(2*pow(l,2))) * pow(r,2) * pow(l,-3);
        }
        return 0;
    }

};

class gamma_exp_Covfun : public CovfunBase
{
    public:
    gamma_exp_Covfun(vector<double> x, int n,bool isR):CovfunBase(x,n,isR)
    {}
    double cov_pd(int i,double r, vector<double> para) override
    {
        double sigma = para[0];
        double l = para[1];
        double gamma = para[2];
        if(0==i)//cov fun
        {
            if(0==r)
            {
                return pow(sigma,2);
            }
            return pow(sigma,2) * exp(-pow(r/l,gamma));
        }
        else if(1==i)//cov pd to para 1 -> sigma
        {
            if(0==r)
            {
                return sigma*2;
            }
            return sigma * 2 * exp(-pow(r/l,gamma));
        }else if(2==i)//cov pd to para 2 -> l
        {
            if(0==r)
            {
               return pow(sigma,2);
            }
            return pow(sigma,2) * exp(-pow(r/l,gamma)) * pow(r,gamma) * gamma * pow(l,-gamma-1);
        }else if(3==i)//cov pd to para3 -> gamma
        {
            if(0==r)
            {
               return pow(sigma,2);
            }
            return pow(sigma,2) * exp(-pow(r/l,gamma)) * -1 * pow(r/l,gamma) * log(r/l);
        }
        return 0;
    }

};

class rational_quad_Covfun : public CovfunBase
{
    public:
    rational_quad_Covfun(vector<double> x, int n,bool isR):CovfunBase(x,n,isR)
    {}
    double cov_pd(int i,double r, vector<double> para) override
    {
        double sigma = para[0];
        double beta = para[1];
        double alpha = para[2];
        double l = para[3];
        if(0==i)//cov fun
        {
            if(0==r)
            {
                return pow(sigma,2);
            }
            return pow(sigma,2) * pow(1+pow(r,2)/(2*alpha*pow(l,2)),-beta);
        }
        else if(1==i)//cov pd to para 1 -> sigma
        {
            if(0==r)
            {
                return sigma*2;
            }
            return sigma*2 * pow(1+pow(r,2)/(2*alpha*pow(l,2)),-beta);
        }else if(2==i)//cov pd to para 2 -> beta
        {
            if(0==r)
            {
               return pow(sigma,2);
            }
            return pow(sigma,2) * pow(1+pow(r,2)/(2*alpha*pow(l,2)),-beta)*log(1+pow(r,2)/(2*alpha*pow(l,2)));
        }else if(3==i)//cov pd to para 3 -> alpha
        {
            if(0==r)
            {
               return pow(sigma,2);
            }
            return pow(sigma,2) * pow(1+pow(r,2)/(2*alpha*pow(l,2)),-beta)*log(1+pow(r,2)/(2*alpha*pow(l,2)))*pow(r,2)/(2*pow(alpha*l,2));

        }else if(4==i)//cov pd to para 4 -> l
        {
            if(0==r)
            {
               return pow(sigma,2);
            }
            return pow(sigma,2)*beta * pow(1+pow(r,2)/(2*alpha*pow(l,2)),-beta-1)*pow(r,2)/(alpha*pow(l,3));
        }
        return 0;
    }

};
class sin_Covfun : public CovfunBase
{
    public:
    sin_Covfun(vector<double> x, int n,bool isR):CovfunBase(x,n,isR)
    {}
    double cov_pd(int i,double r, vector<double> para) override
    {
        double sigma = para[0];
        if(0==i)//cov fun
        {
            if(0==r)
            {
                return pow(sigma,2);
            }
            return pow(sigma,2) * exp(-2*PI*PI*pow(r,2)/(24*24))*cos(2*PI*r/24);
        }
        else if(1==i)
        {
            if(0==r)
            {
                return sigma*2;
            }
            return sigma*2 * exp(-2*PI*PI*pow(r,2)/(24*24))*cos(2*PI*r/24);

        }
        return 0;
    }

};
/*
class sin_Covfun : public CovfunBase
{
    public:
    sin_Covfun(vector<double> x, int n,bool isR):CovfunBase(x,n,isR)
    {}
    double cov_pd(int i,double r, vector<double> para) override
    {
        double sigma = para[0];
        double l = para[1];
        if(0==i)//cov fun
        {
            if(0==r)
            {
                return pow(sigma,2);
            }
            return pow(sigma,2) * exp(-sin(PI*r)/l);
        }
        else if(1==i)//cov pd to para 1 -> sigma
        {
            if(0==r)
            {
                return sigma*2;
            }
            return sigma * 2 * exp(-sin(PI*r)/l);
        }else if(2==i)//cov pd to para 2 -> l
        {
            if(0==r)
            {
               return pow(sigma,2);
            }
            return pow(sigma,2) * exp(-sin(PI*r)/l) * sin(PI*r) * pow(l,-2);
        }
        return 0;
    }

};
*/
class sqr_exp_sin_Covfun : public CovfunBase
{
    public:
    sqr_exp_sin_Covfun(vector<double> x, int n,bool isR):CovfunBase(x,n,isR)
    {}
    double cov_pd(int i,double r, vector<double> para) override
    {
        double sigma = para[0];
        double l1 = para[1];
        double l2 = para[2];
        if(0==i)//cov fun
        {
            if(0==r)
            {
                return pow(sigma,2);
            }
            return pow(sigma,2) * exp(-pow(r,2)/(2*pow(l1,2))-2*pow(sin(PI*r),2)/pow(l2,2));
        }
        else if(1==i)//cov pd to para 1 -> sigma
        {
            if(0==r)
            {
                return sigma*2;
            }
            return sigma*2 * exp(-pow(r,2)/(2*pow(l1,2))-2*pow(sin(PI*r),2)/pow(l2,2));
        }else if(2==i)//cov pd to para 2 -> l1
        {
            if(0==r)
            {
               return pow(sigma,2);
            }
            return pow(sigma,2) * exp(-pow(r,2)/(2*pow(l1,2))-2*pow(sin(PI*r),2)/pow(l2,2)) * pow(r,2)/pow(l1,3);
        }else if(3==i)//cov pd to para3 -> l2
        {
            if(0==r)
            {
               return pow(sigma,2);
            }
            return pow(sigma,2) * exp(-pow(r,2)/(2*pow(l1,2))-2*pow(sin(PI*r),2)/pow(l2,2))*4*pow(sin(PI*r),2)/pow(l2,3);
        }
        return 0;
    }

};
/*
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
*/
void writeMatrix(MatrixXdr &Mat, const char *filename)
{
    ofstream outfile;
    outfile.open(filename, ios::trunc);

    outfile << Mat;
    outfile.close();
    return;
}

MatrixXdr readMatrix(const string &filename)
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
    MatrixXdr result(rows,cols);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            result(i,j) = buff[ cols*i+j ];

    return result;
    };
/*
typedef struct{
    MatrixXdr y;
    MatrixXdr H;
    MatrixXdr x_prior;
    MatrixXdr shape;
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

void gen_cov_matrix(MatrixXdr &Mat, double sigma, double l,cf_pointer cfp, bool isR, bool prime = false)
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
*/

void eigen2torch(MatrixXdr &eigenMat, torch::Tensor &torchMat)
{
    //vector<float> data(eigenMat.data(), eigenMat.data() + eigenMat.size());
    vector<double> data(eigenMat.data(), eigenMat.data() + eigenMat.size());
    int rows = eigenMat.rows();
    int cols = eigenMat.cols();
    torchMat = torch::from_blob(data.data(), {rows, cols},torch::kDouble);
    //torchMat = torch::from_blob(data.data(), {rows, cols}).toType(torch::kFloat);
    torch::Device device = torch::kCUDA;
    torchMat = torchMat.to(device);

    return;
}

void gpu2cpu(torch::Tensor &torchMat)
{
    torch::Device device = torch::kCPU;
    torchMat = torchMat.to(device);
    return;
}
void cpu2gpu(torch::Tensor &torchMat)
{
    torch::Device device = torch::kCUDA;
    torchMat = torchMat.to(device);
    return;
}

void tensor2file(torch::Tensor &torchMat,string filename)
{
    if(torchMat.device().is_cuda())
    {
        gpu2cpu(torchMat);
    }
    std::ofstream file(filename);
    if(!file.is_open())
    {
        std::cerr << "Failed to open the file for writing." << std::endl;
        return;
    }
    file << torchMat << endl;
    file.close();

}

double likehood(const std::vector<double> &x, std::vector<double> &grad, void *data)
{
    //myData *md = (myData *)data;
/*
    double sigma_o = x[0];
    double r_l = x[1];
    double sigma_b = x[2];
    double b_l = x[3];
*/
    cout << "x = ";
    for(auto i : x)
    {
        cout << i << " ";
    }
    cout << endl;
    //MatrixXdr H = md->H;
    //MatrixXdr y = md->y;
    //MatrixXdr x_prior = md->x_prior;
    //MatrixXdr shape = md->shape;

    //torch::TensorOptions options(torch::kCUDA);
    //CovfunBase *cfb_r = new gamma_exp_Covfun(x,obs_num,true); 
    //CovfunBase *cfb_b = new gamma_exp_Covfun(x,grids_num,false); 
    CovfunBase *cfb_r = new R_COV(x,obs_num,true); 
    CovfunBase *cfb_b = new B_COV(x,grids_num,false); 
    
    MatrixXdr &R_theta = *(cfb_r->gen_covpd_fun(0));
    MatrixXdr &B_theta = *(cfb_b->gen_covpd_fun(0));
    //MatrixXdr R_theta = MatrixXdr::Zero(obs_num,obs_num);
    //gen_cov_matrix(R_theta,sigma_o,r_l,exponential_cf,true,false);

    //MatrixXdr B_theta = MatrixXdr::Zero(grids_num,grids_num);
    //gen_cov_matrix(B_theta,sigma_b,b_l,exponential_cf,false,false);

    //MatrixXdr R_o = MatrixXdr::Zero(obs_num,obs_num);
    //gen_cov_matrix(R_o,sigma_o,r_l,exponential_cf_sigma,true,true);

    //MatrixXdr R_l = MatrixXdr::Zero(obs_num,obs_num);
    //gen_cov_matrix(R_l,sigma_o,r_l,exponential_cf_l,true,false);

    //MatrixXdr B_b = MatrixXdr::Zero(grids_num,grids_num);
    //gen_cov_matrix(B_b,sigma_b,b_l,exponential_cf_sigma,false,true);

    //MatrixXdr B_l = MatrixXdr::Zero(grids_num,grids_num);
    //gen_cov_matrix(B_l,sigma_b,b_l,exponential_cf_l,false,false);

    torch::Tensor _R_theta = torch::zeros({R_theta.rows(),R_theta.cols()}, torch::kDouble);
    eigen2torch(R_theta, _R_theta);

    torch::Tensor _B_theta = torch::zeros({B_theta.rows(),B_theta.cols()},torch::kDouble);
    eigen2torch(B_theta, _B_theta);
    /*
    torch::Tensor _R_o = torch::zeros({R_o.rows(),R_o.cols()}, torch::kDouble);
    eigen2torch(R_o, _R_o);
    torch::Tensor _R_l = torch::zeros({R_l.rows(),R_l.cols()}, torch::kDouble);
    eigen2torch(R_l, _R_l);
    torch::Tensor _B_b = torch::zeros({B_b.rows(),B_b.cols()}, torch::kDouble);
    eigen2torch(B_b, _B_b);
    torch::Tensor _B_l = torch::zeros({B_l.rows(),B_l.cols()}, torch::kDouble);
    eigen2torch(B_l, _B_l);
    */
    torch::Tensor _H = torch::zeros({H.rows(),H.cols()},torch::kDouble);
    eigen2torch(H, _H);

    torch::Tensor _y = torch::zeros({y.rows(),y.cols()}, torch::kDouble);
    eigen2torch(y, _y);

    torch::Tensor _x_prior = torch::zeros({x_prior.rows(),x_prior.cols()}, torch::kDouble);
    eigen2torch(x_prior, _x_prior);

    //clock_t start = clock();
    torch::Tensor _tem1 = torch::mm(_H , _B_theta);
    torch::Tensor _tem2 = torch::mm(_tem1, _H.t());
    torch::Tensor _D_theta = _R_theta + _tem2;
    //clock_t end = clock();
    //cout << "乘法花费了" << (double)(end - start) / CLOCKS_PER_SEC << "秒" << endl;
    //start = clock();
    torch::Tensor _D_inv = torch::inverse(_D_theta);
    //end = clock();
    //cout << "求逆花费了" << (double)(end - start) / CLOCKS_PER_SEC << "秒" << endl;
    torch::Tensor _alpha = torch::mm(_D_inv , _y);
    torch::Tensor _term1 = _D_inv - torch::mm(_alpha , _alpha.t());

    if (grad.size()>0) {
    cout << "test1" << endl;
    for (int i = 0; i < p1_num; ++i) {
        torch::Tensor _R = torch::zeros({R_theta.rows(),R_theta.cols()}, torch::kDouble);
        eigen2torch(*(cfb_r->gen_covpd_fun(i)), _R);
        grad[i] = torch::trace(_term1 * _R).item().to<double>() * 0.5;
    }
    for (int i = 0; i < p2_num; ++i) {
        int index = i + p1_num;
        torch::Tensor _B = torch::zeros({B_theta.rows(),B_theta.cols()}, torch::kDouble);
        eigen2torch(*(cfb_b->gen_covpd_fun(index)), _B);
        grad[index] = torch::trace(torch::mm(torch::mm(torch::mm(_term1 , _H) , _B) , _H.t())).item().to<double>() * 0.5;
    }
    /*
        grad[0] = torch::trace(_term1 * _R_o).item().to<double>() * 0.5;
    cout << "12" << endl;
        grad[1] = torch::trace(_term1 * _R_l).item().to<double>() * 0.5;
    cout << "13" << endl;
        grad[2] = torch::trace(torch::mm(torch::mm(torch::mm(_term1 , _H) , _B_b) , _H.t())).item().to<double>() * 0.5;
    cout << "14" << endl;
        grad[3] = torch::trace(torch::mm(torch::mm(torch::mm(_term1 , _H) , _B_l) , _H.t())).item().to<double>() * 0.5;
        */
    }	

    torch::Tensor _y_Hxp = _y - torch::mm(_H , _x_prior);

    torch::Tensor _ret1 = torch::logdet(_D_theta);
    torch::Tensor _ret2 = torch::mm(torch::mm(_y_Hxp.t() , _D_inv) , _y_Hxp);
    double ret = _ret1.item().to<double>() + _ret2.item().to<double>();
    cout << "ret1 = " << _ret1.item().to<double>() << endl;
    cout << "ret2 = " << _ret2.item().to<double>() << endl;
    cout << "ret = " << ret << endl;
    counts++;
    delete cfb_r;
    delete cfb_b;
    return ret;
    
}


int main(int argc, char *argv[])
{
    string pre = "/home/akagi/InverseCO2/data/hhsd/";
    y = readMatrix(pre + "y.txt");
    H = readMatrix(pre + "H.txt");
    H_v = readMatrix(pre + "H_v.txt");
    obs_v = readMatrix(pre + "obs_v.txt");
    x_prior = readMatrix(pre + "x_prior.txt");
    flux = readMatrix(pre + "flux.txt");
    MatrixXdr shape = readMatrix(pre + "shape.txt");
    lon = shape(0,0);
    lat = shape(1,0);
    cout << "read data from file" << endl;
    //myData md = {y,H,x_prior,shape};
    //myData *md_p = &md; 
    vector<double> lb(4, 0 ); //lower bounds
    vector<double> ub={50.0,20.0, 50.0,20.0}; //upper bounds for gamma-exp
   // vector<double> ub={20.0, 40.0, 20.0, 40.0, 2.0}; //upper bounds for rational-quad
    vector<double> x={20.0,10.0,20.0,10.0};
    double minf;
    int p_num = lb.size();
    p1_num = 2;//paras num of R
    p2_num = p_num - p1_num;//paras num of R
    obs_num = y.rows();
    grids_num = x_prior.rows();
    // create the optimization problem
    // opaque pointer type
    nlopt::opt opt(nlopt::GN_ISRES, p_num);
    //opt = nlopt_create(NLOPT_GD_STOGO, 4);
     
    opt.set_lower_bounds(lb);
    opt.set_upper_bounds(ub);
     
    opt.set_min_objective(likehood,NULL);
     
    //nlopt_set_xtol_rel(opt, 1e-4);
    int mins = 20;
    opt.set_maxtime(60*mins);
     
    // initial guess
     
    nlopt::result res=opt.optimize(x,minf);
     
     
    if (res < 0) {
        cout << "nlopt failed!" << endl;
    }
    else {
        cout << "found minimum at f(" ;
        for(auto i:x)
        {
            cout << i << " ";
        }
        cout << ") = " << minf << endl;
        cout << "optim times :" << counts << endl;
    }
    
    CovfunBase *cfb_r = new R_COV(x,obs_num,true); 
    CovfunBase *cfb_b = new B_COV(x,grids_num,false); 
    
    MatrixXdr &R_theta = *(cfb_r->gen_covpd_fun(0));
    MatrixXdr &B_theta = *(cfb_b->gen_covpd_fun(0));
    torch::Tensor _R_theta = torch::zeros({R_theta.rows(),R_theta.cols()}, torch::kDouble);
    eigen2torch(R_theta, _R_theta);
/*
    gpu2cpu(_R_theta);
    cout << _R_theta << endl;
    cpu2gpu(_R_theta);
*/
    torch::Tensor _B_theta = torch::zeros({B_theta.rows(),B_theta.cols()},torch::kDouble);
    eigen2torch(B_theta, _B_theta);
    
    torch::Tensor _y = torch::zeros({y.rows(),y.cols()},torch::kDouble);
    torch::Tensor _H = torch::zeros({H.rows(),H.cols()},torch::kDouble);
    torch::Tensor _H_v = torch::zeros({H_v.rows(),H_v.cols()},torch::kDouble);
    torch::Tensor _obs_v = torch::zeros({obs_v.rows(),obs_v.cols()},torch::kDouble);
    torch::Tensor _x_prior = torch::zeros({x_prior.rows(),x_prior.cols()},torch::kDouble);
    torch::Tensor _flux = torch::zeros({flux.rows(),flux.cols()},torch::kDouble);
    eigen2torch(y, _y);
    eigen2torch(H, _H);
    eigen2torch(H_v, _H_v);
    eigen2torch(obs_v, _obs_v);
    eigen2torch(x_prior, _x_prior);
    eigen2torch(flux, _flux);

    torch::Tensor _tem1 = torch::mm(_H , _B_theta);
    torch::Tensor _tem2 = torch::mm(_tem1, _H.t());
    torch::Tensor _D_theta = _R_theta + _tem2;
    torch::Tensor _D_inv = torch::inverse(_D_theta);

    torch::Tensor _y_Hxp = _y - torch::mm(_H , _x_prior);
    torch::Tensor _x_posterior= _x_prior + torch::mm(torch::mm(torch::mm(_B_theta , _H.t()) , _D_inv) , _y_Hxp);
    double posterior = torch::sum(_x_posterior).item().to<double>();

    
    torch::Tensor _diff = (_x_posterior - _flux).view({-1});

    double sse = torch::dot(_diff,_diff).item().to<double>();
    double rmse = sqrt(sse/grids_num);
/*
    torch::Tensor _y_hat = torch::mm(_H , _x_posterior);
    torch::Tensor _y_real = torch::mm(_H , _flux);
    torch::Tensor _y_diff = (_y_real - _y_hat).view({-1});

    double sse = torch::dot(_y_diff,_y_diff).item().to<double>();
    double rmse = sqrt(sse/obs_num);

    torch::Tensor _y_bar = torch::ones_like(_y_real) * torch::mean(_y_real).item().to<double>();
    torch::Tensor st = (_y_real - _y_bar).view({-1});
    double sst = torch::dot(st,st).item().to<double>();

    double r2 = 1-sse/sst;
*/
    torch::Tensor _verify_hat = torch::mm(_H_v,_x_posterior);
    torch::Tensor _verify_prior = torch::mm(_H_v,_x_prior);
    torch::Tensor _verify = _obs_v;
    torch::Tensor _v_diff = (_verify_hat - _verify).view({-1});

    double sse_v = torch::dot(_v_diff,_v_diff).item().to<double>();
    double rmse_v = sqrt(sse_v/OBS);
    string r_str = STR(R_COV);
    string b_str = STR(B_COV);
    string folder = r_str + "-" + b_str + "-" + std::to_string(lon) + "_" + std::to_string(lat) + "-" +std::to_string(obs_num);
    string pre2 = "/home/akagi/InverseCO2/build/" + folder;
    if(!folderExists(pre2.c_str()))
    {
        createFolder(pre2.c_str());
    }
    pre2 = pre2 + "/";
    for (int i = 1; i < 10; ++i) {
        if(!folderExists((pre2+std::to_string(i)).c_str()))
        {
            pre2 = pre2+std::to_string(i);
            createFolder(pre2.c_str());
            pre2 = pre2 + "/";
            break;
        }
    }
    tensor2file(_verify_hat,pre2+"_verify_hat.txt");
    tensor2file(_verify_prior,pre2+"_verify_prior.txt");
    tensor2file(_verify,pre2+"_verify.txt");
    tensor2file(_x_posterior,pre2+"_x_posterior.txt");
    tensor2file(_x_prior,pre2+"_x_prior.txt");
    tensor2file(_flux,pre2+"_flux.txt");
    writeMatrix(shape,(pre2+"shape.txt").c_str());
    

    cout << "posterior sum = " << posterior << endl;
    cout << "prior sum = " << x_prior.sum() << endl;
    cout << "true flux sum = " << flux.sum() << endl;
    cout << "grids_rmse = " << rmse << endl;
    cout << "rmse_v = " << rmse_v << endl;
    std::ofstream file(pre2+"result.txt");
    if(!file.is_open())
    {
        std::cerr << "Failed to open the file for writing." << std::endl;
        return 0;
    }
        file << "found minimum at f(" ;
        for(auto i:x)
        {
            file << i << " ";
        }
        file << ") = " << minf << endl;
        file << "optim times :" << counts << endl;
    file << "posterior sum = " << posterior << endl;
    file << "prior sum = " << x_prior.sum() << endl;
    file << "true flux sum = " << flux.sum() << endl;
    file << "grids_rmse = " << rmse << endl;
    file << "rmse_v = " << rmse_v << endl;
    file.close();
/*
*/

    delete cfb_r;
    delete cfb_b;
        return 0;
}
