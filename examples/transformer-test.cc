#include <iostream>
#include <vector>
#include <thread>
#include <cmath>

#include <iostream>

#include "external/cnpy/cnpy.h"
#include "external/thread-pool-11/ThreadPool.h"

#include "examples/transformer.hh"
#include "main/unittest.hh"

void print(const char* name, const Tensor& tensor)
{
  std::cout << name
  << " [" << tensor.rows() << " x " << tensor.cols() << "]"
  << std::endl;
  std::cout << tensor << std::endl;
}

void print(const char* name, const SparseTensor& tensor)
{
  std::cout << name
  << " [" << tensor.rows() << " x " << tensor.cols() << "]"
  << std::endl;
  std::cout << tensor << std::endl;
}

void print(const char* name, Function& f)
{
  print(name, f.forward());
}

void compute_result(
  const std::vector<double>& input,
  std::vector<double>& result,
  int index) {
  for (int n=0; n<1000; n++)
  {
    for (int i=0; i<input.size(); i++)
    {
        result[index * input.size() + i] = std::sqrt(input[i]);
    }
  }
}

// Function to perform parallel matrix multiplication
void parallel_compute_result(
  const std::vector<double>& input,
  std::vector<double>& result,
  int num_threads) {
    int num_blocks = result.size() / input.size();

    // Launch threads
    while (num_blocks)
    {
      // Vector to store thread objects
      std::vector<std::thread> threads;

      std::cout << "blocks to do " << num_blocks << std::endl;

      for (int i = 0; i < num_threads && num_blocks; ++i) {
          threads.emplace_back(compute_result, input, std::ref(result), num_blocks-1);
          num_blocks--;
      }

      std::cout << "thread running " << threads.size() << std::endl;

      // Wait for threads to finish
      for (auto& thread : threads) {
          thread.join();
      }
    }
}

void test_threads() {    
    int num_threads = 4;
    std::cout << "threads=" << num_threads << std::endl;

    // Initialize input matrix with consecutive numbers
    std::vector<double> input(1000000, 0);
    for (int i=0; i<input.size(); i++) input[i] = i;
    std::cout << "input=" << input.size() << std::endl;

    int num_blocks = 8;
    std::cout << "blocks=" << num_blocks << std::endl;

    // Initialize result matrix with zeros
    std::vector<double> result(num_blocks * input.size(), 0);
    std::cout << "result=" << result.size() << std::endl;

    // Perform parallel matrix multiplication
    parallel_compute_result(input, result, num_threads);

    // Display the result
    std::cout << "Result" << std::endl;
    for (int i=0; i<10; i++)
    {
        std::cout << "input=" << input[i] << ", result=" << result[i] << std::endl;
    }
}

void test_cnpy()
{
  // load the entire npz file
  cnpy::npz_t my_npz = cnpy::npz_load("transformer.npz");

  for (const auto& item : my_npz)
  {
    auto& name = item.first;
    auto& arr = item.second;

    std::cout << "name:" << name << std::endl;
    std::cout << "word " << arr.word_size << std::endl;
    std::cout << "shape " << arr.shape.size() << std::endl;
    for (auto s: arr.shape)
    {
      std::cout << "dimension:" << s << std::endl;
    }

    const float* data = arr.data<float>();
    for (int i=0; i<4; i++)
    {
      if (arr.shape.size() == 1)
        std::cout << "data:" << *(data+i) << std::endl;
      else
      if (arr.shape.size() == 2)
      {
        int rows = arr.shape[0];
        int cols = arr.shape[1];
        std::cout << "row 2 data:" << *(data+cols+i) << std::endl;
      }
    }
  }
}

void test_thread_pool() {
    int num_threads = 2;
    ThreadPool pool(num_threads);
    std::cout << "thread pool=" << num_threads << std::endl;
    std::vector< std::future<double> > results;

    for(int i = 0; i < 8; ++i) {
        results.emplace_back(
            pool.enqueue([i] {
                double ret = 1;
                for (int k=0; k<1000000000; k++) {
                  ret = log(ret + k);
                }
                return ret + i;
            })
        );
    }

    for(auto && result: results)
        std::cout << result.get() << ' ';
    std::cout << std::endl;
}

void test_scaled_dot_product_attention_forward()
{
    TEST_BEGIN("Scaled Dot-Product Attention Forward")

    /*
    Q = torch.Tensor([
        [1,2,3],
        [4,5,6],
    ])
    K = torch.Tensor([
        [0.1,0.2,0.3],
        [0.4,0.5,0.6],
        [1.4,1.5,1.6],
        [2.4,2.5,2.6],
    ])
    V = torch.Tensor([
        [-2,7,8],
        [4,1,-9],
        [1,2,3],
        [4,5,6],
    ])
    M = torch.tensor([
        [1,1,1,1],
        [0,0,0,0],
    ], dtype=torch.bool)

    // attention
    A = torch.Tensor([
        [ 3.9070,  4.9059,  5.8955,  4.9668,  4.9668],
        [ 3.5844,  1.4156, -7.8225,  2.9307,  2.9307],
    ])
    */

    Graph g;
    DTYPE dropout = 0.0;

    // [2x3]
    auto Q = g.new_constant(2, 3);
    Q->value() << 1,2,3,
                  4,5,6;
    //print("Q", *Q);

    // [4x3]
    auto K = g.new_constant(4, 3);
    K->value() << 0.1,0.2,0.3,
                  0.4,0.5,0.6,
                  1.4,1.5,1.6,
                  2.4,2.5,2.6;
    //print("K", *K);

    // QK_T -> [2x4]

    // [2x4]
    auto M = g.new_constant(2, 4);
    M->value() << 1,1,1,1,
                  1,1,0,0;
    //print("M", *M);

    // [4x5]
    auto V = g.new_constant(4, 5);
    V->value() << -2,7,8,2,2,
                  4,1,-9,3,3,
                  1,2,3,4,4,
                  4,5,6,5,5;
    //print("V", *V);

    // QK_TV -> [2x5]

    int T = Q->value().rows();
    int S = K->value().rows();
    int D = K->value().cols();

    ScaledDotProductAttention attn(g, *Q,*K,*V, M, T, S, D, dropout);

    auto& A = attn();
    //print("A", A);


    Tensor A_torch(2,5);
    A_torch <<  3.9070,  4.9059,  5.8955,  4.9668,  4.9668,
                3.5844,  1.4156, -7.8225,  2.9307,  2.9307;

    ASSERT(A_torch.isApprox(A, 0.00001))

    TEST_END()
}

void test_scaled_dot_product_attention_backward()
{
    TEST_BEGIN("Scaled Dot-Product Attention Backward")

    Graph g;
    DTYPE dropout = 0.0;

    // [2x3]
    auto Q = g.new_variable(2, 3);
    Q->value() << 1,2,3,
                  4,5,6;

    // [4x3]
    auto K = g.new_variable(4, 3);
    K->value() << 0.1,0.2,0.3,
                  0.4,0.5,0.6,
                  1.4,1.5,1.6,
                  2.4,2.5,2.6;

    // QK_T -> [2x4]
    auto M = g.new_constant(2, 4);
    M->value() << 1,1,1,1,
                  1,1,0,0;

    // [4x5]
    auto V = g.new_variable(4, 5);
    V->value() << -2,7,8,2,2,
                  4,1,-9,3,3,
                  1,2,3,4,4,
                  4,5,6,5,5;

    // QK_TV -> [2x5]

    int T = Q->value().rows();
    int S = K->value().rows();
    int D = K->value().cols();

    auto attnptr = new ScaledDotProductAttention(
      g, *Q,*K,*V, M, T, S, D, dropout
    );
    g.keep(attnptr); // for auto-grad
    auto& attn = *attnptr;

    attn.gradient() = Tensor::Ones(attn().rows(),attn().cols());
    attn.gradient().block(0,0, attn.forward().rows(), 1) = 
      5 * Tensor::Ones(attn().rows(), 1);

    Tensor dQ = Q->backward();
    Tensor dK = K->backward();
    Tensor dV = V->backward();

    auto dQ_num = g.dFdX(attn, *Q);
    auto dK_num = g.dFdX(attn, *K);
    auto dV_num = g.dFdX(attn, *V);

    ASSERT(dQ_num.isApprox(dQ, 0.01))
    ASSERT(dK_num.isApprox(dK, 0.01))
    ASSERT(dV_num.isApprox(dV, 0.01))

    Tensor dQ_torch(2,3);
    dQ_torch << 0.4281, 0.4281, 0.4281,
                0.1005, 0.1005, 0.1005;

    Tensor dK_torch(4,3);
    dK_torch << -1.3459, -1.6870, -2.0280,
                1.3277,  1.6505,  1.9732,
                -0.3897, -0.7795, -1.1692,
                0.4080,  0.8160,  1.2240;

    Tensor dV_torch(4,5);
    dV_torch << 0.3480, 0.0696, 0.0696, 0.0696, 0.0696,
                4.6584, 0.9317, 0.9317, 0.9317, 0.9317,
                0.1516, 0.0303, 0.0303, 0.0303, 0.0303,
                4.8420, 0.9684, 0.9684, 0.9684, 0.9684;

    ASSERT(dQ_torch.isApprox(dQ, 0.0001))
    ASSERT(dK_torch.isApprox(dK, 0.0001))
    ASSERT(dV_torch.isApprox(dV, 0.0001))

    TEST_END()
}

void test_multihead_attention_forward()
{
    TEST_BEGIN("Multi-Head Attention Forward")
        
    Graph g;

    int TRG_SIZE = 3;
    int seq_size = 3;
    int EMB_SIZE = 4;
    int num_heads = 2;
    bool bias = true;
    DTYPE dropout = 0.0;

    auto q = g.new_variable(TRG_SIZE, EMB_SIZE);
    auto k = g.new_variable(seq_size, EMB_SIZE);
    auto v = g.new_variable(TRG_SIZE, EMB_SIZE);

    MultiHeadAttention mha(
      g, *q, *k, *v,
      TRG_SIZE, seq_size, EMB_SIZE, num_heads,
      bias, dropout
    );

    q->value() <<
      0.0878, 0.0416, 0.6166, 0.1477,
      0.9752, 0.8866, 0.5407, 0.1911,
      0.5300, 0.2800, 0.5306, 0.4950;   
    k->value() <<
      0.2248, 0.4832, 0.5916, 0.0345,
      0.4916, 0.0881, 0.3768, 0.3048,
      0.0780, 0.3594, 0.0297, 0.6474;
    v->value() <<
      0.2014, 0.0033, 0.2326, 0.5677,
      0.6842, 0.1161, 0.8033, 0.6450,
      0.4097, 0.3034, 0.8000, 0.7103;

    mha.Wq().value() <<
      0.4271,  0.3013, -0.4279, -0.2122,
      0.2983,  0.3350, -0.4619,  0.5432,
      -0.1488,  0.1778, -0.4288, -0.5003,
      0.1173,  0.3713, -0.2347, -0.2251;   
    mha.Wk().value() <<
      0.1557,  0.4673,  0.0920,  0.3889,
      0.5867,  0.0088,  0.4371,  0.0371,
      0.4897, -0.0109, -0.0646,  0.5190,
      -0.5768,  0.1376, -0.5507,  0.5315;
    mha.Wv().value() <<
      -0.3599, -0.4841,  0.0526, -0.5235,
      -0.1576,  0.4844, -0.3817,  0.2549,
      -0.1432,  0.5141, -0.5741, -0.0179,
      -0.0103, -0.4235, -0.5195, -0.1589;
    mha.Wo().value() <<
      -0.2588,  0.4873,  0.0642,  0.4206,
      0.3272,  0.3202,  0.4458, -0.3825,
      -0.4631, -0.2740, -0.2628, -0.4749,
      -0.3654,  0.4841,  0.4618, -0.1188;

    if (bias)
    {
      mha.bq().value() <<
        0.4755, 0.1042, 0.6459, 0.2230;      
      mha.bk().value() <<
        0.0739, 0.6705, 0.8532, 0.7830;
      mha.bv().value() <<
        0.1097, 0.8451, 0.7208, 0.2440;
      mha.bo().value() <<
        0.0307, 0.1667, 0.4442, 0.1971;
    }

    Tensor mha_hat(TRG_SIZE, EMB_SIZE);
    mha_hat <<   
        0.4363, 0.5356, 0.4469, 0.9232,
        0.4293, 0.5403, 0.4531, 0.9223,
        0.4353, 0.5364, 0.4456, 0.9224;

    ASSERT(mha().isApprox(mha_hat, 0.0001))
    ASSERT(mha().rows() == TRG_SIZE)
    ASSERT(mha().cols() == EMB_SIZE)

    TEST_END()
}

void test_multihead_attention_backward()
{
    TEST_BEGIN("Multi-Head Attention Backward")

    Graph g;

    int TRG_SIZE = 3;
    int seq_size = 3;
    int EMB_SIZE = 4;
    int num_heads = 2;
    bool bias = true;
    DTYPE dropout = 0.0;

    auto q = g.new_variable(TRG_SIZE, EMB_SIZE);
    auto k = g.new_variable(seq_size, EMB_SIZE);
    auto v = g.new_variable(TRG_SIZE, EMB_SIZE);

    auto mhaptr = new MultiHeadAttention(
      g, *q, *k, *v,
      TRG_SIZE, seq_size, EMB_SIZE, num_heads,
      bias, dropout
    );
    g.keep(mhaptr); // for auto-grad
    auto& mha = *mhaptr;

    q->value() <<
      0.0878, 0.0416, 0.6166, 0.1477,
      0.9752, 0.8866, 0.5407, 0.1911,
      0.5300, 0.2800, 0.5306, 0.4950;
    k->value() <<
      0.2248, 0.4832, 0.5916, 0.0345,
      0.4916, 0.0881, 0.3768, 0.3048,
      0.0780, 0.3594, 0.0297, 0.6474;
    v->value() <<
      0.2014, 0.0033, 0.2326, 0.5677,
      0.6842, 0.1161, 0.8033, 0.6450,
      0.4097, 0.3034, 0.8000, 0.7103;

    mha.Wq().value() <<
      0.4271,  0.3013, -0.4279, -0.2122,
      0.2983,  0.3350, -0.4619,  0.5432,
      -0.1488,  0.1778, -0.4288, -0.5003,
      0.1173,  0.3713, -0.2347, -0.2251;
    mha.Wk().value() <<
      0.1557,  0.4673,  0.0920,  0.3889,
      0.5867,  0.0088,  0.4371,  0.0371,
      0.4897, -0.0109, -0.0646,  0.5190,
      -0.5768,  0.1376, -0.5507,  0.5315;
    mha.Wv().value() <<
      -0.3599, -0.4841,  0.0526, -0.5235,
      -0.1576,  0.4844, -0.3817,  0.2549,
      -0.1432,  0.5141, -0.5741, -0.0179,
      -0.0103, -0.4235, -0.5195, -0.1589;
    mha.Wo().value() <<
      -0.2588,  0.4873,  0.0642,  0.4206,
      0.3272,  0.3202,  0.4458, -0.3825,
      -0.4631, -0.2740, -0.2628, -0.4749,
      -0.3654,  0.4841,  0.4618, -0.1188;

    if (bias)
    {
      mha.bq().value() <<
        0.4755, 0.1042, 0.6459, 0.2230;
      mha.bk().value() <<
        0.0739, 0.6705, 0.8532, 0.7830;
      mha.bv().value() <<
        0.1097, 0.8451, 0.7208, 0.2440; 
      mha.bo().value() <<
        0.0307, 0.1667, 0.4442, 0.1971;
    }

    auto& F = mha();

    mha.gradient() = Tensor::Ones(TRG_SIZE, EMB_SIZE);
    mha.gradient()(0) = 5;

    const Tensor dFdq = q->backward();
    const Tensor dFdk = k->backward();
    const Tensor dFdv = v->backward();

    Tensor dFdq_hat(TRG_SIZE, EMB_SIZE);
    dFdq_hat <<
      -0.0026, -0.0249,  0.0291,  0.0100,
      0.0004,  0.0037, -0.0015, -0.0093,
      0.0003,  0.0034, -0.0013, -0.0091;

    Tensor dFdk_hat(TRG_SIZE, EMB_SIZE);
    dFdk_hat <<
      0.0074, -0.0047, -0.0078,  0.0137,
     -0.0168, -0.0184,  0.0007, -0.0346,
      0.0094,  0.0231,  0.0071,  0.0209;

    Tensor dFdv_hat(TRG_SIZE, EMB_SIZE);
    dFdv_hat <<
      0.0305,  1.7188, -1.1291,  0.9921,
      0.0227,  1.7295, -1.1511,  0.9818,
      0.0102,  1.7986, -1.1801,  0.9916;

    ASSERT(dFdq.isApprox(dFdq_hat, 0.01));
    ASSERT(dFdk.isApprox(dFdk_hat, 0.01));
    ASSERT(dFdv.isApprox(dFdv_hat, 0.01));

    auto dFdq_num = g.dFdX(mha, *q);
    auto dFdk_num = g.dFdX(mha, *k);
    auto dFdv_num = g.dFdX(mha, *v);

    ASSERT(dFdq.isApprox(dFdq_num, 0.01));
    ASSERT(dFdk.isApprox(dFdk_num, 0.01));
    ASSERT(dFdv.isApprox(dFdv_num, 0.01));

    TEST_END()
}


void test_position_wise_ff_forward()
{
    TEST_BEGIN("Position-Wise FF Forward")
        
    Graph g;

    int EMB_SIZE = 4;
    int HID_SIZE = 3;
    int TRG_SIZE = 2;
    DTYPE dropout = 0.0;

    auto& x = *g.new_variable(TRG_SIZE, EMB_SIZE);
    ASSERT(x.value().rows() == TRG_SIZE);
    ASSERT(x.value().cols() == EMB_SIZE);

    x.value() <<
      0.0878, 0.0416, 0.6166, 0.1477,
      0.5300, 0.2800, 0.5306, 0.4950;

    PositionwiseFeedForward ff(
      g, x, EMB_SIZE, HID_SIZE, dropout
    );
    
    ASSERT(ff.L1().W().value().rows() == HID_SIZE);
    ASSERT(ff.L1().W().value().cols() == EMB_SIZE);
    ASSERT(ff.L1().b().value().rows() == 1);
    ASSERT(ff.L1().b().value().cols() == HID_SIZE);

    ASSERT(ff.L2().W().value().rows() == EMB_SIZE);
    ASSERT(ff.L2().W().value().cols() == HID_SIZE);
    ASSERT(ff.L2().b().value().rows() == 1);
    ASSERT(ff.L2().b().value().cols() == EMB_SIZE);

    ff.L1().W().value() <<
      -0.3883,  0.2742, -0.4652, -0.1417,
      -0.0996, -0.4170, -0.0302,  0.1254,
      -0.2065,  0.0107,  0.3998,  0.3775;
    ff.L2().W().value() <<
       0.0348,  0.3779, -0.5751,
      -0.0708, -0.4522, -0.4000,
       0.3196,  0.2163,  0.5397,
      -0.1805,  0.0472, -0.4630;

    ff.L1().b().value() <<
      0.4282,  0.2099, -0.2209;
    ff.L2().b().value() <<
      -0.4660, -0.4707,  0.4046, -0.4392;

    Tensor ff_hat(TRG_SIZE, EMB_SIZE);
    ff_hat <<   
      -0.4298, -0.5862,  0.5099, -0.4777,
      -0.4746, -0.5384,  0.4620, -0.4683;

    ASSERT(ff().isApprox(ff_hat, 0.0001))
    ASSERT(ff().rows() == TRG_SIZE)
    ASSERT(ff().cols() == EMB_SIZE)

    TEST_END()
}


void test_position_wise_ff_backward()
{
    TEST_BEGIN("Position-Wise FF Backward")
    
    Graph g;

    int EMB_SIZE = 4;
    int HID_SIZE = 3;
    int TRG_SIZE = 2;
    DTYPE dropout = 0.0;

    auto& x = *g.new_variable(TRG_SIZE, EMB_SIZE);
    ASSERT(x.value().rows() == TRG_SIZE);
    ASSERT(x.value().cols() == EMB_SIZE);

    x.value() <<
      0.0878, 0.0416, 0.6166, 0.1477,
      0.5300, 0.2800, 0.5306, 0.4950;

    auto& ff = *(new PositionwiseFeedForward(
      g, x, EMB_SIZE, HID_SIZE, dropout
    ));
    g.keep(&ff);    
    
    ASSERT(ff.L1().W().value().rows() == HID_SIZE);
    ASSERT(ff.L1().W().value().cols() == EMB_SIZE);
    ASSERT(ff.L1().b().value().rows() == 1);
    ASSERT(ff.L1().b().value().cols() == HID_SIZE);

    ASSERT(ff.L2().W().value().rows() == EMB_SIZE);
    ASSERT(ff.L2().W().value().cols() == HID_SIZE);
    ASSERT(ff.L2().b().value().rows() == 1);
    ASSERT(ff.L2().b().value().cols() == EMB_SIZE);

    ff.L1().W().value() <<
      -0.3883,  0.2742, -0.4652, -0.1417,
      -0.0996, -0.4170, -0.0302,  0.1254,
      -0.2065,  0.0107,  0.3998,  0.3775;
    ff.L2().W().value() <<
       0.0348,  0.3779, -0.5751,
      -0.0708, -0.4522, -0.4000,
       0.3196,  0.2163,  0.5397,
      -0.1805,  0.0472, -0.4630;

    ff.L1().b().value() <<
      0.4282,  0.2099, -0.2209;
    ff.L2().b().value() <<
      -0.4660, -0.4707,  0.4046, -0.4392;

    Tensor ff_hat(TRG_SIZE, EMB_SIZE);
    ff_hat <<   
      -0.4298, -0.5862,  0.5099, -0.4777,
      -0.4746, -0.5384,  0.4620, -0.4683;
      
    ASSERT(ff().isApprox(ff_hat, 0.0001))
    ASSERT(ff().rows() == TRG_SIZE)
    ASSERT(ff().cols() == EMB_SIZE)

    Tensor dF(TRG_SIZE, EMB_SIZE);
    dF <<
      5., 1., 1., 1.,
      1., 1., 1., 1.;      
    ff.gradient() = dF;
    
    Tensor dFdW1 = ff.L1().W().backward();
    Tensor dFdW2 = ff.L2().W().backward();
    Tensor dFdb1 = ff.L1().b().backward();
    Tensor dFdb2 = ff.L2().b().backward();
    Tensor dFdx = x.backward();

    Tensor dFdW1_num = g.dFdX(ff, ff.L1().W());
    Tensor dFdW2_num = g.dFdX(ff, ff.L2().W());
    Tensor dFdb1_num = g.dFdX(ff, ff.L1().b());
    Tensor dFdb2_num = g.dFdX(ff, ff.L2().b());
    Tensor dFdx_num = g.dFdX(ff, x);
        
    Tensor dFdW1_hat(HID_SIZE, EMB_SIZE);
    dFdW1_hat <<
      0.0213,  0.0101,  0.1494,  0.0358,
      0.2496,  0.1237,  1.1491,  0.3449,
      -0.7570, -0.3846, -2.4491, -0.9172;    

    Tensor dFdb1_hat(1, HID_SIZE);
    dFdb1_hat <<
      0.2423,  1.8900, -4.0972;

    Tensor dFdW2_hat(EMB_SIZE, HID_SIZE);
    dFdW2_hat <<
      0.4887, 1.0049, 0.3901,
      0.0977, 0.2701, 0.1353,
      0.0977, 0.2701, 0.1353,
      0.0977, 0.2701, 0.1353;
         
    Tensor dFdb2_hat(1, EMB_SIZE);
    dFdb2_hat <<
      6., 2., 2., 2.;

    Tensor dFdx_hat(TRG_SIZE, EMB_SIZE);
    dFdx_hat <<
      0.3971, -0.6770, -1.4430, -1.0286,
      0.1667, -0.0885, -0.3649, -0.3154;

    ASSERT(dFdW1.isApprox(dFdW1_hat, 0.0001))
    ASSERT(dFdW2.isApprox(dFdW2_hat, 0.0001))
    ASSERT(dFdb1.isApprox(dFdb1_hat, 0.0001))
    ASSERT(dFdb2.isApprox(dFdb2_hat, 0.0001))
    ASSERT(dFdx.isApprox(dFdx_hat, 0.0001))

    ASSERT(dFdW1.isApprox(dFdW1_num, 0.0001))
    ASSERT(dFdW2.isApprox(dFdW2_num, 0.0001))
    ASSERT(dFdb1.isApprox(dFdb1_num, 0.0001))
    ASSERT(dFdb2.isApprox(dFdb2_num, 0.0001))
    ASSERT(dFdx.isApprox(dFdx_num, 0.0001))
              
    TEST_END()
}


int main(int argc, char* argv[]) {

    //test_cnpy();
    //test_threads();
    //test_thread_pool();
    //test_scaled_dot_product_attention_forward();
    //test_scaled_dot_product_attention_backward();
    test_multihead_attention_forward();
    test_multihead_attention_backward();
    test_position_wise_ff_forward();
    test_position_wise_ff_backward();

    return 0;
}
