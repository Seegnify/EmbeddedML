#include <iostream>
#include <vector>
#include <thread>
#include <cmath>

#include <iostream>

#include "external/cnpy/cnpy.h"
#include "external/thread-pool-11/ThreadPool.h"

#include "examples/transformer.hh"
#include "main/unittest.hh"

bool isApprox(const Tensor& A, const Tensor& B, DTYPE eps)
{
  return (A-B).norm() < eps;
}

void print(const std::string& name, const Tensor& tensor)
{
  std::cout << name
  << " [" << tensor.rows() << " x " << tensor.cols() << "]"
  << std::endl;
  std::cout << tensor << std::endl;
}

void print(const std::string& name, const SparseTensor& tensor)
{
  std::cout << name
  << " [" << tensor.rows() << " x " << tensor.cols() << "]"
  << std::endl;
  std::cout << tensor << std::endl;
}

void print(const std::string& name, Function& f)
{
  print(name, f.forward());
}

void print(const Graph& g, bool values = false, bool grads = false)
{
  auto vars = g.named_variables();

  for (const auto& it: vars)
  {
    auto name = it.first;
    auto var = it.second;

    // print name and varaible value
    auto& tensor = var->forward();
    std::cout << "node[" << name << "]"
    << " [" << tensor.rows() << " x " << tensor.cols() << "]"
    << std::endl;
    if (values)
    {
      std::cout << "value" << std::endl;
      std::cout << tensor << std::endl;
    }
    if (grads)
    {
      std::cout << "gradient" << std::endl;
      std::cout << var->gradient() << std::endl;
    }
  }
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
  TEST_BEGIN("Numpy Test")

  Tensor two = Tensor::Zero(5,10);
  Tensor one = Tensor::Zero(1,5);

  for (int i=0; i<one.size(); i++) one(i) = i;
  for (int i=0; i<two.size(); i++) two(i) = i;

  // save array of tensors to npz file
  cnpy::npz_save("transformer-test.npz","varTwo",two.data(),{5,10},"w"); //"w" create
  cnpy::npz_save("transformer-test.npz","varOne",one.data(),{5},"a"); //"a" append

  // load the entire npz file
  cnpy::npz_t my_npz = cnpy::npz_load("transformer-test.npz");

  for (const auto& item : my_npz)
  {
    auto& name = item.first;
    auto& arr = item.second;

    std::cout << "name:" << name << std::endl;
    std::cout << "word: " << arr.word_size << std::endl;
    std::cout << "dims: " << arr.shape.size() << std::endl;
    size_t size = 1;
    for (auto s: arr.shape)
    {
      std::cout << "dimension:" << s << std::endl;
      size *= s;
    }
    std::cout << "size: " << size << std::endl;

    const float* data = arr.data<float>();
    if (arr.shape.size() == 1)
    {
      ConstTensorMap t(data, 1, arr.shape[0]);
      ASSERT(t == one);
    }
    if (arr.shape.size() == 2)
    {
      ConstTensorMap t(data, arr.shape[0], arr.shape[1]);
      ASSERT(t == two);
    }
  }

  TEST_END()
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

void test_sequence_mask()
{
    TEST_BEGIN("Sequence Mask")

    int TGT_TOKENS = 10;
    int BOS_TOKEN = 7;
    int EOS_TOKEN = 8;
    int PAD_TOKEN = 9;
    int SEQ_SIZE = 5;
    DTYPE I = std::numeric_limits<DTYPE>::infinity();

    Graph g;
    SequenceMask m(g, PAD_TOKEN, SEQ_SIZE);

    std::vector<int> src_seq = {1,2,3,PAD_TOKEN,PAD_TOKEN};
    std::vector<int> tgt_seq = {BOS_TOKEN,2,3,4,PAD_TOKEN};

    // test source mask
    m.source(src_seq);
    Tensor source(SEQ_SIZE, SEQ_SIZE);
    source <<
      0, 0, 0,-I,-I,
      0, 0, 0,-I,-I,
      0, 0, 0,-I,-I,
      0, 0, 0,-I,-I,
      0, 0, 0,-I,-I,
    ASSERT(m() == source)

    // test target mask
    m.target(tgt_seq);
    Tensor target(SEQ_SIZE, SEQ_SIZE);
    target <<
      0,-I,-I,-I,-I,
      0, 0,-I,-I,-I,
      0, 0, 0,-I,-I,
      0, 0, 0, 0,-I,
      0, 0, 0, 0,-I,
    ASSERT(m() == target)

    TEST_END()
}

void test_scaled_dot_product_attention_forward()
{
    TEST_BEGIN("Scaled Dot-Product Attention Forward")

    Graph g;
    DTYPE dropout = 0.0;
    DTYPE I = std::numeric_limits<DTYPE>::infinity();

    // [2x3]
    auto Q = g.new_constant(2, 3);
    Q->value() << 1,2,3,
                  4,5,6;

    // [4x3]
    auto K = g.new_constant(4, 3);
    K->value() << 0.1,0.2,0.3,
                  0.4,0.5,0.6,
                  1.4,1.5,1.6,
                  2.4,2.5,2.6;

    // QK_T -> [2x4]

    // [2x4]
    auto M = g.new_constant(2, 4);
    M->value() << 0, 0, 0, 0,
                  0, 0,-I,-I;

    // [4x5]
    auto V = g.new_constant(4, 5);
    V->value() << -2,7,8,2,2,
                  4,1,-9,3,3,
                  1,2,3,4,4,
                  4,5,6,5,5;

    // QK_TV -> [2x5]

    int T = Q->value().rows();
    int S = K->value().rows();
    int D = K->value().cols();

    ScaledDotProductAttention attn(g, *Q,*K,*V,*M, T, S, D, dropout);

    auto& A = attn();

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
    DTYPE I = std::numeric_limits<DTYPE>::infinity();

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
    M->value() << 0, 0, 0, 0,
                  0, 0,-I,-I;

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
      g, *Q,*K,*V,*M, T, S, D, dropout
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

    int TGT_TOKENS = 10;
    int BOS_TOKEN = 7;
    int EOS_TOKEN = 8;
    int PAD_TOKEN = 9;

    int TGT_SIZE = 3;
    int SEQ_SIZE = 3;
    int EMB_SIZE = 4;
    int NUM_HEADS = 2;
    bool bias = true;
    DTYPE dropout = 0.0;

    auto q = g.new_variable(TGT_SIZE, EMB_SIZE);
    auto k = g.new_variable(SEQ_SIZE, EMB_SIZE);
    auto v = g.new_variable(TGT_SIZE, EMB_SIZE);

    SequenceMask m(g, PAD_TOKEN, SEQ_SIZE);
    m.source({1,2,3}); // no special tokens

    MultiHeadAttention mha(
      g, *q, *k, *v, m,
      TGT_SIZE, SEQ_SIZE, EMB_SIZE, NUM_HEADS,
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

    // print(g);
    auto vars = g.named_variables();
    auto& Wq = *vars["MHA.Wq"];
    auto& Wk = *vars["MHA.Wk"];
    auto& Wv = *vars["MHA.Wv"];
    auto& Wo = *vars["MHA.Wo"];
    auto& bq = *vars["MHA.bq"];
    auto& bk = *vars["MHA.bk"];
    auto& bv = *vars["MHA.bv"];
    auto& bo = *vars["MHA.bo"];

    // Norm.A == 1
    // Norm.B == 0
    // Norm.A.1 == 1
    // Norm.B.1 == 0

    Wq.value() <<
      0.4271,  0.3013, -0.4279, -0.2122,
      0.2983,  0.3350, -0.4619,  0.5432,
      -0.1488,  0.1778, -0.4288, -0.5003,
      0.1173,  0.3713, -0.2347, -0.2251;   
    Wk.value() <<
      0.1557,  0.4673,  0.0920,  0.3889,
      0.5867,  0.0088,  0.4371,  0.0371,
      0.4897, -0.0109, -0.0646,  0.5190,
      -0.5768,  0.1376, -0.5507,  0.5315;
    Wv.value() <<
      -0.3599, -0.4841,  0.0526, -0.5235,
      -0.1576,  0.4844, -0.3817,  0.2549,
      -0.1432,  0.5141, -0.5741, -0.0179,
      -0.0103, -0.4235, -0.5195, -0.1589;
    Wo.value() <<
      -0.2588,  0.4873,  0.0642,  0.4206,
      0.3272,  0.3202,  0.4458, -0.3825,
      -0.4631, -0.2740, -0.2628, -0.4749,
      -0.3654,  0.4841,  0.4618, -0.1188;

    if (bias)
    {
      bq.value() <<
        0.4755, 0.1042, 0.6459, 0.2230;      
      bk.value() <<
        0.0739, 0.6705, 0.8532, 0.7830;
      bv.value() <<
        0.1097, 0.8451, 0.7208, 0.2440;
      bo.value() <<
        0.0307, 0.1667, 0.4442, 0.1971;
    }

    Tensor mha_hat(TGT_SIZE, EMB_SIZE);
    mha_hat <<   
        0.4363, 0.5356, 0.4469, 0.9232,
        0.4293, 0.5403, 0.4531, 0.9223,
        0.4353, 0.5364, 0.4456, 0.9224;

    ASSERT(mha().isApprox(mha_hat, 0.0001))
    ASSERT(mha().rows() == TGT_SIZE)
    ASSERT(mha().cols() == EMB_SIZE)

    TEST_END()
}

void test_multihead_attention_backward()
{
    TEST_BEGIN("Multi-Head Attention Backward")

    Graph g;

    int TGT_TOKENS = 5;
    int BOS_TOKEN = 7;
    int EOS_TOKEN = 8;
    int PAD_TOKEN = 9;

    int TGT_SIZE = 3;
    int SEQ_SIZE = 3;
    int EMB_SIZE = 4;
    int NUM_HEADS = 2;
    bool bias = true;
    DTYPE dropout = 0.0;

    auto q = g.new_variable(TGT_SIZE, EMB_SIZE);
    auto k = g.new_variable(SEQ_SIZE, EMB_SIZE);
    auto v = g.new_variable(TGT_SIZE, EMB_SIZE);

    SequenceMask m(g, PAD_TOKEN, SEQ_SIZE);
    m.source({1,2,3}); // no PAD_TOKEN

    auto mhaptr = new MultiHeadAttention(
      g, *q, *k, *v, m,
      TGT_SIZE, SEQ_SIZE, EMB_SIZE, NUM_HEADS,
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

    // print(g);
    auto vars = g.named_variables();
    auto& Wq = *vars["MHA.Wq"];
    auto& Wk = *vars["MHA.Wk"];
    auto& Wv = *vars["MHA.Wv"];
    auto& Wo = *vars["MHA.Wo"];
    auto& bq = *vars["MHA.bq"];
    auto& bk = *vars["MHA.bk"];
    auto& bv = *vars["MHA.bv"];
    auto& bo = *vars["MHA.bo"];

    // Norm.A == 1
    // Norm.B == 0
    // Norm.A.1 == 1
    // Norm.B.1 == 0

    Wq.value() <<
      0.4271,  0.3013, -0.4279, -0.2122,
      0.2983,  0.3350, -0.4619,  0.5432,
      -0.1488,  0.1778, -0.4288, -0.5003,
      0.1173,  0.3713, -0.2347, -0.2251;
    Wk.value() <<
      0.1557,  0.4673,  0.0920,  0.3889,
      0.5867,  0.0088,  0.4371,  0.0371,
      0.4897, -0.0109, -0.0646,  0.5190,
      -0.5768,  0.1376, -0.5507,  0.5315;
    Wv.value() <<
      -0.3599, -0.4841,  0.0526, -0.5235,
      -0.1576,  0.4844, -0.3817,  0.2549,
      -0.1432,  0.5141, -0.5741, -0.0179,
      -0.0103, -0.4235, -0.5195, -0.1589;
    Wo.value() <<
      -0.2588,  0.4873,  0.0642,  0.4206,
      0.3272,  0.3202,  0.4458, -0.3825,
      -0.4631, -0.2740, -0.2628, -0.4749,
      -0.3654,  0.4841,  0.4618, -0.1188;

    bq.value() <<
      0.4755, 0.1042, 0.6459, 0.2230;
    bk.value() <<
      0.0739, 0.6705, 0.8532, 0.7830;
    bv.value() <<
      0.1097, 0.8451, 0.7208, 0.2440;
    bo.value() <<
      0.0307, 0.1667, 0.4442, 0.1971;

    auto& F = mha();

    mha.gradient() = Tensor::Ones(TGT_SIZE, EMB_SIZE);
    mha.gradient()(0) = 5;

    const Tensor dFdq = q->backward();
    const Tensor dFdk = k->backward();
    const Tensor dFdv = v->backward();

    const Tensor dFdWq = Wq.backward();
    const Tensor dFdWk = Wk.backward();
    const Tensor dFdWv = Wv.backward();
    const Tensor dFdWo = Wo.backward();

    const Tensor dFdbq = bq.backward();
    const Tensor dFdbk = bk.backward();
    const Tensor dFdbv = bv.backward();
    const Tensor dFdbo = bo.backward();

    Tensor dFdq_hat(TGT_SIZE, EMB_SIZE);
    dFdq_hat <<
      -0.0026, -0.0249,  0.0291,  0.0100,
      0.0004,  0.0037, -0.0015, -0.0093,
      0.0003,  0.0034, -0.0013, -0.0091;

    Tensor dFdk_hat(TGT_SIZE, EMB_SIZE);
    dFdk_hat <<
      0.0074, -0.0047, -0.0078,  0.0137,
     -0.0168, -0.0184,  0.0007, -0.0346,
      0.0094,  0.0231,  0.0071,  0.0209;

    Tensor dFdv_hat(TGT_SIZE, EMB_SIZE);
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

    Tensor dFdWq_hat(EMB_SIZE, EMB_SIZE);
    dFdWq_hat <<
       0.0061,  0.0045,  0.0097,  0.0038,
      -0.0148, -0.0109, -0.0233, -0.0092,
      -0.0005,  0.0005, -0.0178, -0.0036,
       0.0172,  0.0145, -0.0096,  0.0035;

    Tensor dFdWk_hat(EMB_SIZE, EMB_SIZE);
    dFdWk_hat <<
      -0.0162,  0.0084, -0.0182,  0.0185,
      -0.0059,  0.0026, -0.0076,  0.0079,
      -0.0058,  0.0105,  0.0085, -0.0102,
      -0.0052,  0.0043, -0.0026,  0.0023;

    Tensor dFdWv_hat(EMB_SIZE, EMB_SIZE);
    dFdWv_hat <<
      -1.4292e+00, -4.6630e-01, -2.0257e+00, -2.1247e+00,
       2.1559e+00,  7.0466e-01,  3.0573e+00,  3.2059e+00,
       1.0382e+00,  3.5606e-01,  1.4924e+00,  1.5373e+00,
       8.4130e-03, -4.8284e-03,  2.0528e-03,  7.0446e-03;

    Tensor dFdWo_hat(EMB_SIZE, EMB_SIZE);
    dFdWo_hat <<
      -2.9211,  5.4293,  2.5510, -1.7456,
      -1.2491,  2.3256,  1.0927, -0.7558,
      -1.2491,  2.3256,  1.0927, -0.7558,
      -1.2491,  2.3256,  1.0927, -0.7558;

    ASSERT(dFdWq.isApprox(dFdWq_hat, 0.01))
    ASSERT(dFdWk.isApprox(dFdWk_hat, 0.01))
    ASSERT(dFdWv.isApprox(dFdWv_hat, 0.01))
    ASSERT(dFdWo.isApprox(dFdWo_hat, 0.01))

    Tensor dFdbq_hat(1, EMB_SIZE);
    dFdbq_hat <<
       0.0167, -0.0400, -0.0285, -0.0119;

    Tensor dFdbk_hat(1, EMB_SIZE);
    dFdbk_hat <<
       5.9605e-08, 2.2352e-08, 1.1176e-08, 1.5832e-08;

    Tensor dFdbv_hat(1, EMB_SIZE);
    dFdbv_hat <<
       -3.3155,  5.0020,  2.3838,  0.0156;

    Tensor dFdbo_hat(1, EMB_SIZE);
    dFdbo_hat <<
       7., 3., 3., 3.;

    ASSERT(isApprox(dFdbq, dFdbq_hat, 0.0001))
    ASSERT(isApprox(dFdbk, dFdbk_hat, 0.0001))
    ASSERT(isApprox(dFdbv, dFdbv_hat, 0.0001))
    ASSERT(isApprox(dFdbo, dFdbo_hat, 0.0001))

    TEST_END()
}


void test_position_wise_ff_forward()
{
    TEST_BEGIN("Position-Wise FF Forward")
        
    Graph g;

    int EMB_SIZE = 4;
    int FF_SIZE = 3;
    int TRG_SIZE = 2;
    DTYPE dropout = 0.0;

    auto& x = *g.new_variable(TRG_SIZE, EMB_SIZE);
    ASSERT(x.value().rows() == TRG_SIZE);
    ASSERT(x.value().cols() == EMB_SIZE);

    x.value() <<
      0.0878, 0.0416, 0.6166, 0.1477,
      0.5300, 0.2800, 0.5306, 0.4950;

    PositionWiseFeedForward ff(
      g, x, EMB_SIZE, FF_SIZE, dropout
    );
    
    // print(g);
    auto vars = g.named_variables();
    auto& L1W = *vars["Linear.W"];
    auto& L1b = *vars["Linear.b"];
    auto& L2W = *vars["Linear.W.1"];
    auto& L2b = *vars["Linear.b.1"];
    
    ASSERT(L1W.value().rows() == FF_SIZE);
    ASSERT(L1W.value().cols() == EMB_SIZE);
    ASSERT(L1b.value().rows() == 1);
    ASSERT(L1b.value().cols() == FF_SIZE);

    ASSERT(L2W.value().rows() == EMB_SIZE);
    ASSERT(L2W.value().cols() == FF_SIZE);
    ASSERT(L2b.value().rows() == 1);
    ASSERT(L2b.value().cols() == EMB_SIZE);

    L1W.value() <<
      -0.3883,  0.2742, -0.4652, -0.1417,
      -0.0996, -0.4170, -0.0302,  0.1254,
      -0.2065,  0.0107,  0.3998,  0.3775;
    L2W.value() <<
       0.0348,  0.3779, -0.5751,
      -0.0708, -0.4522, -0.4000,
       0.3196,  0.2163,  0.5397,
      -0.1805,  0.0472, -0.4630;

    L1b.value() <<
      0.4282,  0.2099, -0.2209;
    L2b.value() <<
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
    int FF_SIZE = 3;
    int TRG_SIZE = 2;
    DTYPE dropout = 0.0;

    auto& x = *g.new_variable(TRG_SIZE, EMB_SIZE);
    ASSERT(x.value().rows() == TRG_SIZE);
    ASSERT(x.value().cols() == EMB_SIZE);

    x.value() <<
      0.0878, 0.0416, 0.6166, 0.1477,
      0.5300, 0.2800, 0.5306, 0.4950;

    auto& ff = *(new PositionWiseFeedForward(
      g, x, EMB_SIZE, FF_SIZE, dropout
    ));
    g.keep(&ff);

    // print(g);
    auto vars = g.named_variables();
    auto& L1W = *vars["Linear.W"];
    auto& L1b = *vars["Linear.b"];
    auto& L2W = *vars["Linear.W.1"];
    auto& L2b = *vars["Linear.b.1"];
    
    ASSERT(L1W.value().rows() == FF_SIZE);
    ASSERT(L1W.value().cols() == EMB_SIZE);
    ASSERT(L1b.value().rows() == 1);
    ASSERT(L1b.value().cols() == FF_SIZE);

    ASSERT(L2W.value().rows() == EMB_SIZE);
    ASSERT(L2W.value().cols() == FF_SIZE);
    ASSERT(L2b.value().rows() == 1);
    ASSERT(L2b.value().cols() == EMB_SIZE);

    L1W.value() <<
      -0.3883,  0.2742, -0.4652, -0.1417,
      -0.0996, -0.4170, -0.0302,  0.1254,
      -0.2065,  0.0107,  0.3998,  0.3775;
    L2W.value() <<
       0.0348,  0.3779, -0.5751,
      -0.0708, -0.4522, -0.4000,
       0.3196,  0.2163,  0.5397,
      -0.1805,  0.0472, -0.4630;

    L1b.value() <<
      0.4282,  0.2099, -0.2209;
    L2b.value() <<
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
    
    Tensor dFdW1 = L1W.backward();
    Tensor dFdW2 = L2W.backward();
    Tensor dFdb1 = L1b.backward();
    Tensor dFdb2 = L2b.backward();
    Tensor dFdx = x.backward();

    Tensor dFdW1_num = g.dFdX(ff, L1W);
    Tensor dFdW2_num = g.dFdX(ff, L2W);
    Tensor dFdb1_num = g.dFdX(ff, L1b);
    Tensor dFdb2_num = g.dFdX(ff, L2b);
    Tensor dFdx_num = g.dFdX(ff, x);
        
    Tensor dFdW1_hat(FF_SIZE, EMB_SIZE);
    dFdW1_hat <<
      0.0213,  0.0101,  0.1494,  0.0358,
      0.2496,  0.1237,  1.1491,  0.3449,
      -0.7570, -0.3846, -2.4491, -0.9172;    

    Tensor dFdb1_hat(1, FF_SIZE);
    dFdb1_hat <<
      0.2423,  1.8900, -4.0972;

    Tensor dFdW2_hat(EMB_SIZE, FF_SIZE);
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

void test_positional_encoding_forward()
{
    TEST_BEGIN("PositionalEncoding Forward")

    int EMB_SIZE = 4;
    int SEQ_SIZE = 5;
    int MAX_SEQ_SIZE = 7;

    Graph g;

    auto& x = *g.new_variable(SEQ_SIZE, EMB_SIZE);
    x.value() <<
      1,2,3,4,
      5,6,7,8,
      0.0878, 0.0416, 0.6166, 0.1477,
      -0.3883,  0.2742, -0.4652, -0.1417,
      0.5300, 0.2800, 0.5306, 0.4950;

    PositionalEncoding pe(g, x, MAX_SEQ_SIZE, EMB_SIZE);

    auto y = pe();

    Tensor y_hat(SEQ_SIZE, EMB_SIZE);
    y_hat <<
      1.0000,  3.0000,  3.0000,  5.0000,
      5.8415,  6.5403,  7.0100,  9.0000,
      0.9971, -0.3745,  0.6366,  1.1475,
      -0.2472, -0.7158, -0.4352,  0.8579,
      -0.2268, -0.3736,  0.5706,  1.4942;

    ASSERT(y.isApprox(y_hat, 0.0001))

    TEST_END()
}

void test_positional_encoding_backward()
{
    TEST_BEGIN("PositionalEncoding Backward")

    int EMB_SIZE = 4;
    int SEQ_SIZE = 5;
    int MAX_SEQ_SIZE = 7;

    Graph g;

    auto& x = *g.new_variable(SEQ_SIZE, EMB_SIZE);
    x.value() <<
      1,2,3,4,
      5,6,7,8,
      0.0878, 0.0416, 0.6166, 0.1477,
      -0.3883,  0.2742, -0.4652, -0.1417,
      0.5300, 0.2800, 0.5306, 0.4950;

    auto& pe = *(new PositionalEncoding(g, x, MAX_SEQ_SIZE, EMB_SIZE));
    g.keep(&pe);

    auto y = pe();

    pe.gradient() = Tensor::Ones(SEQ_SIZE, EMB_SIZE);
    pe.gradient()(0) = 5;

    Tensor dFdx = x.backward();
    Tensor dFdx_num = g.dFdX(pe, x);
    Tensor dFdx_hat = pe.gradient();

    ASSERT(dFdx == dFdx_hat)
    ASSERT(dFdx.isApprox(dFdx_num, 0.0001))

    TEST_END()
}

void test_encoder_layer_forward()
{
    TEST_BEGIN("EncoderLayer Forward")

    int TGT_TOKENS = 10;
    int BOS_TOKEN = 7;
    int EOS_TOKEN = 8;
    int PAD_TOKEN = 9;

    int EMB_SIZE = 4;
    int SEQ_SIZE = 5;
    int NUM_HEADS = 2;
    int FF_SIZE = 3;
    DTYPE dropout = 0.0;

    Graph g;

    auto& x = *g.new_variable(SEQ_SIZE, EMB_SIZE, "x");
    x.value() <<
      1,2,3,4,
      5,6,7,8,
      0.0878, 0.0416, 0.6166, 0.1477,
      -0.3883,  0.2742, -0.4652, -0.1417,
      0.5300, 0.2800, 0.5306, 0.4950;

    SequenceMask m(g, PAD_TOKEN, SEQ_SIZE);
    m.source({1,2,3,4,5});

    EncoderLayer el(g, x, m,
      SEQ_SIZE, EMB_SIZE, NUM_HEADS, FF_SIZE, dropout);

    auto vars = g.named_variables();
    // print(g);

    auto& Wq = *vars["MHA.Wq"];
    auto& Wk = *vars["MHA.Wk"];
    auto& Wv = *vars["MHA.Wv"];
    auto& Wo = *vars["MHA.Wo"];

    auto& bq = *vars["MHA.bq"];
    auto& bk = *vars["MHA.bk"];
    auto& bv = *vars["MHA.bv"];
    auto& bo = *vars["MHA.bo"];

    // element-wise feed forward
    auto& ffL1w = *vars["Linear.W"];
    auto& ffL1b = *vars["Linear.b"];
    auto& ffL2w = *vars["Linear.W.1"];
    auto& ffL2b = *vars["Linear.b.1"];

    // norms (default to A=1 and B=0)

    Wq.value() <<
      -1.2321, -0.4785, -0.4598, -0.1860,
       0.4576,  0.4961, -0.0903, -0.4833,
      -0.1442,  0.3495,  0.4236, -0.0846,
      -0.3082,  0.0956, -0.2470,  0.3061;
    bq.value() <<
      -1.3717, -0.1179, -0.0096, -0.4240;
    Wk.value() <<
      -2.2321, -0.4785, -0.4598, -0.1860,
       0.4576,  0.4961, -0.0903, -0.4833,
      -0.1442,  0.3495,  0.4236, -0.0846,
      -0.3082,  0.0956, -0.2470,  0.3061;
    bk.value() <<
      -2.3717, -0.1179, -0.0096, -0.4240;
    Wv.value() <<
      -3.2321, -0.4785, -0.4598, -0.1860,
       0.4576,  0.4961, -0.0903, -0.4833,
      -0.1442,  0.3495,  0.4236, -0.0846,
      -0.3082,  0.0956, -0.2470,  0.3061;
    bv.value() <<
      -3.3717, -0.1179, -0.0096, -0.4240;
    Wo.value() <<
      -4.2321, -0.4785, -0.4598, -0.1860,
       0.4576,  0.4961, -0.0903, -0.4833,
      -0.1442,  0.3495,  0.4236, -0.0846,
      -0.3082,  0.0956, -0.2470,  0.3061;
    bo.value() <<
      -4.3717, -0.1179, -0.0096, -0.4240;
    ffL1w.value() <<
      -5.4208,  0.2836, -0.1770,  0.3684,
       0.3448,  0.4124, -0.2545,  0.2874,
      -0.4372,  0.4165, -0.2362,  0.1144;
    ffL1b.value() <<
       5.2621, -0.3262,  0.4815;
    ffL2w.value() <<
      -6.3926, -0.1717,  0.2300,
       0.0701,  0.3166, -0.2458,
       0.1431, -0.3391,  0.5407,
       0.4126, -0.3719,  0.5352;
    ffL2b.value() <<
      -6.5333, -0.0515, -0.1337,  0.0297;

    g.recache();
    auto y = el();

    Tensor y_hat(SEQ_SIZE, EMB_SIZE);
    y_hat <<
      -1.7227,  0.4192,  0.5928,  0.7106,
      -1.7228,  0.4187,  0.5948,  0.7092,
      -1.7241,  0.4332,  0.5882,  0.7027,
      -1.7244,  0.4406,  0.5782,  0.7056,
      -1.7241,  0.4344,  0.5848,  0.7048;
         
    ASSERT(y.isApprox(y_hat, 0.0001))

    TEST_END()
}

void test_encoder_layer_backward()
{
    TEST_BEGIN("EncoderLayer Backward")

    int TGT_TOKENS = 10;
    int BOS_TOKEN = 7;
    int EOS_TOKEN = 8;
    int PAD_TOKEN = 9;

    int EMB_SIZE = 4;
    int SEQ_SIZE = 5;
    int NUM_HEADS = 2;
    int FF_SIZE = 3;
    DTYPE dropout = 0.0;

    Graph g;

    auto& x = *g.new_variable(SEQ_SIZE, EMB_SIZE, "x");
    x.value() <<
      1,2,3,4,
      5,6,7,8,
      0.0878, 0.0416, 0.6166, 0.1477,
      -0.3883,  0.2742, -0.4652, -0.1417,
      0.5300, 0.2800, 0.5306, 0.4950;

    SequenceMask m(g, PAD_TOKEN, SEQ_SIZE);
    m.source({1,2,3,4,5});

    auto& F = *(new EncoderLayer(g, x, m,
      SEQ_SIZE, EMB_SIZE, NUM_HEADS, FF_SIZE, dropout));
    g.keep(&F);

    auto vars = g.named_variables();

    //print(g);
    auto& Wq = *vars["MHA.Wq"];
    auto& Wk = *vars["MHA.Wk"];
    auto& Wv = *vars["MHA.Wv"];
    auto& Wo = *vars["MHA.Wo"];

    auto& bq = *vars["MHA.bq"];
    auto& bk = *vars["MHA.bk"];
    auto& bv = *vars["MHA.bv"];
    auto& bo = *vars["MHA.bo"];

    // element-wise feed forward
    auto& ffL1w = *vars["Linear.W"];
    auto& ffL1b = *vars["Linear.b"];
    auto& ffL2w = *vars["Linear.W.1"];
    auto& ffL2b = *vars["Linear.b.1"];

    // norms (default to A=1 and B=0)

    Wq.value() <<
      -1.2321, -0.4785, -0.4598, -0.1860,
       0.4576,  0.4961, -0.0903, -0.4833,
      -0.1442,  0.3495,  0.4236, -0.0846,
      -0.3082,  0.0956, -0.2470,  0.3061;
    bq.value() <<
      -1.3717, -0.1179, -0.0096, -0.4240;
    Wk.value() <<
      -2.2321, -0.4785, -0.4598, -0.1860,
       0.4576,  0.4961, -0.0903, -0.4833,
      -0.1442,  0.3495,  0.4236, -0.0846,
      -0.3082,  0.0956, -0.2470,  0.3061;
    bk.value() <<
      -2.3717, -0.1179, -0.0096, -0.4240;
    Wv.value() <<
      -3.2321, -0.4785, -0.4598, -0.1860,
       0.4576,  0.4961, -0.0903, -0.4833,
      -0.1442,  0.3495,  0.4236, -0.0846,
      -0.3082,  0.0956, -0.2470,  0.3061;
    bv.value() <<
      -3.3717, -0.1179, -0.0096, -0.4240;
    Wo.value() <<
      -4.2321, -0.4785, -0.4598, -0.1860,
       0.4576,  0.4961, -0.0903, -0.4833,
      -0.1442,  0.3495,  0.4236, -0.0846,
      -0.3082,  0.0956, -0.2470,  0.3061;
    bo.value() <<
      -4.3717, -0.1179, -0.0096, -0.4240;
    ffL1w.value() <<
      -5.4208,  0.2836, -0.1770,  0.3684,
       0.3448,  0.4124, -0.2545,  0.2874,
      -0.4372,  0.4165, -0.2362,  0.1144;
    ffL1b.value() <<
       5.2621, -0.3262,  0.4815;
    ffL2w.value() <<
      -6.3926, -0.1717,  0.2300,
       0.0701,  0.3166, -0.2458,
       0.1431, -0.3391,  0.5407,
       0.4126, -0.3719,  0.5352;
    ffL2b.value() <<
      -6.5333, -0.0515, -0.1337,  0.0297;

    F();

    F.gradient() = Tensor::Ones(SEQ_SIZE, EMB_SIZE);
    F.gradient()(0) = 1250;

    const Tensor dFdx = x.backward();

    Tensor dFdx_hat = Tensor::Zero(SEQ_SIZE, EMB_SIZE);
    dFdx_hat << // torch output
      -0.2000, -0.8638,  0.1263,  0.9108,
      -0.5233, -0.0899, -0.1796,  0.2701,
      -0.0053,  0.0018, -0.0040,  0.0051,
      -0.0045,  0.0028, -0.0010,  0.0033,
      -0.0051,  0.0018, -0.0037,  0.0049;

    ASSERT(dFdx.isApprox(dFdx_hat, 0.001))

    TEST_END()
}

void test_decoder_layer_forward()
{
    TEST_BEGIN("DecoderLayer Forward")

    int TGT_TOKENS = 10;
    int BOS_TOKEN = 7;
    int EOS_TOKEN = 8;
    int PAD_TOKEN = 9;

    int EMB_SIZE = 4;
    int SEQ_SIZE = 5;
    int NUM_HEADS = 2;
    int FF_SIZE = 3;
    DTYPE dropout = 0.0;

    Graph g;

    auto& x = *g.new_variable(SEQ_SIZE, EMB_SIZE, "x");
    x.value() <<
      0.1878, 0.5416, -0.1166, 0.4477,
      0.2878, -0.6416, 0.2166, -0.9477,
      0.0878, 0.0416, 0.6166, 0.1477,
      -0.3883,  0.2742, -0.4652, -0.1417,
      0.5300, 0.2800, 0.5306, 0.4950;

    auto& e = *g.new_variable(SEQ_SIZE, EMB_SIZE, "e");
    e.value() <<
      -1.7227,  0.4192,  0.5928,  0.7106,
      -1.7228,  0.4187,  0.5948,  0.7092,
      -1.7241,  0.4332,  0.5882,  0.7027,
      -1.7244,  0.4406,  0.5782,  0.7056,
      -1.7241,  0.4344,  0.5848,  0.7048;

    SequenceMask src_mask(g, PAD_TOKEN, SEQ_SIZE);
    SequenceMask tgt_mask(g, PAD_TOKEN, SEQ_SIZE);

    src_mask.source({1,2,3,4,5});
    tgt_mask.target({1,2,3,4,5});

    DecoderLayer dl(g, x, e, src_mask, tgt_mask,
      SEQ_SIZE, EMB_SIZE, NUM_HEADS, FF_SIZE, dropout);

    auto vars = g.named_variables();
    //print(g);

    // self-attention
    auto& Wq = *vars["MHA.Wq"];
    auto& Wk = *vars["MHA.Wk"];
    auto& Wv = *vars["MHA.Wv"];
    auto& Wo = *vars["MHA.Wo"];

    auto& bq = *vars["MHA.bq"];
    auto& bk = *vars["MHA.bk"];
    auto& bv = *vars["MHA.bv"];
    auto& bo = *vars["MHA.bo"];

    // cross-attention
    auto& cross_attn_Wq = *vars["MHA.Wq.1"];
    auto& cross_attn_Wk = *vars["MHA.Wk.1"];
    auto& cross_attn_Wv = *vars["MHA.Wv.1"];
    auto& cross_attn_Wo = *vars["MHA.Wo.1"];

    auto& cross_attn_bq = *vars["MHA.bq.1"];
    auto& cross_attn_bk = *vars["MHA.bk.1"];
    auto& cross_attn_bv = *vars["MHA.bv.1"];
    auto& cross_attn_bo = *vars["MHA.bo.1"];

    // element-wise feed forward
    auto& ffL1w = *vars["Linear.W"];
    auto& ffL1b = *vars["Linear.b"];
    auto& ffL2w = *vars["Linear.W.1"];
    auto& ffL2b = *vars["Linear.b.1"];

    // norms (default to A=1 and B=0)

    Wq.value() <<
      -1.2321, -0.4785, -0.4598, -0.1860,
       0.4576,  0.4961, -0.0903, -0.4833,
      -0.1442,  0.3495,  0.4236, -0.0846,
      -0.3082,  0.0956, -0.2470,  0.3061;
    bq.value() <<
      -1.3717, -0.1179, -0.0096, -0.4240;
    Wk.value() <<
      -2.2321, -0.4785, -0.4598, -0.1860,
       0.4576,  0.4961, -0.0903, -0.4833,
      -0.1442,  0.3495,  0.4236, -0.0846,
      -0.3082,  0.0956, -0.2470,  0.3061;
    bk.value() <<
      -2.3717, -0.1179, -0.0096, -0.4240;
    Wv.value() <<
      -3.2321, -0.4785, -0.4598, -0.1860,
       0.4576,  0.4961, -0.0903, -0.4833,
      -0.1442,  0.3495,  0.4236, -0.0846,
      -0.3082,  0.0956, -0.2470,  0.3061;
    bv.value() <<
      -3.3717, -0.1179, -0.0096, -0.4240;
    Wo.value() <<
      -4.2321, -0.4785, -0.4598, -0.1860,
       0.4576,  0.4961, -0.0903, -0.4833,
      -0.1442,  0.3495,  0.4236, -0.0846,
      -0.3082,  0.0956, -0.2470,  0.3061;
    bo.value() <<
      -4.3717, -0.1179, -0.0096, -0.4240;

    cross_attn_Wq.value() <<
       0.0675,  0.0034,  0.2860, -0.0438,
       0.3234,  0.4208, -0.0814, -0.0883,
      -0.3376,  0.2880,  0.0641, -0.4295,
       0.4480,  0.4328, -0.4657,  0.1207;
    cross_attn_bq.value() <<
      -0.3390,  0.0716,  0.4804, -0.4253;
    cross_attn_Wk.value() <<
       0.2975,  0.0247,  0.4618, -0.1429,
      -0.0016, -0.0542, -0.3919,  0.1051,
       0.4285,  0.0760, -0.3002, -0.2579,
      -0.1038,  0.4511,  0.4412,  0.2605;
    cross_attn_bk.value() <<
      -0.3793,  0.4552,  0.1502,  0.3554;
    cross_attn_Wv.value() <<
      -0.4192, -0.4004,  0.0120, -0.4717,
      -0.3308, -0.4728, -0.1381,  0.3374,
       0.1521, -0.1548,  0.2885,  0.4352,
      -0.1196, -0.2579, -0.3167,  0.0128;
    cross_attn_bv.value() <<
       0.4992, -0.2558,  0.1871, -0.3701;
    cross_attn_Wo.value() <<
       1.5146e-01,  5.0816e-02,  3.9053e-04, -4.6405e-01,
      -1.2832e-01, -4.3910e-01, -1.8390e-01, -5.1324e-02,
       4.4734e-01, -3.3816e-01,  1.3738e-01, -1.3041e-01,
       1.8204e-01, -2.9708e-01,  3.2434e-01, -6.3109e-02;
    cross_attn_bo.value() <<
      -0.4427, -0.0959, -0.2821, -0.2209;

    ffL1w.value() <<
      -5.4208,  0.2836, -0.1770,  0.3684,
       0.3448,  0.4124, -0.2545,  0.2874,
      -0.4372,  0.4165, -0.2362,  0.1144;
    ffL1b.value() <<
       5.2621, -0.3262,  0.4815;
    ffL2w.value() <<
      -6.3926, -0.1717,  0.2300,
       0.0701,  0.3166, -0.2458,
       0.1431, -0.3391,  0.5407,
       0.4126, -0.3719,  0.5352;
    ffL2b.value() <<
      -6.5333, -0.0515, -0.1337,  0.0297;

    g.recache();
    auto y = dl();

    Tensor y_hat(SEQ_SIZE, EMB_SIZE);
    y_hat <<
      -1.7098,  0.3158,  0.6462,  0.7478,
      -1.7089,  0.3055,  0.7284,  0.6750,
      -1.7062,  0.2879,  0.7098,  0.7085,
      -1.7133,  0.3351,  0.6518,  0.7263,
      -1.7080,  0.2995,  0.6785,  0.7299;

    ASSERT(isApprox(y, y_hat, 0.0002))

    TEST_END()
}

void test_decoder_layer_backward()
{
    TEST_BEGIN("DecoderLayer Backward")

    int TGT_TOKENS = 10;
    int BOS_TOKEN = 7;
    int EOS_TOKEN = 8;
    int PAD_TOKEN = 9;

    int EMB_SIZE = 4;
    int SEQ_SIZE = 5;
    int NUM_HEADS = 2;
    int FF_SIZE = 3;
    DTYPE dropout = 0.0;

    Graph g;

    auto& x = *g.new_variable(SEQ_SIZE, EMB_SIZE, "x");
    x.value() <<
      0.1878, 0.5416, -0.1166, 0.4477,
      0.2878, -0.6416, 0.2166, -0.9477,
      0.0878, 0.0416, 0.6166, 0.1477,
      -0.3883,  0.2742, -0.4652, -0.1417,
      0.5300, 0.2800, 0.5306, 0.4950;

    auto& e = *g.new_variable(SEQ_SIZE, EMB_SIZE, "e");
    e.value() <<
      -1.7227,  0.4192,  0.5928,  0.7106,
      -1.7228,  0.4187,  0.5948,  0.7092,
      -1.7241,  0.4332,  0.5882,  0.7027,
      -1.7244,  0.4406,  0.5782,  0.7056,
      -1.7241,  0.4344,  0.5848,  0.7048;

    SequenceMask src_mask(g, PAD_TOKEN, SEQ_SIZE);
    SequenceMask tgt_mask(g, PAD_TOKEN, SEQ_SIZE);

    src_mask.source({1,2,3,4,5});
    tgt_mask.target({1,2,3,4,5});

    auto& F = *(new DecoderLayer(g, x, e, src_mask, tgt_mask,
      SEQ_SIZE, EMB_SIZE, NUM_HEADS, FF_SIZE, dropout));
    g.keep(&F);

    auto vars = g.named_variables();
    //print(g);

    // self-attention
    auto& Wq = *vars["MHA.Wq"];
    auto& Wk = *vars["MHA.Wk"];
    auto& Wv = *vars["MHA.Wv"];
    auto& Wo = *vars["MHA.Wo"];

    auto& bq = *vars["MHA.bq"];
    auto& bk = *vars["MHA.bk"];
    auto& bv = *vars["MHA.bv"];
    auto& bo = *vars["MHA.bo"];

    // cross-attention
    auto& cross_attn_Wq = *vars["MHA.Wq.1"];
    auto& cross_attn_Wk = *vars["MHA.Wk.1"];
    auto& cross_attn_Wv = *vars["MHA.Wv.1"];
    auto& cross_attn_Wo = *vars["MHA.Wo.1"];

    auto& cross_attn_bq = *vars["MHA.bq.1"];
    auto& cross_attn_bk = *vars["MHA.bk.1"];
    auto& cross_attn_bv = *vars["MHA.bv.1"];
    auto& cross_attn_bo = *vars["MHA.bo.1"];

    // element-wise feed forward
    auto& ffL1w = *vars["Linear.W"];
    auto& ffL1b = *vars["Linear.b"];
    auto& ffL2w = *vars["Linear.W.1"];
    auto& ffL2b = *vars["Linear.b.1"];

    // norms (default to A=1 and B=0)

    Wq.value() <<
      -1.2321, -0.4785, -0.4598, -0.1860,
       0.4576,  0.4961, -0.0903, -0.4833,
      -0.1442,  0.3495,  0.4236, -0.0846,
      -0.3082,  0.0956, -0.2470,  0.3061;
    bq.value() <<
      -1.3717, -0.1179, -0.0096, -0.4240;
    Wk.value() <<
      -2.2321, -0.4785, -0.4598, -0.1860,
       0.4576,  0.4961, -0.0903, -0.4833,
      -0.1442,  0.3495,  0.4236, -0.0846,
      -0.3082,  0.0956, -0.2470,  0.3061;
    bk.value() <<
      -2.3717, -0.1179, -0.0096, -0.4240;
    Wv.value() <<
      -3.2321, -0.4785, -0.4598, -0.1860,
       0.4576,  0.4961, -0.0903, -0.4833,
      -0.1442,  0.3495,  0.4236, -0.0846,
      -0.3082,  0.0956, -0.2470,  0.3061;
    bv.value() <<
      -3.3717, -0.1179, -0.0096, -0.4240;
    Wo.value() <<
      -4.2321, -0.4785, -0.4598, -0.1860,
       0.4576,  0.4961, -0.0903, -0.4833,
      -0.1442,  0.3495,  0.4236, -0.0846,
      -0.3082,  0.0956, -0.2470,  0.3061;
    bo.value() <<
      -4.3717, -0.1179, -0.0096, -0.4240;

    cross_attn_Wq.value() <<
       0.0675,  0.0034,  0.2860, -0.0438,
       0.3234,  0.4208, -0.0814, -0.0883,
      -0.3376,  0.2880,  0.0641, -0.4295,
       0.4480,  0.4328, -0.4657,  0.1207;
    cross_attn_bq.value() <<
      -0.3390,  0.0716,  0.4804, -0.4253;
    cross_attn_Wk.value() <<
       0.2975,  0.0247,  0.4618, -0.1429,
      -0.0016, -0.0542, -0.3919,  0.1051,
       0.4285,  0.0760, -0.3002, -0.2579,
      -0.1038,  0.4511,  0.4412,  0.2605;
    cross_attn_bk.value() <<
      -0.3793,  0.4552,  0.1502,  0.3554;
    cross_attn_Wv.value() <<
      -0.4192, -0.4004,  0.0120, -0.4717,
      -0.3308, -0.4728, -0.1381,  0.3374,
       0.1521, -0.1548,  0.2885,  0.4352,
      -0.1196, -0.2579, -0.3167,  0.0128;
    cross_attn_bv.value() <<
       0.4992, -0.2558,  0.1871, -0.3701;
    cross_attn_Wo.value() <<
       1.5146e-01,  5.0816e-02,  3.9053e-04, -4.6405e-01,
      -1.2832e-01, -4.3910e-01, -1.8390e-01, -5.1324e-02,
       4.4734e-01, -3.3816e-01,  1.3738e-01, -1.3041e-01,
       1.8204e-01, -2.9708e-01,  3.2434e-01, -6.3109e-02;
    cross_attn_bo.value() <<
      -0.4427, -0.0959, -0.2821, -0.2209;

    ffL1w.value() <<
      -5.4208,  0.2836, -0.1770,  0.3684,
       0.3448,  0.4124, -0.2545,  0.2874,
      -0.4372,  0.4165, -0.2362,  0.1144;
    ffL1b.value() <<
       5.2621, -0.3262,  0.4815;
    ffL2w.value() <<
      -6.3926, -0.1717,  0.2300,
       0.0701,  0.3166, -0.2458,
       0.1431, -0.3391,  0.5407,
       0.4126, -0.3719,  0.5352;
    ffL2b.value() <<
      -6.5333, -0.0515, -0.1337,  0.0297;

    F();

    F.gradient() = Tensor::Ones(SEQ_SIZE, EMB_SIZE);
    F.gradient().leftCols(1) = Tensor::Constant(SEQ_SIZE,1, 1250);

    const Tensor dFdx = x.backward();
    const Tensor dFde = e.backward();

    Tensor dFdx_hat(SEQ_SIZE, EMB_SIZE);
    dFdx_hat << // torch output
      -14.5332,  -9.9376,   1.3404,  13.4942,
       -8.9433,  -8.1209,   5.1974,   8.7049,
       -5.5773, -10.1252,   5.7661,   9.0569,
       -2.3403, -10.5440,   4.0029,   9.6241,
       -5.8445,  -8.6438,   2.9895,   6.3003;

    Tensor dFde_hat(SEQ_SIZE, EMB_SIZE);
    dFde_hat << // torch output
       -7.0047, -15.9333,   6.9179,  -0.0261,
       -7.0010, -15.9295,   6.9130,  -0.0180,
       -6.9935, -15.9405,   6.9058,  -0.0468,
       -7.0046, -15.9595,   6.9215,  -0.0879,
       -6.9993, -15.9472,   6.9133,  -0.0605;

    ASSERT(isApprox(dFdx, dFdx_hat, 0.001))
    ASSERT(isApprox(dFde, dFde_hat, 0.001))

    TEST_END()
}

void test_transformer_forward()
{
    TEST_BEGIN("Transformer Forward")

    int NUM_LAYERS = 1;
    int NUM_HEADS = 2;
    int EMB_SIZE = 4;
    int SEQ_SIZE = 5;
    int FF_SIZE = 3;
    DTYPE DROPOUT = 0.0;

    int SRC_TOKENS = 10; // input vocabulary size
    int TGT_TOKENS = 10; // output vocabulary size
    int BOS_TOKEN = 8; // begining of sequence token
    int EOS_TOKEN = 9; // end of sequence token
    int PAD_TOKEN = 0; // padding token (marks positions to ignore)

    Graph g;

    std::vector<int> src_x = {1, 2, 3, PAD_TOKEN, PAD_TOKEN}; // input sequence

    std::vector<int> tgt_x = {BOS_TOKEN, 1, 2, 3, PAD_TOKEN}; // target sequence

    std::vector<int> y_hat = {1, 2, 3, EOS_TOKEN, PAD_TOKEN}; // output sequence

    Transformer T(g, SRC_TOKENS, TGT_TOKENS, PAD_TOKEN,
      NUM_LAYERS, NUM_HEADS, EMB_SIZE, FF_SIZE, SEQ_SIZE, DROPOUT);

    auto vars = g.named_variables();
    //print(g);

    /////////////////////////////////////////////////////
    // embedding weights
    /////////////////////////////////////////////////////

    auto& enc_emb = *vars["Embedding.E"];
    auto& dec_emb = *vars["Embedding.E.1"];

    enc_emb.value() <<
      -1.5104, -0.0284,  0.8700,  0.8670,
       1.1990,  1.1361,  0.8644,  0.1473,
      -0.7335,  0.3807,  0.7741,  0.4396,
      -0.7043,  1.6892, -0.5124,  0.4657,
       0.0162, -1.9550,  1.6194,  1.5560,
       1.3285,  0.2094, -1.5481,  0.2234,
      -0.8587, -0.4757,  0.1260, -1.9552,
      -0.7352, -0.9533, -0.7015, -1.8978,
      -0.3166,  1.9976,  0.1297,  0.9044,
       0.7586, -1.0734, -0.4338, -0.8578;
    dec_emb.value() <<
      -0.5281,  0.1697, -0.9366,  0.0129,
       0.1969, -1.1860,  0.0961, -0.6315,
      -0.1603, -0.1080,  0.1573,  0.9020,
       1.2745,  0.0302,  1.0822, -0.4542,
      -0.5246, -0.9769, -0.6335,  2.4609,
      -0.7565,  1.0006, -1.5738,  1.5421,
      -0.7897, -3.4246,  0.4188, -1.1293,
       0.9453, -0.1520, -0.7238,  0.7383,
       0.0729,  0.3859,  0.0163, -0.9825,
      -0.9148,  1.0367, -1.0432,  1.4304;

    /////////////////////////////////////////////////////
    // encoder weights
    /////////////////////////////////////////////////////
    
    auto& Wq = *vars["encoder:MHA.Wq"];
    auto& Wk = *vars["encoder:MHA.Wk"];
    auto& Wv = *vars["encoder:MHA.Wv"];
    auto& Wo = *vars["encoder:MHA.Wo"];

    auto& bq = *vars["encoder:MHA.bq"];
    auto& bk = *vars["encoder:MHA.bk"];
    auto& bv = *vars["encoder:MHA.bv"];
    auto& bo = *vars["encoder:MHA.bo"];

    // element-wise feed forward
    auto& ffL1w = *vars["encoder:Linear.W"];
    auto& ffL1b = *vars["encoder:Linear.b"];
    auto& ffL2w = *vars["encoder:Linear.W.1"];
    auto& ffL2b = *vars["encoder:Linear.b.1"];

    // norms (default to A=1 and B=0)

    Wq.value() <<
      -1.2321, -0.4785, -0.4598, -0.1860,
       0.4576,  0.4961, -0.0903, -0.4833,
      -0.1442,  0.3495,  0.4236, -0.0846,
      -0.3082,  0.0956, -0.2470,  0.3061;
    bq.value() <<
      -1.3717, -0.1179, -0.0096, -0.4240;
    Wk.value() <<
      -2.2321, -0.4785, -0.4598, -0.1860,
       0.4576,  0.4961, -0.0903, -0.4833,
      -0.1442,  0.3495,  0.4236, -0.0846,
      -0.3082,  0.0956, -0.2470,  0.3061;
    bk.value() <<
      -2.3717, -0.1179, -0.0096, -0.4240;
    Wv.value() <<
      -3.2321, -0.4785, -0.4598, -0.1860,
       0.4576,  0.4961, -0.0903, -0.4833,
      -0.1442,  0.3495,  0.4236, -0.0846,
      -0.3082,  0.0956, -0.2470,  0.3061;
    bv.value() <<
      -3.3717, -0.1179, -0.0096, -0.4240;
    Wo.value() <<
      -4.2321, -0.4785, -0.4598, -0.1860,
       0.4576,  0.4961, -0.0903, -0.4833,
      -0.1442,  0.3495,  0.4236, -0.0846,
      -0.3082,  0.0956, -0.2470,  0.3061;
    bo.value() <<
      -4.3717, -0.1179, -0.0096, -0.4240;
    ffL1w.value() <<
      -5.4208,  0.2836, -0.1770,  0.3684,
       0.3448,  0.4124, -0.2545,  0.2874,
      -0.4372,  0.4165, -0.2362,  0.1144;
    ffL1b.value() <<
       5.2621, -0.3262,  0.4815;
    ffL2w.value() <<
      -6.3926, -0.1717,  0.2300,
       0.0701,  0.3166, -0.2458,
       0.1431, -0.3391,  0.5407,
       0.4126, -0.3719,  0.5352;
    ffL2b.value() <<
      -6.5333, -0.0515, -0.1337,  0.0297;

    /////////////////////////////////////////////////////
    // decoder weights
    /////////////////////////////////////////////////////

    // self-attention
    auto& self_attn_Wq = *vars["decoder:MHA.Wq"];
    auto& self_attn_Wk = *vars["decoder:MHA.Wk"];
    auto& self_attn_Wv = *vars["decoder:MHA.Wv"];
    auto& self_attn_Wo = *vars["decoder:MHA.Wo"];

    auto& self_attn_bq = *vars["decoder:MHA.bq"];
    auto& self_attn_bk = *vars["decoder:MHA.bk"];
    auto& self_attn_bv = *vars["decoder:MHA.bv"];
    auto& self_attn_bo = *vars["decoder:MHA.bo"];

    // cross-attention
    auto& cross_attn_Wq = *vars["decoder:MHA.Wq.1"];
    auto& cross_attn_Wk = *vars["decoder:MHA.Wk.1"];
    auto& cross_attn_Wv = *vars["decoder:MHA.Wv.1"];
    auto& cross_attn_Wo = *vars["decoder:MHA.Wo.1"];

    auto& cross_attn_bq = *vars["decoder:MHA.bq.1"];
    auto& cross_attn_bk = *vars["decoder:MHA.bk.1"];
    auto& cross_attn_bv = *vars["decoder:MHA.bv.1"];
    auto& cross_attn_bo = *vars["decoder:MHA.bo.1"];

    // element-wise feed forward
    auto& decoder_ffL1w = *vars["decoder:Linear.W"];
    auto& decoder_ffL1b = *vars["decoder:Linear.b"];
    auto& decoder_ffL2w = *vars["decoder:Linear.W.1"];
    auto& decoder_ffL2b = *vars["decoder:Linear.b.1"];

    // norms (default to A=1 and B=0)

    self_attn_Wq.value() <<
      -1.2321, -0.4785, -0.4598, -0.1860,
       0.4576,  0.4961, -0.0903, -0.4833,
      -0.1442,  0.3495,  0.4236, -0.0846,
      -0.3082,  0.0956, -0.2470,  0.3061;
    self_attn_bq.value() <<
      -1.3717, -0.1179, -0.0096, -0.4240;
    self_attn_Wk.value() <<
      -2.2321, -0.4785, -0.4598, -0.1860,
       0.4576,  0.4961, -0.0903, -0.4833,
      -0.1442,  0.3495,  0.4236, -0.0846,
      -0.3082,  0.0956, -0.2470,  0.3061;
    self_attn_bk.value() <<
      -2.3717, -0.1179, -0.0096, -0.4240;
    self_attn_Wv.value() <<
      -3.2321, -0.4785, -0.4598, -0.1860,
       0.4576,  0.4961, -0.0903, -0.4833,
      -0.1442,  0.3495,  0.4236, -0.0846,
      -0.3082,  0.0956, -0.2470,  0.3061;
    self_attn_bv.value() <<
      -3.3717, -0.1179, -0.0096, -0.4240;
    self_attn_Wo.value() <<
      -4.2321, -0.4785, -0.4598, -0.1860,
       0.4576,  0.4961, -0.0903, -0.4833,
      -0.1442,  0.3495,  0.4236, -0.0846,
      -0.3082,  0.0956, -0.2470,  0.3061;
    self_attn_bo.value() <<
      -4.3717, -0.1179, -0.0096, -0.4240;

    cross_attn_Wq.value() <<
       0.0675,  0.0034,  0.2860, -0.0438,
       0.3234,  0.4208, -0.0814, -0.0883,
      -0.3376,  0.2880,  0.0641, -0.4295,
       0.4480,  0.4328, -0.4657,  0.1207;
    cross_attn_bq.value() <<
      -0.3390,  0.0716,  0.4804, -0.4253;
    cross_attn_Wk.value() <<
       0.2975,  0.0247,  0.4618, -0.1429,
      -0.0016, -0.0542, -0.3919,  0.1051,
       0.4285,  0.0760, -0.3002, -0.2579,
      -0.1038,  0.4511,  0.4412,  0.2605;
    cross_attn_bk.value() <<
      -0.3793,  0.4552,  0.1502,  0.3554;
    cross_attn_Wv.value() <<
      -0.4192, -0.4004,  0.0120, -0.4717,
      -0.3308, -0.4728, -0.1381,  0.3374,
       0.1521, -0.1548,  0.2885,  0.4352,
      -0.1196, -0.2579, -0.3167,  0.0128;
    cross_attn_bv.value() <<
       0.4992, -0.2558,  0.1871, -0.3701;
    cross_attn_Wo.value() <<
       1.5146e-01,  5.0816e-02,  3.9053e-04, -4.6405e-01,
      -1.2832e-01, -4.3910e-01, -1.8390e-01, -5.1324e-02,
       4.4734e-01, -3.3816e-01,  1.3738e-01, -1.3041e-01,
       1.8204e-01, -2.9708e-01,  3.2434e-01, -6.3109e-02;
    cross_attn_bo.value() <<
      -0.4427, -0.0959, -0.2821, -0.2209;

    decoder_ffL1w.value() <<
      -5.4208,  0.2836, -0.1770,  0.3684,
       0.3448,  0.4124, -0.2545,  0.2874,
      -0.4372,  0.4165, -0.2362,  0.1144;
    decoder_ffL1b.value() <<
       5.2621, -0.3262,  0.4815;
    decoder_ffL2w.value() <<
      -6.3926, -0.1717,  0.2300,
       0.0701,  0.3166, -0.2458,
       0.1431, -0.3391,  0.5407,
       0.4126, -0.3719,  0.5352;
    decoder_ffL2b.value() <<
      -6.5333, -0.0515, -0.1337,  0.0297;

    /////////////////////////////////////////////////////
    // fully connected weights
    /////////////////////////////////////////////////////

    auto& fc_W = *vars["Linear.W"];
    auto& fc_b = *vars["Linear.b"];

    fc_W.value() <<
       0.4024,  0.2209, -0.3322, -0.2039,
      -0.0586, -0.3453, -0.4044,  0.3376,
      -0.4428,  0.0175, -0.4929, -0.2737,
      -0.4433, -0.2716, -0.0390,  0.4631,
      -0.0599,  0.1389,  0.0554, -0.2265,
      -0.4810, -0.2936,  0.2530, -0.0608,
       0.1361,  0.1135, -0.1584, -0.0923,
       0.3696, -0.2719, -0.0755,  0.3822,
      -0.2697,  0.1172, -0.0242,  0.4085,
      -0.2495, -0.1300,  0.2470,  0.3172;
    fc_b.value() <<
       0.1317, -0.1626, -0.0434, -0.4033,  0.0458, -0.1930,  0.3019, -0.3306,
      -0.1221,  0.3670;

    /////////////////////////////////////////////////////
    
    auto y_logit = T.forward(src_x, tgt_x);
    Tensor y_logit_hat(SEQ_SIZE, TGT_TOKENS);
    y_logit_hat <<
      -0.8373, -0.2303,  0.2102,  0.5463,  0.0835,  0.6512, -0.0579, -0.8578,
      0.6470,  1.1275,
      -0.8642, -0.1822,  0.1841,  0.5928,  0.0569,  0.6655, -0.0717, -0.8065,
      0.6609,  1.1561,
      -0.8683, -0.1421,  0.1810,  0.6232,  0.0386,  0.6596, -0.0741, -0.7732,
      0.6778,  1.1673,
      -0.8707, -0.1883,  0.1762,  0.5909,  0.0579,  0.6734, -0.0750, -0.8069,
      0.6552,  1.1593,
      -0.8581, -0.1423,  0.1925,  0.6191,  0.0411,  0.6497, -0.0688, -0.7795,
      0.6817,  1.1603;

    ASSERT(isApprox(y_logit, y_logit_hat, 0.0003))

    TEST_END()
}

void test_transformer_backward()
{
    TEST_BEGIN("Transformer Backward")

    int NUM_LAYERS = 1;
    int NUM_HEADS = 2;
    int EMB_SIZE = 4;
    int SEQ_SIZE = 5;
    int FF_SIZE = 3;
    DTYPE DROPOUT = 0.0;

    int SRC_TOKENS = 10; // input vocabulary size
    int TGT_TOKENS = 10; // output vocabulary size
    int BOS_TOKEN = 8; // begining of sequence token
    int EOS_TOKEN = 9; // end of sequence token
    int PAD_TOKEN = 0; // padding token (marks positions to ignore)

    Graph g;

    std::vector<int> src_x = {1, 2, 3, PAD_TOKEN, PAD_TOKEN}; // input sequence

    std::vector<int> tgt_x = {BOS_TOKEN, 1, 2, 3, PAD_TOKEN}; // target sequence

    std::vector<int> y_hat = {1, 2, 3, EOS_TOKEN, PAD_TOKEN}; // output sequence

    auto& T = *(new Transformer(g, SRC_TOKENS, TGT_TOKENS, PAD_TOKEN,
      NUM_LAYERS, NUM_HEADS, EMB_SIZE, FF_SIZE, SEQ_SIZE, DROPOUT));
    g.keep(&T);

    auto vars = g.named_variables();
    //print(g);

    /////////////////////////////////////////////////////
    // embedding weights
    /////////////////////////////////////////////////////

    auto& enc_emb = *vars["Embedding.E"];
    auto& dec_emb = *vars["Embedding.E.1"];

    enc_emb.value() <<
      -1.5104, -0.0284,  0.8700,  0.8670,
       1.1990,  1.1361,  0.8644,  0.1473,
      -0.7335,  0.3807,  0.7741,  0.4396,
      -0.7043,  1.6892, -0.5124,  0.4657,
       0.0162, -1.9550,  1.6194,  1.5560,
       1.3285,  0.2094, -1.5481,  0.2234,
      -0.8587, -0.4757,  0.1260, -1.9552,
      -0.7352, -0.9533, -0.7015, -1.8978,
      -0.3166,  1.9976,  0.1297,  0.9044,
       0.7586, -1.0734, -0.4338, -0.8578;
    dec_emb.value() <<
      -0.5281,  0.1697, -0.9366,  0.0129,
       0.1969, -1.1860,  0.0961, -0.6315,
      -0.1603, -0.1080,  0.1573,  0.9020,
       1.2745,  0.0302,  1.0822, -0.4542,
      -0.5246, -0.9769, -0.6335,  2.4609,
      -0.7565,  1.0006, -1.5738,  1.5421,
      -0.7897, -3.4246,  0.4188, -1.1293,
       0.9453, -0.1520, -0.7238,  0.7383,
       0.0729,  0.3859,  0.0163, -0.9825,
      -0.9148,  1.0367, -1.0432,  1.4304;

    /////////////////////////////////////////////////////
    // encoder weights
    /////////////////////////////////////////////////////

    auto& Wq = *vars["encoder:MHA.Wq"];
    auto& Wk = *vars["encoder:MHA.Wk"];
    auto& Wv = *vars["encoder:MHA.Wv"];
    auto& Wo = *vars["encoder:MHA.Wo"];

    auto& bq = *vars["encoder:MHA.bq"];
    auto& bk = *vars["encoder:MHA.bk"];
    auto& bv = *vars["encoder:MHA.bv"];
    auto& bo = *vars["encoder:MHA.bo"];

    // element-wise feed forward
    auto& ffL1w = *vars["encoder:Linear.W"];
    auto& ffL1b = *vars["encoder:Linear.b"];
    auto& ffL2w = *vars["encoder:Linear.W.1"];
    auto& ffL2b = *vars["encoder:Linear.b.1"];

    // norms (default to A=1 and B=0)

    Wq.value() <<
      -1.2321, -0.4785, -0.4598, -0.1860,
       0.4576,  0.4961, -0.0903, -0.4833,
      -0.1442,  0.3495,  0.4236, -0.0846,
      -0.3082,  0.0956, -0.2470,  0.3061;
    bq.value() <<
      -1.3717, -0.1179, -0.0096, -0.4240;
    Wk.value() <<
      -2.2321, -0.4785, -0.4598, -0.1860,
       0.4576,  0.4961, -0.0903, -0.4833,
      -0.1442,  0.3495,  0.4236, -0.0846,
      -0.3082,  0.0956, -0.2470,  0.3061;
    bk.value() <<
      -2.3717, -0.1179, -0.0096, -0.4240;
    Wv.value() <<
      -3.2321, -0.4785, -0.4598, -0.1860,
       0.4576,  0.4961, -0.0903, -0.4833,
      -0.1442,  0.3495,  0.4236, -0.0846,
      -0.3082,  0.0956, -0.2470,  0.3061;
    bv.value() <<
      -3.3717, -0.1179, -0.0096, -0.4240;
    Wo.value() <<
      -4.2321, -0.4785, -0.4598, -0.1860,
       0.4576,  0.4961, -0.0903, -0.4833,
      -0.1442,  0.3495,  0.4236, -0.0846,
      -0.3082,  0.0956, -0.2470,  0.3061;
    bo.value() <<
      -4.3717, -0.1179, -0.0096, -0.4240;
    ffL1w.value() <<
      -5.4208,  0.2836, -0.1770,  0.3684,
       0.3448,  0.4124, -0.2545,  0.2874,
      -0.4372,  0.4165, -0.2362,  0.1144;
    ffL1b.value() <<
       5.2621, -0.3262,  0.4815;
    ffL2w.value() <<
      -6.3926, -0.1717,  0.2300,
       0.0701,  0.3166, -0.2458,
       0.1431, -0.3391,  0.5407,
       0.4126, -0.3719,  0.5352;
    ffL2b.value() <<
      -6.5333, -0.0515, -0.1337,  0.0297;

    /////////////////////////////////////////////////////
    // decoder weights
    /////////////////////////////////////////////////////

    // self-attention
    auto& self_attn_Wq = *vars["decoder:MHA.Wq"];
    auto& self_attn_Wk = *vars["decoder:MHA.Wk"];
    auto& self_attn_Wv = *vars["decoder:MHA.Wv"];
    auto& self_attn_Wo = *vars["decoder:MHA.Wo"];

    auto& self_attn_bq = *vars["decoder:MHA.bq"];
    auto& self_attn_bk = *vars["decoder:MHA.bk"];
    auto& self_attn_bv = *vars["decoder:MHA.bv"];
    auto& self_attn_bo = *vars["decoder:MHA.bo"];

    // cross-attention
    auto& cross_attn_Wq = *vars["decoder:MHA.Wq.1"];
    auto& cross_attn_Wk = *vars["decoder:MHA.Wk.1"];
    auto& cross_attn_Wv = *vars["decoder:MHA.Wv.1"];
    auto& cross_attn_Wo = *vars["decoder:MHA.Wo.1"];

    auto& cross_attn_bq = *vars["decoder:MHA.bq.1"];
    auto& cross_attn_bk = *vars["decoder:MHA.bk.1"];
    auto& cross_attn_bv = *vars["decoder:MHA.bv.1"];
    auto& cross_attn_bo = *vars["decoder:MHA.bo.1"];

    // element-wise feed forward
    auto& decoder_ffL1w = *vars["decoder:Linear.W"];
    auto& decoder_ffL1b = *vars["decoder:Linear.b"];
    auto& decoder_ffL2w = *vars["decoder:Linear.W.1"];
    auto& decoder_ffL2b = *vars["decoder:Linear.b.1"];

    // norms (default to A=1 and B=0)

    self_attn_Wq.value() <<
      -1.2321, -0.4785, -0.4598, -0.1860,
       0.4576,  0.4961, -0.0903, -0.4833,
      -0.1442,  0.3495,  0.4236, -0.0846,
      -0.3082,  0.0956, -0.2470,  0.3061;
    self_attn_bq.value() <<
      -1.3717, -0.1179, -0.0096, -0.4240;
    self_attn_Wk.value() <<
      -2.2321, -0.4785, -0.4598, -0.1860,
       0.4576,  0.4961, -0.0903, -0.4833,
      -0.1442,  0.3495,  0.4236, -0.0846,
      -0.3082,  0.0956, -0.2470,  0.3061;
    self_attn_bk.value() <<
      -2.3717, -0.1179, -0.0096, -0.4240;
    self_attn_Wv.value() <<
      -3.2321, -0.4785, -0.4598, -0.1860,
       0.4576,  0.4961, -0.0903, -0.4833,
      -0.1442,  0.3495,  0.4236, -0.0846,
      -0.3082,  0.0956, -0.2470,  0.3061;
    self_attn_bv.value() <<
      -3.3717, -0.1179, -0.0096, -0.4240;
    self_attn_Wo.value() <<
      -4.2321, -0.4785, -0.4598, -0.1860,
       0.4576,  0.4961, -0.0903, -0.4833,
      -0.1442,  0.3495,  0.4236, -0.0846,
      -0.3082,  0.0956, -0.2470,  0.3061;
    self_attn_bo.value() <<
      -4.3717, -0.1179, -0.0096, -0.4240;

    cross_attn_Wq.value() <<
       0.0675,  0.0034,  0.2860, -0.0438,
       0.3234,  0.4208, -0.0814, -0.0883,
      -0.3376,  0.2880,  0.0641, -0.4295,
       0.4480,  0.4328, -0.4657,  0.1207;
    cross_attn_bq.value() <<
      -0.3390,  0.0716,  0.4804, -0.4253;
    cross_attn_Wk.value() <<
       0.2975,  0.0247,  0.4618, -0.1429,
      -0.0016, -0.0542, -0.3919,  0.1051,
       0.4285,  0.0760, -0.3002, -0.2579,
      -0.1038,  0.4511,  0.4412,  0.2605;
    cross_attn_bk.value() <<
      -0.3793,  0.4552,  0.1502,  0.3554;
    cross_attn_Wv.value() <<
      -0.4192, -0.4004,  0.0120, -0.4717,
      -0.3308, -0.4728, -0.1381,  0.3374,
       0.1521, -0.1548,  0.2885,  0.4352,
      -0.1196, -0.2579, -0.3167,  0.0128;
    cross_attn_bv.value() <<
       0.4992, -0.2558,  0.1871, -0.3701;
    cross_attn_Wo.value() <<
       1.5146e-01,  5.0816e-02,  3.9053e-04, -4.6405e-01,
      -1.2832e-01, -4.3910e-01, -1.8390e-01, -5.1324e-02,
       4.4734e-01, -3.3816e-01,  1.3738e-01, -1.3041e-01,
       1.8204e-01, -2.9708e-01,  3.2434e-01, -6.3109e-02;
    cross_attn_bo.value() <<
      -0.4427, -0.0959, -0.2821, -0.2209;

    decoder_ffL1w.value() <<
      -5.4208,  0.2836, -0.1770,  0.3684,
       0.3448,  0.4124, -0.2545,  0.2874,
      -0.4372,  0.4165, -0.2362,  0.1144;
    decoder_ffL1b.value() <<
       5.2621, -0.3262,  0.4815;
    decoder_ffL2w.value() <<
      -6.3926, -0.1717,  0.2300,
       0.0701,  0.3166, -0.2458,
       0.1431, -0.3391,  0.5407,
       0.4126, -0.3719,  0.5352;
    decoder_ffL2b.value() <<
      -6.5333, -0.0515, -0.1337,  0.0297;

    /////////////////////////////////////////////////////
    // fully connected weights
    /////////////////////////////////////////////////////

    auto& fc_W = *vars["Linear.W"];
    auto& fc_b = *vars["Linear.b"];

    fc_W.value() <<
       0.4024,  0.2209, -0.3322, -0.2039,
      -0.0586, -0.3453, -0.4044,  0.3376,
      -0.4428,  0.0175, -0.4929, -0.2737,
      -0.4433, -0.2716, -0.0390,  0.4631,
      -0.0599,  0.1389,  0.0554, -0.2265,
      -0.4810, -0.2936,  0.2530, -0.0608,
       0.1361,  0.1135, -0.1584, -0.0923,
       0.3696, -0.2719, -0.0755,  0.3822,
      -0.2697,  0.1172, -0.0242,  0.4085,
      -0.2495, -0.1300,  0.2470,  0.3172;
    fc_b.value() <<
       0.1317, -0.1626, -0.0434, -0.4033,  0.0458, -0.1930,  0.3019, -0.3306,
      -0.1221,  0.3670;

    /////////////////////////////////////////////////////

    Tensor y = T.forward(src_x, tgt_x);
    T.gradient() = Tensor::Ones(SEQ_SIZE, TGT_TOKENS);
    T.gradient().leftCols(1) = Tensor::Constant(SEQ_SIZE,1, 1250);

    Tensor dTdenc_emb = enc_emb.backward();
    Tensor dTddec_emb = dec_emb.backward();

    Tensor dTdenc_emb_hat(SRC_TOKENS, EMB_SIZE);
    dTdenc_emb_hat <<
       0.0000,  0.0000,  0.0000,  0.0000,
       0.4342,  1.0443, -2.2485, -0.0814,
       0.6311,  1.0999, -2.1718, -0.0151,
       0.4310,  1.2264, -2.1344,  0.0286,
       0.0000,  0.0000,  0.0000,  0.0000,
       0.0000,  0.0000,  0.0000,  0.0000,
       0.0000,  0.0000,  0.0000,  0.0000,
       0.0000,  0.0000,  0.0000,  0.0000,
       0.0000,  0.0000,  0.0000,  0.0000,
       0.0000,  0.0000,  0.0000,  0.0000;

    Tensor dTddec_emb_hat(TGT_TOKENS, EMB_SIZE);
    dTddec_emb_hat <<
      3.4459,  14.6792, -14.8203,  -2.4398,
     23.8857,   9.9135, -12.1941,  -4.5663,
     11.0370,  10.6798, -13.9681,  -3.3901,
     15.5422,   9.4706,  -7.8664,  -2.6836,
      0.0000,   0.0000,   0.0000,   0.0000,
      0.0000,   0.0000,   0.0000,   0.0000,
      0.0000,   0.0000,   0.0000,   0.0000,
      0.0000,   0.0000,   0.0000,   0.0000,
     -8.6914,  20.0081, -29.9860, -17.2931,
      0.0000,   0.0000,   0.0000,   0.0000;

    ASSERT(isApprox(dTdenc_emb, dTdenc_emb_hat, 0.0005))
    ASSERT(isApprox(dTddec_emb, dTddec_emb_hat, 0.0005))

    TEST_END()
}

int main(int argc, char* argv[]) {

    test_cnpy();
    //test_threads();
    //test_thread_pool();

    test_sequence_mask();
    test_scaled_dot_product_attention_forward();
    test_scaled_dot_product_attention_backward();
    test_multihead_attention_forward();
    test_multihead_attention_backward();
    test_position_wise_ff_forward();
    test_position_wise_ff_backward();
    test_positional_encoding_forward();
    test_positional_encoding_backward();
    test_encoder_layer_forward();
    test_encoder_layer_backward();
    test_decoder_layer_forward();
    test_decoder_layer_backward();
    test_transformer_forward();
    test_transformer_backward();

    return 0;
}
