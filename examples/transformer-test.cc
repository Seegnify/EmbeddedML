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

void test_attention_forward()
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

    Attention attn(g, *Q,*K,*V, M, T, S, D, dropout);

    auto& A = attn();
    //print("A", A);


    Tensor A_torch(2,5);
    A_torch <<  3.9070,  4.9059,  5.8955,  4.9668,  4.9668,
                3.5844,  1.4156, -7.8225,  2.9307,  2.9307;

    ASSERT(A_torch.isApprox(A, 0.00001))

    TEST_END()
}

void test_attention_backward()
{
    TEST_BEGIN("Scaled Dot-Product Attention Backward")

    Graph g;
    DTYPE dropout = 0.0;

    // [2x3]
    auto Q = g.new_variable(2, 3);
    Q->value() << 1,2,3,
                  4,5,6;
    //print("Q", *Q);

    // [4x3]
    auto K = g.new_variable(4, 3);
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
    auto V = g.new_variable(4, 5);
    V->value() << -2,7,8,2,2,
                  4,1,-9,3,3,
                  1,2,3,4,4,
                  4,5,6,5,5;
    //print("V", *V);

    // QK_TV -> [2x5]

    int T = Q->value().rows();
    int S = K->value().rows();
    int D = K->value().cols();

    Attention attn(g, *Q,*K,*V, M, T, S, D, dropout);

    auto& A = attn();
    //print("A", A);

    attn.forward();
    attn.gradient() = Tensor::Ones(attn.forward().rows(),attn.forward().cols());
    attn.gradient().block(0,0, attn.forward().rows(), 1) = 
      5 * Tensor::Ones(attn.forward().rows(), 1);
    //print("dA", attn.gradient());

    Tensor dQ = Q->backward();
    Tensor dK = K->backward();
    Tensor dV = V->backward();
    //print("dQ", dQ);
    //print("dK", dK);
    //print("dV", dV);

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

    int trg_size = 3;
    int seq_size = 3;
    int emb_size = 4;
    int num_heads = 2;
    bool bias = true;
    DTYPE dropout = 0.0;

    auto q = g.new_variable(trg_size, emb_size);
    auto k = g.new_variable(seq_size, emb_size);
    auto v = g.new_variable(trg_size, emb_size);
    auto o = g.new_constant(trg_size, emb_size);

    MultiHeadAttention mha(
      g, *q, *k, *v,
      trg_size, seq_size, emb_size, num_heads,
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
    o->value() <<
      0.5673, 0.5978, 0.1625, 0.9602,
      0.3483, 0.4360, 0.5943, 0.8270,
      0.4020, 0.5694, 0.5695, 0.9832;

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

    Tensor mha_hat(trg_size, emb_size);
    mha_hat <<   
        0.4363, 0.5356, 0.4469, 0.9232,
        0.4293, 0.5403, 0.4531, 0.9223,
        0.4353, 0.5364, 0.4456, 0.9224;

    print("MHA_hat", mha_hat);
    print("MHA", mha);

    ASSERT(mha().isApprox(mha_hat, 0.0001))

    TEST_END()
}

int main(int argc, char* argv[]) {

    //test_cnpy();
    //test_threads();
    //test_thread_pool();
    //test_attention_forward();
    //test_attention_backward();
    test_multihead_attention_forward();

    return 0;
}
