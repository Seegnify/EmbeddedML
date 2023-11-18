#include <iostream>
#include <vector>
#include <thread>
#include <cmath>

#include <iostream>

#include "external/cnpy/cnpy.h"
#include "external/thread-pool-11/ThreadPool.h"

#include "examples/transformer.hh"

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

void test_transformer()
{
    /*
    Q = torch.Tensor([
        [1,2,3],
        [4,5,6],
    ])
    K = torch.Tensor([
        [0.1,0.2,0.3],
        [0.4,0.5,0.6],
    ])
    V = torch.Tensor([
        [-2,7,8],
        [4,1,-9],
    ])
    M = None
    */

    Graph g;
    int S = 2;
    int D = 3;
    DTYPE dropout = 0.0;

    auto Q = g.new_constant(S, D);
    Q->value() <<  1,2,3,
                  4,5,6;
    print("Q", *Q);

    auto K = g.new_constant(S, D);
    K->value() <<  0.1,0.2,0.3,
                  0.4,0.5,0.6;
    print("K", *K);

    auto V = g.new_constant(S, D);
    V->value() <<  -2,7,8,
                  4,1,-9;
    print("V", *V);

    auto M = g.new_constant(S, D);
    M->value() <<  1,1,1,
                  1,1,1;
    print("M", *M);
    M = nullptr;

    Attention attn(g, *Q,*K,*V,M, dropout);

    auto A = attn();
    print("A", A);
}

int main(int argc, char* argv[]) {

    //test_cnpy();
    //test_threads();
    //test_thread_pool();
    test_transformer();

    return 0;
}
