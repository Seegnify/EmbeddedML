/*
 * Copyright 2020-2024 Greg Padiasek. All Rights Reserved.
 */

#include "graph/optimizer.hh"
#include "utils/training.hh"
#include "utils/storage.hh"
#include "external/cnpy/cnpy.h"

#include "transformer.hh"

/* Standard Transfomer options */
#define NUM_LAYERS 6    // number of encoder/decoder layers
#define NUM_HEADS 8     // numberof multiheads
#define EMB_SIZE 512    // size of embeddings
#define SEQ_SIZE 100    // max seqeunce size
#define FF_SIZE 2048    // size of feed forward hidden layer
#define DROPOUT 0.1     // dropout probability

/* Test Transfomer options */
/*
#define NUM_LAYERS 2    // number of encoder/decoder layers
#define NUM_HEADS 2     // numberof multiheads
#define EMB_SIZE 64     // size of embeddings
#define SEQ_SIZE 48     // max seqeunce size
#define FF_SIZE 16      // size of feed forward hidden layer
#define DROPOUT 0.0     // dropout probability
*/

#define SRC_TOKENS 128  // input vocabulary size
#define TGT_TOKENS 128  // output vocabulary size
#define BOS_TOKEN 1     // begining of sequence token
#define EOS_TOKEN 2     // end of sequence token
#define PAD_TOKEN 3     // padding token (marks positions to ignore)

/////////////////////////////////////
// training instance implementation
/////////////////////////////////////
class TransformerClient : public Training
{
public:
  TransformerClient(int worker) : Training(worker)
  {
    std::cout << "TRANSFORMER training " << worker << std::endl;
    
    // load data
    _train_data.push_back(
      {
        "Hi, how is it going?",
        "Czesc, co u ciebie?"
      }
    );
    _train_data.push_back(
      {
        "When is your next train?",
        "Kiedy jest twoj nastepny pociag?",
      }
    );
    _train_data.push_back(
      {
        "I like chocolate.",
        "Lubie czekolade.",
      }
    );

    // init train batch sequence
    for (int i=0; i<_train_data.size(); i++) _train_batch.push_back(i);

    Graph& g = graph();

    // model
    _model = new Transformer(g, SRC_TOKENS, TGT_TOKENS, PAD_TOKEN,
      NUM_LAYERS, NUM_HEADS, EMB_SIZE, FF_SIZE, SEQ_SIZE, DROPOUT);
    g.keep(_model);

    // predicted output
    auto y = g.new_rowwise(*_model, SEQ_SIZE, TGT_TOKENS,
      [&](Function& row) { return g.new_log_softmax(row); });
    g.name(y, "softmax_y");

    // expected output
    _y_hat = new SequenceMask(g, PAD_TOKEN, SEQ_SIZE);
    g.keep(_y_hat);
    
    // loss
    _loss = g.new_sum(- *_y_hat * *y); // cross entropy

    // optimizer
    _optimizer = new Adam(g.variables(), 0.0001);
    //_optimizer = new SGD(g.variables(), 0.1, 0.1);

    // counters
    _batch = 0;
  }

  ~TransformerClient()
  {
    delete _optimizer;
  }

  // print variables
  void print(bool values = false, bool grads = false)
  {
    auto vars = graph().named_variables();

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
        std::cout << "value norm: " << tensor.norm() << std::endl;
        //std::cout << tensor << std::endl;
      }
      if (grads)
      {
        std::cout << "gradient norm:" << var->gradient().norm() << std::endl;
        //std::cout << var->gradient() << std::endl;
      }
    }
  }

  // tokenize and pad source sequence
  std::vector<int> source_tokens(const std::string& text)
  {
    std::vector<int> tokens(SEQ_SIZE, PAD_TOKEN);

    auto size = std::min<int>(text.size(), SEQ_SIZE);
    for (int i=0; i<size; i++) tokens[i] = text[i];

    return tokens;
  }

  // tokenize and pad target sequence
  std::vector<int> target_tokens(const std::string& text)
  {
    std::vector<int> tokens(SEQ_SIZE, PAD_TOKEN);

    tokens[0] = BOS_TOKEN; // BOS as first token

    auto size = std::min<int>(text.size(), SEQ_SIZE - 1); // 1 for BOS
    for (int i=0; i<size; i++) tokens[i + 1] = text[i]; // 1 for BOS

    return tokens;
  }

  // generate text from token sequence
  std::string output_text(const std::vector<int>& out)
  {
    std::string text;

    for (auto t: out)
    {
      if (t == EOS_TOKEN || t == PAD_TOKEN) break;
      text.push_back(t);
    }

    return text;
  }

  // generate output mask for training loss
  Tensor output_mask(const std::vector<int>& tgt)
  {
    Tensor mask = Tensor::Zero(SEQ_SIZE, TGT_TOKENS);

    auto it = std::find(tgt.begin(), tgt.end(), PAD_TOKEN);
    size_t tgt_size = ((it != tgt.end()) ? it - tgt.begin() : tgt.size()) - 1;

    // one-hot token
    for (int i=0; i<tgt_size; i++)
    {
      mask(i, tgt[i + 1]) = 1; // i + 1 for BOS
    }

    // one-hot EOS
    mask(tgt_size, EOS_TOKEN) = 1;
    return mask;
  }

  void save_numpy(const std::string& filepath)
  {
    int count = 0;
    auto vars = graph().named_variables();

    for (const auto& it: vars)
    {
      auto name = it.first;
      auto var = it.second;
      auto data = var->value().data();
      size_t rows = var->value().rows();
      size_t cols = var->value().cols();

      std::string mode = (count) ? "a" : "w"; // a to append, w to create
      if (rows == 1)
      {
        cnpy::npz_save(filepath, name, data, {cols}, mode);
      }
      else
      {
        cnpy::npz_save(filepath, name, data, {rows, cols}, mode);
      }

      count+=1;
    }
  }

  virtual void batch_train()
  {
    if (worker() == 0 && _batch == 0)
    {
      save_numpy("transformer-cpp.npz");
    }

    // restore references
    auto& g = graph();
    auto& opt = *_optimizer;

    int batch_size = _train_data.size();
    _batch++;

    // select training batch
    g.random().shuffle(_train_batch.begin(), _train_batch.end());

    // batch train
    for (int i=0; i<batch_size; i++)
    {
      g.recache();

      // training sample
      auto& x = _train_data[_train_batch[i]];
      auto src_x = source_tokens(x.first);
      auto tgt_x = target_tokens(x.second);

      // loss mask
      _y_hat->value() = output_mask(tgt_x);

      // set model inputs
      auto& y = _model->forward(src_x, tgt_x);

      // traning loss
      auto& loss = _loss->forward();
      if (worker() == 0)
      {
        std::cout << "time:" << time_to_string(time_now())
                  << " loss:" << loss
                  << std::endl;
      }

      // backward pass from loss
      g.backward(*_loss, loss.array().min(Tensor::Ones(1,1).array()));
    }

    /*
    if (worker() == 0)
    {
      std::cout << "model variables:" << std::endl;
      print(true, true);
    }
    */

    // update weights
    opt.update();
    g.zero_grad();

    int valid_step = 1;
    if (_batch % valid_step == 0 and worker() == 0)
    {
      std::cout << "validation success "
                << validate(g)
                << std::endl;
    }
  }

  float validate(Graph& g)
  {
    float success = 0;

    int size = _train_data.size();

    for (int i=0; i<size; i++)
    {
      g.recache();

      // validation sample
      // TODO: change data to validation
      auto& x = _train_data[i];
      auto src_x = source_tokens(x.first);
      auto tgt_x = target_tokens(x.second);

      // generate output
      auto y = _model->generate(src_x, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN);

      // compare output with target
      auto y_txt = output_text(y);
      success += (y_txt == x.second);

      if (y_txt != x.second)
      {
        std::cout << "Source:[" << x.first << "]" << std::endl;
        std::cout << "Target:[" << x.second << "]" << std::endl;
        std::cout << "Output:[" << y_txt << "]" << std::endl;

        std::cout << "First Target Token:" << "First Tgt=" << tgt_x.front() << std::endl;
        for (int i=0; i<SEQ_SIZE-1; i++)
        {
          auto diff = (tgt_x[i+1] == y[i])? " (==)" : " (!=)";
          std::cout << "Tgt["<< i+1 << "]=" << tgt_x[i+1]
                    << " Out[" << i << "]=" << y[i]
                    << diff << std::endl;
        }
        std::cout << "Last Output Token:" << "Last Out=" << y.back() << std::endl;
      }
    }

    return success / size;
  }

private:
  std::vector<std::pair<std::string, std::string>> _train_data;
  std::vector<int> _train_batch;
  Transformer *_model;
  Optimizer *_optimizer;
  SequenceMask* _y_hat;
  Function* _loss;
  int _batch;
};

///////////////////////////////////
extern "C" { // export C signatures
///////////////////////////////////

DLL_EXPORT Training* create(int idx)
{
  return new TransformerClient(idx);
}

DLL_EXPORT void destroy(Training* ptr)
{
  delete (TransformerClient*)ptr;
}

///////////////////////////////////
} // export C signatures
///////////////////////////////////

