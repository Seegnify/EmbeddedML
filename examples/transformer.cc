/*
 * Copyright 2020-2023 Greg Padiasek. All Rights Reserved.
 */

#include "main/training.hh"
#include "main/optimizer.hh"
#include "main/storage.hh"

#include "transformer.hh"


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
    // TODO:

    // create graph
    Graph& g = graph();
    _model = new Transformer(g);

    // optimizer
    _optimizer = new Adam(g.variables(), 0.0001); // linear reg 0.87 good but slow
    //_optimizer = new SGD(g.variables(), 0.00001, 0.5); // linear reg 0.87 good and fast

    // counters
    _batch = 0;
  }

  ~TransformerClient()
  {
    delete _model;
    delete _optimizer;
  }

  virtual void batch_train()
  {
    // restore references
    auto& g = graph();
    auto& opt = *_optimizer;

    int batch_size = 100;
    _batch++;
    float success = 0;

    // batch train
    for (int i=0; i<batch_size; i++)
    {
      g.recache();

      // generator
      //success += _model->forward(image, label, true);
      //_model->backward(image);
    }

    // update weights
    opt.update();
    g.zero_grad();

    int valid_step = 100;
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

    int size = 0;

    for (int i=0; i<size; i++)
    {
      g.recache();

      // generator
      //success += _model->forward(image, label, false);
    }

    return success / size;
  }

private:
  Transformer *_model;
  Optimizer *_optimizer;
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

