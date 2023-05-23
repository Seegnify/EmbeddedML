#ifndef _SEEGNIFY_TYPES_H_
#define _SEEGNIFY_TYPES_H_

#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace seegnify {

// computation precision type
#define DTYPE float

// differential delta (lower is more accurate)
#define FINITE_DELTA 1e-3

// epsilon neighborhood
#define EPSILON 1e-6

// learning rate (lower is slower but more stable)
#define LEARNING_RATE 1e-2

// reinforcement learning reward discount
#define GAMMA_DISCOUNT 0.99

// Eigen types
typedef Eigen::SparseMatrix<DTYPE, Eigen::ColMajor, int32_t> SparseTensor;
typedef Eigen::Map<SparseTensor> SparseTensorMap;
typedef Eigen::Map<const SparseTensor> ConstSparseTensorMap;

typedef Eigen::Matrix<DTYPE, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> Tensor;
typedef Eigen::Map<Tensor> TensorMap;
typedef Eigen::Map<const Tensor> ConstTensorMap;

typedef Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> TensorXi;
typedef Eigen::Map<TensorXi> TensorXiMap;
typedef Eigen::Map<const TensorXi> ConstTensorXiMap;

typedef Eigen::Matrix<DTYPE, Eigen::Dynamic, 1, Eigen::ColMajor> Vector;
typedef Eigen::Map<Vector> VectorMap;
typedef Eigen::Map<const Vector> ConstVectorMap;

typedef Eigen::Matrix<DTYPE, 1, Eigen::Dynamic, Eigen::RowMajor> RowVector;
typedef Eigen::Map<RowVector> RowVectorMap;
typedef Eigen::Map<const RowVector> ConstRowVectorMap;

// Export symbols
#ifndef DLL_EXPORT
#define DLL_EXPORT __attribute__ ((visibility ("default")))
#endif

#ifndef DLL_EXPORT
#define DLL_LOCAL  __attribute__ ((visibility ("hidden")))
#endif

} /* namespace */

#endif /*_SEEGNIFY_TYPES_H_*/
