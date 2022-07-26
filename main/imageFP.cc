#include <sstream>
#include <fstream>
#include <cstring>
#include <cmath>
#include <limits>

#include "imageFP.hh"

namespace seegnify {

//////////////////////////////////////////////////////////////////////////////
// FP Header
//////////////////////////////////////////////////////////////////////////////

struct FPHeader {
  unsigned int version = 1;
  unsigned int rows = 0;
  unsigned int cols = 0;
  unsigned int channels = 0;
  unsigned int imagesize = 0;
  unsigned int filesize = 0;
};

//////////////////////////////////////////////////////////////////////////////
// Floating Point Image
//////////////////////////////////////////////////////////////////////////////

void ImageFP::set(uint32_t row, uint32_t col, uint8_t channel, DTYPE value)
{
  const size_t index = _channels * (row * _cols + col);
  _data[index + channel] = value;
}

DTYPE ImageFP::get(uint32_t row, uint32_t col, uint8_t channel) const
{
  const size_t index = _channels * (row * _cols + col);
  return _data[index + channel];
}

ImageFP::Status ImageFP::load(const std::string& filename)
{
  clear();

  // Default header
  FPHeader header;

  // Open the image file in binary mode
  std::ifstream f_img(filename.c_str(), std::ios::binary);

  if (!f_img.is_open())
    return STATUS_FILE_NOT_OPENED;

  // Read the header structure into header
  f_img.read(reinterpret_cast<char*>(&header), sizeof (header));

  if (header.version != 1)
  {
    f_img.close();
    return STATUS_INVALID_FILE;    
  }

  unsigned int rows = header.rows;
  unsigned int cols = header.cols;
  unsigned int channels = header.channels;

  if (rows * cols * channels != header.imagesize)
  {
    f_img.close();
    return STATUS_INVALID_FILE;    
  }

  init(rows, cols, channels);

  unsigned int data_size = header.imagesize * sizeof(DTYPE);
  f_img.read(reinterpret_cast<char*>(_data), data_size);

  f_img.close();
  return STATUS_OK;
}

ImageFP::Status ImageFP::save(const std::string& filename) const
{
  // Init header
  FPHeader header;
  header.rows = _rows;
  header.cols = _cols;
  header.channels = _channels;
  header.imagesize = size();
  header.filesize = header.imagesize * sizeof(DTYPE) + sizeof(header);

  // Open the image file in binary mode
  std::ofstream f_img(filename.c_str(), std::ios::binary);

  if (!f_img.is_open())
    return STATUS_FILE_NOT_OPENED;

  unsigned int data_size = header.imagesize * sizeof(DTYPE);
  f_img.write(reinterpret_cast<const char*>(&header), sizeof(header));
  f_img.write(reinterpret_cast<const char*>(_data), data_size);

  // NOTE: All good
  f_img.close();
  return STATUS_OK;  
}

ImageFP ImageFP::crop(uint32_t row, uint32_t col, uint32_t rows, uint32_t cols) const
{
  ImageFP im(rows,  cols, _channels);

  std::memset(im._data, 0, rows * cols * _channels);

  for (int r=0; r<rows; r++)
  {
    for (int c=0; c<cols; c++)
    {
      auto this_r = r + row;
      auto this_c = c + col;

      if (this_r >= 0 and this_r < _rows)
      if (this_c >= 0 and this_c < _cols)
      {
        // compute array offset given row order
        auto offset = _channels * (r * cols + c);
        auto this_offset = _channels * (this_r * _cols + this_c);

        // copy channel values
        for (int b=0; b<_channels; b++)
        {
          im._data[offset + b] = _data[this_offset + b];
        }
      }
    }
  }

  return im;
}

ImageFP ImageFP::norm(DTYPE range) const
{
  ImageFP im(_rows,  _cols, _channels);

  auto this_data = data();

  DTYPE max_value = std::numeric_limits<DTYPE>::min();
  DTYPE min_value = std::numeric_limits<DTYPE>::max();

  for (int i=size(); i>0; i--)
  {
    max_value = std::max<DTYPE>(this_data[i-1], max_value);
    min_value = std::min<DTYPE>(this_data[i-1], min_value);
  }

  if (std::abs(max_value - min_value) < EPSILON)
  {
    max_value = range;
    min_value = 0.0;
  }

  auto scale = range / (max_value - min_value);

  auto im_data = im.data();

  for (int i=size(); i>0; i--)
  {
    im_data[i-1] = scale * (this_data[i-1] - min_value);
  }

  return im;
}

ImageFP ImageFP::scale(uint32_t rows, uint32_t cols, Interpolation interp) const
{
  switch(interp)
  {
    case INTERPOLATE_NEAREST:
      return scale_nearest(rows, cols);
    case INTERPOLATE_BILINEAR:
      return scale_bilinear(rows, cols);
  }
}

ImageFP ImageFP::scale_nearest(uint32_t rows, uint32_t cols) const
{
  ImageFP im(rows,  cols, _channels);

  auto scale_r = (float)(_rows - 1)/(rows - 1);
  auto scale_c = (float)(_cols - 1)/(cols - 1);

  for (auto r=0; r<rows; r++)
  for (auto c=0; c<cols; c++)
  {
    uint32_t this_c = std::round(scale_c * c);
    uint32_t this_r = std::round(scale_r * r);

    auto offset = _channels * (r * cols + c);
    auto this_offset = _channels * (this_r * _cols + this_c);

    // copy pixel bytes
    for (int b=0; b<_channels; b++)
    {
      im._data[offset + b] = _data[this_offset + b];
    }
  }

  return im;
}

ImageFP ImageFP::scale_bilinear(uint32_t rows, uint32_t cols) const
{
  ImageFP im(rows,  cols, _channels);

  auto scale_r = (float)(_rows - 1)/(rows - 1);
  auto scale_c = (float)(_cols - 1)/(cols - 1);

  for (auto r=0; r<rows; r++)
  for (auto c=0; c<cols; c++)
  {
    auto float_r = scale_r * r;
    auto float_c = scale_c * c;

    uint32_t this_r0 = float_r;
    uint32_t this_c0 = float_c;

    uint32_t this_r1 = std::min<uint32_t>(this_r0 + 1, _rows - 1); 
    uint32_t this_c1 = std::min<uint32_t>(this_c0 + 1, _cols - 1); 

    auto r0_c0 = _channels * (this_r0 * _cols + this_c0);
    auto r1_c0 = _channels * (this_r1 * _cols + this_c0);
    auto r0_c1 = _channels * (this_r0 * _cols + this_c1);
    auto r1_c1 = _channels * (this_r1 * _cols + this_c1);

    float sr = float_r - this_r0;
    float sc = float_c - this_c0;

    auto offset = _channels * (r * cols + c);

    // interpolate pixel bytes
    for (int b=0; b<_channels; b++)
    {
      // linear along rows
      auto col_r0 = (1 - sr) * _data[r0_c0 + b] + sr * _data[r1_c0 + b];
      auto col_r1 = (1 - sr) * _data[r0_c1 + b] + sr * _data[r1_c1 + b];

      // linear along cols
      im._data[offset + b] = (1 - sc) * col_r0 + sc * col_r1;
    }
  }

  return im;
}

} /* namespace */
