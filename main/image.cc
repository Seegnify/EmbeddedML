#include "image.hh"

#include <sstream>
#include <cstring>

#include <Magick++.h>

namespace seegnify {

void Image::load(const std::string& path)
{
  clear();

  Magick::Image image;
  image.read(path);

  _rows = image.rows();
  _cols = image.columns();
  _bits_per_pixel = image.depth();

  Magick::Blob blob;

  switch(_bits_per_pixel)
  {
    case 8:
    {
      image.magick("R");
      image.type(Magick::GrayscaleType);
      image.write(&blob);
    }
    case 16:
    {
      image.magick("R");
      image.type(Magick::GrayscaleType);
      image.depth(16);
      image.write(&blob);
    }
    case 24:
    {
      image.magick("RGB");
      image.write(&blob);
    }
    case 32:
    {
      image.magick("RGBA");
      image.write(&blob);
    }
  };

  _data = new uint8_t[blob.length()];
  std::copy_n((uint8_t*)blob.data(), blob.length(), (uint8_t*)_data);
}

void Image::save(const std::string& path)
{
  switch(_bits_per_pixel)
  {
    case 8:
    {
      Magick::Image image(_cols, _rows, "R", Magick::CharPixel, _data);
      image.type(Magick::GrayscaleType);
      image.write(path);
      break;
    }
    case 16:
    {
      Magick::Image image(_cols, _rows, "R", Magick::CharPixel, _data);
      image.type(Magick::GrayscaleType);
      image.depth(16);
      image.write(path);
      break;
    }
    case 24:
    {
      Magick::Image image(_cols, _rows, "RGB", Magick::CharPixel, _data);
      image.write(path);
      break;
    }
    case 32:
    {
      Magick::Image image(_cols, _rows, "RGBA", Magick::CharPixel, _data);
      image.write(path);
      break;
    }
    default:
    {
      std::ostringstream msg;
      msg << "Unsupported image depth " << _bits_per_pixel;
      throw std::runtime_error(msg.str());
    }
  };
}

Image Image::crop(int row, int col, uint32_t rows, uint32_t cols)
{
  Image im(rows,  cols, _bits_per_pixel);

  auto bytes_per_pixel = _bits_per_pixel / 8;

  std::memset(im._data, 0, rows * cols * bytes_per_pixel);

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
        auto offset = bytes_per_pixel * (r * cols + c);
        auto this_offset = bytes_per_pixel * (this_r * _cols + this_c);

        // copy pixel bytes
        for (int b=0; b<bytes_per_pixel; b++)
        {
          im._data[offset + b] = _data[this_offset + b];
        }
      }
    }
  }

  return im;
}

Image Image::scale(uint32_t rows, uint32_t cols,
Interpolation interp)
{
  switch(interp)
  {
    case INTERPOLATE_NEAREST:
      return scale_nearest(rows, cols);
    case INTERPOLATE_BILINEAR:
      return scale_bilinear(rows, cols);
  }
}

Image Image::scale_nearest(uint32_t rows, uint32_t cols)
{
  Image im(rows,  cols, _bits_per_pixel);

  auto scale_r = (float)(_rows - 1)/(rows - 1);
  auto scale_c = (float)(_cols - 1)/(cols - 1);
  auto bytes_per_pixel = _bits_per_pixel / 8;

  for (auto r=0; r<rows; r++)
  for (auto c=0; c<cols; c++)
  {
    uint32_t this_c = scale_c * c;
    uint32_t this_r = scale_r * r;

    auto offset = bytes_per_pixel * (r * cols + c);
    auto this_offset = bytes_per_pixel * (this_r * _cols + this_c);

    // copy pixel bytes
    for (int b=0; b<bytes_per_pixel; b++)
    {
      im._data[offset + b] = _data[this_offset + b];
    }
  }

  return im;
}

Image Image::scale_bilinear(uint32_t rows, uint32_t cols)
{
  Image im(rows,  cols, _bits_per_pixel);

  auto scale_r = (float)(_rows - 1)/(rows - 1);
  auto scale_c = (float)(_cols - 1)/(cols - 1);
  auto bytes_per_pixel = _bits_per_pixel / 8;

  for (auto r=0; r<rows; r++)
  for (auto c=0; c<cols; c++)
  {
    auto float_r = scale_r * r;
    auto float_c = scale_c * c;

    uint32_t this_r0 = float_r;
    uint32_t this_c0 = float_c;

    uint32_t this_r1 = std::min<uint32_t>(this_r0 + 1, _rows - 1); 
    uint32_t this_c1 = std::min<uint32_t>(this_c0 + 1, _cols - 1); 

    auto r0_c0 = bytes_per_pixel * (this_r0 * _cols + this_c0);
    auto r1_c0 = bytes_per_pixel * (this_r1 * _cols + this_c0);
    auto r0_c1 = bytes_per_pixel * (this_r0 * _cols + this_c1);
    auto r1_c1 = bytes_per_pixel * (this_r1 * _cols + this_c1);

    float sr = float_r - this_r0;
    float sc = float_c - this_c0;

    auto offset = bytes_per_pixel * (r * cols + c);

    // interpolate pixel bytes
    for (int b=0; b<bytes_per_pixel; b++)
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
