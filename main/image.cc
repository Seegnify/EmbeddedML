#include "image.hh"

#include <sstream>
#include <fstream>
#include <cstring>
#include <cmath>
#include <limits>

namespace seegnify {

#define BMP_MAGIC                 19778
#define BMP_PADDING(cols, bpp)    ((4 - (((cols) * (bpp) / 8) % 4)) % 4)

//////////////////////////////////////////////////////////////////////////////
// BMP Header
//////////////////////////////////////////////////////////////////////////////

struct BMPHeader {
  unsigned int bfSize = 0;
  unsigned int bfReserved = 0;
  unsigned int bfOffBits = 54;
  unsigned int biSize = 40;
  int biWidth = 0;
  int biHeight = 0;
  unsigned short biPlanes = 1;
  unsigned short biBitCount = 24;
  unsigned int biCompression = 0;
  unsigned int biSizeImage = 0;
  int biXPelsPerMeter = 0;
  int biYPelsPerMeter = 0;
  unsigned int biClrUsed = 0;
  unsigned int biClrImportant = 0;
};

//////////////////////////////////////////////////////////////////////////////
// Image
//////////////////////////////////////////////////////////////////////////////

void Image::set(uint32_t row, uint32_t col, uint8_t channel, uint8_t value)
{
  uint8_t bytes_per_pixel = _bits_per_pixel / 8;
  const size_t index = row * _cols * bytes_per_pixel + col * bytes_per_pixel;
  _data[index + channel] = value;
}

uint8_t Image::get(uint32_t row, uint32_t col, uint8_t channel) const
{
  uint8_t bytes_per_pixel = _bits_per_pixel / 8;
  const size_t index = row * _cols * bytes_per_pixel + col * bytes_per_pixel;
  return _data[index + channel];
}

void Image::set(uint32_t row, uint32_t col, uint8_t r, uint8_t g, uint8_t b)
{
  uint8_t bytes_per_pixel = _bits_per_pixel / 8;
  const size_t index = row * _cols * bytes_per_pixel + col * bytes_per_pixel;
  _data[index + CHANNEL_BLUE] = b;
  _data[index + CHANNEL_GREEN] = g;
  _data[index + CHANNEL_RED] = r;
}

void Image::write_row (uint32_t row, std::ofstream& f) const
{
  const size_t row_len = _cols * _bits_per_pixel / 8;
  f.write (reinterpret_cast<char*> (_data + row * row_len), row_len);
}

void Image::read_row (uint32_t row, std::ifstream& f)
{
  const size_t row_len = _cols * _bits_per_pixel / 8;
  f.read (reinterpret_cast<char*> (_data + row * row_len), row_len);
}

Image::Status Image::load(const std::string& filename)
{
  clear();

  // Default header
  BMPHeader header;

  // Open the image file in binary mode
  std::ifstream f_img (filename.c_str(), std::ios::binary);

  if (!f_img.is_open ())
    return STATUS_FILE_NOT_OPENED;

  // Since an adress must be passed to fread, create a variable!
  unsigned short magic;

  // Check if its an bmp file by comparing the magic nbr
  f_img.read(reinterpret_cast<char*>(&magic), sizeof (magic));

  if (magic != BMP_MAGIC)
  {
    f_img.close ();
    return STATUS_INVALID_FILE;
  }

  // Read the header structure into header
  f_img.read (reinterpret_cast<char*>(&header), sizeof (header));

  // Select the mode (bottom-up or top-down)
  const int h = std::abs (header.biHeight);
  const int offset = (header.biHeight > 0 ? 0 : h - 1);
  const int padding = BMP_PADDING (header.biWidth, header.biBitCount);

  // Allocate the pixel buffer
  init (h, header.biWidth, header.biBitCount);

  for (int y = h - 1; y >= 0; y--)
  {
    // Read a whole row of pixels from the file
    read_row ((int)std::abs (y - offset), f_img);

    // Skip the padding
    f_img.seekg (padding, std::ios::cur);
  }

  // NOTE: All good
  f_img.close ();
  return STATUS_OK;
}

Image::Status Image::save(const std::string& filename) const
{
  const int padding = BMP_PADDING(_cols, _bits_per_pixel);

  // Init header
  BMPHeader header;
  header.biWidth = _cols;
  header.biHeight = _rows;
  header.biBitCount = _bits_per_pixel;
  header.biSizeImage = (_cols * _bits_per_pixel / 8 + padding) * _rows;
  header.bfSize = sizeof(short) /*magic*/ + sizeof(header) + header.biSizeImage;

  // Open the image file in binary mode
  std::ofstream f_img (filename.c_str(), std::ios::binary);

  if (!f_img.is_open ())
    return STATUS_FILE_NOT_OPENED;

  // Since an adress must be passed to fwrite, create a variable!
  const unsigned short magic = BMP_MAGIC;

  f_img.write (reinterpret_cast<const char*>(&magic), sizeof (magic));
  f_img.write (reinterpret_cast<const char*>(&header), sizeof (header));

  // Select the mode (bottom-up or top-down)
  const int h = std::abs (header.biHeight);
  const int offset = (header.biHeight > 0 ? 0 : h - 1);

  for (int y = h - 1; y >= 0; y--)
  {
    // Write a whole row of pixels into the file
    write_row ((int)std::abs (y - offset), f_img);

    // Write the padding
    f_img.write ("\0\0\0", padding);
  }

  // NOTE: All good
  f_img.close ();
  return STATUS_OK;  
}

Image Image::crop(uint32_t row, uint32_t col, uint32_t rows, uint32_t cols) const
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

Image Image::norm() const
{
  Image im(_rows,  _cols, _bits_per_pixel);

  auto this_data = data();

  short max_value = std::numeric_limits<short>::min();
  short min_value = std::numeric_limits<short>::max();

  for (int i=size(); i>0; i--)
  {
    max_value = std::max<short>(this_data[i-1], max_value);
    min_value = std::min<short>(this_data[i-1], min_value);
  }

  if (max_value == min_value)
  {
    max_value = 255;
    min_value = 0;
  }

  auto scale = 255.0 / (max_value - min_value);

  auto im_data = im.data();

  for (int i=size(); i>0; i--)
  {
    im_data[i-1] = std::round(scale * (this_data[i-1] - min_value));
  }

  return im;
}

Image Image::scale(uint32_t rows, uint32_t cols, Interpolation interp) const
{
  switch(interp)
  {
    case INTERPOLATE_NEAREST:
      return scale_nearest(rows, cols);
    case INTERPOLATE_BILINEAR:
      return scale_bilinear(rows, cols);
  }
}

Image Image::scale_nearest(uint32_t rows, uint32_t cols) const
{
  Image im(rows,  cols, _bits_per_pixel);

  auto scale_r = (float)(_rows - 1)/(rows - 1);
  auto scale_c = (float)(_cols - 1)/(cols - 1);
  auto bytes_per_pixel = _bits_per_pixel / 8;

  for (auto r=0; r<rows; r++)
  for (auto c=0; c<cols; c++)
  {
    uint32_t this_c = std::round(scale_c * c);
    uint32_t this_r = std::round(scale_r * r);

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

Image Image::scale_bilinear(uint32_t rows, uint32_t cols) const
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
