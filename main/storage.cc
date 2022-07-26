/*
 * Copyright 2020-2021 Greg Padiasek and Seegnify <http://www.seegnify.org>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <sstream>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <memory>
#include <ctime>
#include <sys/stat.h>
#include <arpa/inet.h>

#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/coded_stream.h>

#include <Magick++.h>
#include <sndfile.hh>

#include "storage.hh"

namespace seegnify {

// get file size
long file_size(const std::string& path)
{
  struct stat st;
  int rc = stat(path.data(), &st);
  return rc == 0 ? st.st_size : -1;  
}

// copy limited number of bytes between streams
void copy(std::istream& in, std::ostream& out, long length) {
  // copy entire buffer
  if (length < 0) {
    out << in.rdbuf();
    return;
  }

  // adjust length based on bytes available in input
  long pos = in.tellg();
  in.seekg(0, in.end);
  length = std::min<long>(in.tellg() - pos, length);
  in.seekg(pos);

  // create copy buffer
  long count = 0;
  std::vector<char> buffer(1024 * 1024);
  char *data = buffer.data();

  // copy 'length' bytes
  while (count < length) {
    long size = std::min<long>(length - count, buffer.size());
    in.read(data, size);
    out.write(data, size);
    count += size;
  }
}

// load chunk of file to output stream
void read_chunk(const std::string& path, std::ostream& data, 
                long position, long size)
{
  // open the file
  std::ifstream file(path, std::ios::in | std::ifstream::binary);
  if (!file) {
    std::ostringstream error;
    error << "Failed to read chunk '" << path << "'. Error code ";
    error << errno << " ." << std::endl;
    throw std::runtime_error(error.str());
  }

  // set read position
  file.seekg(position);

  // read chunk from file
  copy(file, data, size);

  // close file
  file.close();
}

// write chunk of file from input stream
void write_chunk(const std::string& path, std::istream& data, 
                long position)
{
  // open the file
  std::ofstream file(path, std::ios::ate | std::ofstream::binary);
  if (!file) {
    std::ostringstream error;
    error << "Failed to write chunk '" << path << "'. Error code ";
    error << errno << " ." << std::endl;
    throw std::runtime_error(error.str());
  }

  // set write position
  if (position >= 0) file.seekp(position);

  // write chunk to file
  copy(data, file);

  // close file
  file.close();
}

// read file to stream
void read_file(const std::string& path, std::ostream& data) {
  // open the file
  std::ifstream file(path, std::ios::in | std::ifstream::binary);
  if (!file) {
    std::ostringstream error;
    error << "Failed to read file '" << path << "'. Error code ";
    error << errno << " ." << std::endl;
    throw std::runtime_error(error.str());
  }

  // copy everything
  copy(file, data);

  // close file
  file.close();
}

// write stream to file
void write_file(const std::string& path, std::istream& data) {
  // open the file
  std::ofstream file(path, std::ios::out | std::ofstream::binary);
  if (!file) {
    std::ostringstream error;
    error << "Failed to write file '" << path << "'. Error code ";
    error << errno << " ." << std::endl;
    throw std::runtime_error(error.str());
  }

  // copy everything
  copy(data, file);

  // save file
  file.close();
}

// integer encoding
void write_uint32(uint32_t val, std::ostream& stream) {
  val = htonl(val);
  stream.write((const char*)&val, sizeof(uint32_t));
}

// integer decoding
bool read_uint32(uint32_t &val, std::istream& stream) {
  uint32_t nval;
  stream.read((char*)&nval, sizeof(uint32_t));
  val = ntohl(nval);
  return (stream) ? true : false;
}

// write protobuf message in envelope
void write_pb(const ::google::protobuf::Message& pb, std::ostream& output) {
  write_uint32(pb.ByteSize(), output);
  google::protobuf::io::OstreamOutputStream ostream(&output);
  google::protobuf::io::CodedOutputStream ocoded(&ostream);
  pb.SerializeWithCachedSizes(&ocoded);
}

// read protobuf message in envelope
bool read_pb(::google::protobuf::Message& pb, std::istream& input) {
  // check EOF
  if (input.peek() == EOF) {
    return false;
  }

  uint32_t size;
  if (!read_uint32(size, input)) {
    throw std::runtime_error("Failed to read protobuf message size");
  }

  // copy message to buffer
  std::unique_ptr<uint8_t[]> buffer(new uint8_t[size]);
  input.read((char*)buffer.get(), size);
  google::protobuf::io::CodedInputStream icoded(buffer.get(), size);

  // parse message from buffer
  auto limit = icoded.PushLimit(size);
  if (!pb.ParseFromCodedStream(&icoded)) {
    std::ostringstream error;
    error << "Failed to read protobuf message of type '";
    error << pb.GetDescriptor()->name() << "'";
    throw std::runtime_error(error.str());
  }
  if (icoded.BytesUntilLimit() > 0) {
    std::ostringstream error;
    error << "Incomplete protobuf message of type '";
    error << pb.GetDescriptor()->name() << "'";
    throw std::runtime_error(error.str());
  }
  icoded.PopLimit(limit);

  return true;
}

// read int from stream
int read_int(std::istream& in)
{
  int v;
  in.read((char*)&v, sizeof(v));
  return v;
}

// write int to stream
void write_int(int v, std::ostream& out)
{
  out.write((const char*)&v, sizeof(v));
}

// read DTYPE from stream
DTYPE read_dtype(std::istream& in)
{
  DTYPE v;
  in.read((char*)&v, sizeof(v));
  return v;
}

// write DTYPE to stream
void write_dtype(DTYPE v, std::ostream& out)
{
  out.write((const char*)&v, sizeof(v));
}

// read Tensor from stream
Tensor read_tensor(std::istream& in)
{
  int rows = read_int(in);
  int cols = read_int(in);
  Tensor t(rows, cols);
  in.read((char*)t.data(), t.size() * sizeof(Tensor::Scalar));
  return t;
}

// write Tensor to stream
void write_tensor(const Tensor& t, std::ostream& out)
{
  write_int(t.rows(), out);
  write_int(t.cols(), out);
  out.write((const char*)t.data(), t.size() * sizeof(Tensor::Scalar));
}

// suseconds_t (microseconds) to string of format 2009-06-15 20:20:00.1234567
std::string usec_to_string(const suseconds_t& time) {
  time_t secs = time / 1000000;
  uint32_t usec = time % 1000000;
  std::tm * ptm = std::localtime(&secs);
  char buffer[32];
  auto len = std::strftime(buffer, 20, "%Y-%m-%d %H:%M:%S", ptm);
  std::sprintf(buffer + len, ".%06d", usec);
  return std::string(buffer);
}

// time_t (seconds) to string of format 2009-06-15 20:20:00
std::string time_to_string(const time_t& time) {
  std::tm * ptm = std::localtime(&time);
  char buffer[32];
  auto len = std::strftime(buffer, 20, "%Y-%m-%d %H:%M:%S", ptm);
  return std::string(buffer);
}

// time_t (seconds) to date string of format 2009-06-15
std::string date_string(const time_t& time) {
  std::tm * ptm = std::gmtime(&time);
  char buffer[32];
  auto len = std::strftime(buffer, 20, "%Y-%m-%d", ptm);
  return std::string(buffer);
}

// time_t (seconds) to time string of format 20:20:00
std::string time_string(const time_t& time) {
  std::tm * ptm = std::gmtime(&time);
  char buffer[32];
  auto len = std::strftime(buffer, 20, "%H:%M:%S", ptm);
  return std::string(buffer);
}

// get local time
time_t time_now() {
  return std::time(NULL);
}

// get time zone relative to UCT
time_t time_zone(const time_t& now) {
  tm* _local = std::localtime(&now);
  time_t local = std::mktime(std::localtime(&now));
  time_t gmt = std::mktime(std::gmtime(&now));
  time_t dst = ((_local->tm_isdst>0) ? 3600 : 0);
  return difftime(local, gmt) + dst;
}

// get calandar date to time
tm time_to_date(const time_t& time) {
  return *std::localtime(&time);
}

// get calandar date to time
time_t date_to_time(const tm& date) {
  time_t now = std::time(NULL);
  tm* _local = std::localtime(&now);

  *_local = date;
  return mktime(_local); // seconds
}

// reinforcement learning reward discount
std::vector<DTYPE> discount_reward(
  const std::vector<DTYPE>& reward, 
  DTYPE gamma = GAMMA_DISCOUNT)
{
  // compute discounted rewards
  DTYPE r = 0;
  std::vector<DTYPE> dreward(reward.size());
  for (int i=reward.size()-1; i>=0; i--)
  {
    r = gamma * r  + reward[i];
    dreward[i] = r;
  }
  return dreward;
}

// save sequence of RGB images as animation to file
void save_animation(const std::string& filename,
const std::vector<const uint8_t*>& rgb, int width, int height, int delay) {
  //InitializeMagick("");
  std::vector<Magick::Image> frames;

  for (auto frame: rgb) {
    Magick::Image image(width, height, "RGB", Magick::CharPixel, frame);
    image.animationDelay(delay);
    frames.push_back(std::move(image));
  }

  Magick::writeImages(frames.begin(), frames.end(), filename);
}

// read audio file
void load_audio(const std::string& filename, std::vector<DTYPE>& samples,
int& num_channels, int& sample_rate)
{
  SndfileHandle file(filename);

  uint32_t format = file.format();
  bool is_wav = format & SF_FORMAT_WAV;
  bool is_pcm16 = format & SF_FORMAT_PCM_16;
  uint32_t num_samples = file.frames();
  uint32_t sample_bits = (is_wav && is_pcm16) ? 16 : 0;

  num_channels = file.channels();
  sample_rate = file.samplerate();

  samples.resize(num_channels * num_samples);
  file.read(samples.data(), samples.size());
}

// write audio file
void save_audio(const std::string& filename, const std::vector<DTYPE>& samples,
int num_channels, int sample_rate)
{
  int format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;

  SndfileHandle file(filename, SFM_WRITE, format, num_channels, sample_rate);

  file.write(samples.data(), samples.size());\
}

// read image file
void load_image(const std::string& filename, std::vector<uint8_t>& data, int& rows, int& cols, int& bits_per_pixel)
{
  Magick::Image image;
  image.read(filename);

  rows = image.rows();
  cols = image.columns();
  bits_per_pixel = image.depth();

  Magick::Blob blob;

  switch(bits_per_pixel)
  {
    case 8:
    {
      image.magick("R");
      image.type(Magick::GrayscaleType);
      image.write(&blob);
      break;
    }
    case 16:
    {
      image.magick("R");
      image.type(Magick::GrayscaleType);
      image.depth(16);
      image.write(&blob);
      break;
    }
    case 24:
    {
      image.magick("RGB");
      image.write(&blob);
      break;
    }
    case 32:
    {
      image.magick("RGBA");
      image.write(&blob);
      break;
    }
    default:
    {
      std::ostringstream msg;
      msg << "Unsupported image depth " << bits_per_pixel;
      throw std::runtime_error(msg.str());
    }
  };

  data.resize(blob.length());
  std::copy_n((uint8_t*)blob.data(), blob.length(), (uint8_t*)data.data());
}

// write image file
void save_image(const std::string& filename, const uint8_t* data, int rows, int cols, int bits_per_pixel)
{
  switch(bits_per_pixel)
  {
    case 8:
    {
      Magick::Image image(cols, rows, "R", Magick::CharPixel, data);
      image.type(Magick::GrayscaleType);
      image.write(filename);
      break;
    }
    case 16:
    {
      Magick::Image image(cols, rows, "R", Magick::CharPixel, data);
      image.type(Magick::GrayscaleType);
      image.depth(16);
      image.write(filename);
      break;
    }
    case 24:
    {
      Magick::Image image(cols, rows, "RGB", Magick::CharPixel, data);
      image.write(filename);
      break;
    }
    case 32:
    {
      Magick::Image image(cols, rows, "RGBA", Magick::CharPixel, data);
      image.write(filename);
      break;
    }
    default:
    {
      std::ostringstream msg;
      msg << "Unsupported image depth " << bits_per_pixel;
      throw std::runtime_error(msg.str());
    }
  };
}

} /* namespace */

