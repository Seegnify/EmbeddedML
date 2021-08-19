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

#ifndef _SEEGNIFY_STORAGE_H_
#define _SEEGNIFY_STORAGE_H_

#include <time.h>
#include <iostream>

#include <google/protobuf/message.h>

#include "types.hh"

namespace seegnify {

// get file size
long file_size(const std::string& path);

// copy input stream to output stream
void copy(std::istream& in, std::ostream& out, long length = -1);

// load chunk of file to output stream
void read_chunk(const std::string& path, std::ostream& data, 
                long position, long size);

// write chunk of file from input stream
void write_chunk(const std::string& path, std::istream& data, 
                long position = -1);

// load file to output stream
void read_file(const std::string& path, std::ostream& data);

// save input stream to file
void write_file(const std::string& path, std::istream& data);

// integer encoding
void write_uint32(uint32_t val, std::ostream& stream);

// integer decoding
bool read_uint32(uint32_t &val, std::istream& stream);

// write delimited protobuf to stream
void write_pb(const ::google::protobuf::Message& pb, std::ostream& stream);

// read delimited protobuf from stream
bool read_pb(::google::protobuf::Message& pb, std::istream& stream);

// read DTYPE from stream
DTYPE read_dtype(std::istream& in);

// write DTYPE to stream
void write_dtype(DTYPE v, std::ostream& out);

// read int from stream
int read_int(std::istream& in);

// write int to stream
void write_int(int v, std::ostream& out);

// read Tensor from stream
Tensor read_tensor(std::istream& in);

// write Tensor to stream
void write_tensor(const Tensor& t, std::ostream& out);

// suseconds_t (microseconds) to string of format 2009-06-15 20:20:00.123456
std::string usec_to_string(const suseconds_t& time);

// suseconds_t (seconds) to string of format 2009-06-15 20:20:00
std::string time_to_string(const time_t& time);

// time_t (seconds) to string of format 20:20:00
std::string time_string(const time_t& time);

// suseconds_t (seconds) to string of format 2009-06-15
std::string date_string(const time_t& time);

// get local time
time_t time_now();

// get time zone relative to UCT
time_t time_zone(const time_t& now);

// get calandar date to time
tm time_to_date(const time_t& time);

// get calandar date to time
time_t date_to_time(const tm& date);

// reinforcement learning reward discount
std::vector<DTYPE> discount_reward(
const std::vector<DTYPE>& reward, DTYPE gamma);

// save sequence of RGB images as animation to file
void save_animation(const std::string& filename,
const std::vector<const uint8_t*>& rgb, int width, int height, int delay);

// load image file
void load_image(const std::string& filename, std::vector<uint8_t>& image, int& rows, int& cols, int& bits_per_pixel);

// save image file
void save_image(const std::string& filename, const uint8_t* rgb, int rows, int cols, int bits_per_pixel = 24);

// read audio file
void load_audio(const std::string& filename, std::vector<DTYPE>& samples,
int& num_channels, int& sample_rate);

// write audio file
void save_audio(const std::string& filename, const std::vector<DTYPE>& samples,
int num_channels, int sample_rate);

} /* namespace */

#endif /* _SEEGNIFY_STORAGE_H_ */
