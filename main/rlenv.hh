/*
 * Copyright 2020-2023 Greg Padiasek
 */

#ifndef _SEEGNIFY_RL_ENV_H_
#define _SEEGNIFY_RL_ENV_H_

#include <cstdint>
#include <string>
#include <iostream>
#include "image.hh"

namespace seegnify {

// RL environment
class RLEnv
{
public:
    RLEnv();
    virtual ~RLEnv();

    ////////////////////////// instance RL API /////////////////////////////////

    void new_episode();
    std::string get_info();

    void set_data_rgb(uint8_t* rgb, uint16_t depth, uint16_t rows, uint16_t cols);
    void set_view_size(uint16_t rows, uint16_t cols);

    ////////////////////////// instance UI API /////////////////////////////////

    void get_data_size(uint16_t& rows, uint16_t& cols);
    void get_view_size(uint16_t& rows, uint16_t& cols);

    Image get_data_rgb();
    Image get_view_rgb();

    void enable_view_frame(bool show) { show_view_frame = show; }

    ////////////////////////// discrete actions ////////////////////////////////

    void action_up();
    void action_down();
    void action_left();
    void action_right();
    void action_forward();
    void action_backward();
    void action_zoom_in();
    void action_zoom_out();

    ////////////////////////// continous actions ///////////////////////////////

    void action_horizontal(float rx);
    void action_vertical(float ry);
    void action_depth(float rz);
    void action_zoom(float zoom);

    ////////////////////////// private methods /////////////////////////////////

protected:
    uint32_t data_row(uint32_t view_row);
    uint32_t data_col(uint32_t view_col);
    uint32_t view_row(uint32_t data_row);
    uint32_t view_col(uint32_t data_col);

    virtual void clear();

    void draw_agent_frame(Image& img);

protected:

    // data
    uint8_t *data;
    uint16_t slices;
    uint16_t data_rows;
    uint16_t data_cols;
    uint16_t view_rows;
    uint16_t view_cols;

    // position
    float slice;
    float x, y;
    float scale;

    // action cache
    uint32_t action_step;

    // UI
    bool show_view_frame;
};

} /* namespace */

#endif
