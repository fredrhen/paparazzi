/*
 * Copyright (C) Kirk Scheper <kirkscheper@gmail.com>
 *
 * This file is part of paparazzi
 *
 */
/**
 * @file "modules/orange_avoider/orange_avoider_guided.c"
 * @author Kirk Scheper
 * This module is an example module for the course AE4317 Autonomous Flight of Micro Air Vehicles at the TU Delft.
 * This module is used in combination with a color filter (cv_detect_color_object) and the guided mode of the autopilot.
 * The avoidance strategy is to simply count the total number of orange pixels. When above a certain percentage threshold,
 * (given by color_count_frac) we assume that there is an obstacle and we turn.
 *
 * The color filter settings are set using the cv_detect_color_object. This module can run multiple filters simultaneously
 * so you have to define which filter to use with the ORANGE_AVOIDER_VISUAL_DETECTION_ID setting.
 * This module differs from the simpler orange_avoider.xml in that this is flown in guided mode. This flight mode is
 * less dependent on a global positioning estimate as witht the navigation mode. This module can be used with a simple
 * speed estimate rather than a global position.
 *
 * Here we also need to use our onboard sensors to stay inside of the cyberzoo and not collide with the nets. For this
 * we employ a simple color detector, similar to the orange poles but for green to detect the floor. When the total amount
 * of green drops below a given threshold (given by floor_count_frac) we assume we are near the edge of the zoo and turn
 * around. The color detection is done by the cv_detect_color_object module, use the FLOOR_VISUAL_DETECTION_ID setting to
 * define which filter to use.
 */

#include "modules/computer_vision/opticflow/inter_thread_data.h"
#include "modules/orange_avoider/orange_avoider_guided.h"
#include "firmwares/rotorcraft/guidance/guidance_h.h"
#include "generated/airframe.h"
#include "state.h"
#include "modules/core/abi.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define ORANGE_AVOIDER_VERBOSE TRUE

#define PRINT(string,...) fprintf(stderr, "[orange_avoider_guided->%s()] " string,_FUNCTION_ , ##_VA_ARGS_)
#if ORANGE_AVOIDER_VERBOSE
#define VERBOSE_PRINT PRINT
#else
#define VERBOSE_PRINT(...)
#endif



enum navigation_state_t {
    SAFE,
    OBSTACLE_FOUND,
    SEARCH_FOR_SAFE_HEADING,
    OUT_OF_BOUNDS,
    REENTER_ARENA,
    AVOID_RIGHT_OBJECT,
    AVOID_LEFT_OBJECT
};

// define settings
float oag_color_count_frac = 0.18f;       // obstacle detection threshold as a fraction of total of image
float oag_floor_count_frac = 0.05f;       // floor detection threshold as a fraction of total of image
float oag_max_speed = 5.f;               // max flight speed [m/s]
float oag_heading_rate = RadOfDeg(20.f);  // heading change setpoint for avoidance [rad/s]
struct opticflow_result_t *result;
float flow_threshold;
float flow_threshold_const= 0.01;
float flow_threshold_min = 0.05 ;
float absdiff;

// define and initialise global variables
enum navigation_state_t navigation_state = SEARCH_FOR_SAFE_HEADING;   // current state in state machine
int32_t color_count = 0;                // orange color count from color filter for obstacle detection
int32_t floor_count = 0;                // green color count from color filter for floor detection
int32_t floor_centroid = 0;             // floor detector centroid in y direction (along the horizon)
float avoidance_heading_direction = 0;  // heading change direction for avoidance [rad/s]
int16_t obstacle_free_confidence = 0;   // a measure of how certain we are that the way ahead if safe.

const int16_t max_trajectory_confidence = 5;  // number of consecutive negative object detections to be sure we are obstacle free

// This call back will be used to receive the color count from the orange detector
#ifndef ORANGE_AVOIDER_VISUAL_DETECTION_ID
#error This module requires two color filters, as such you have to define ORANGE_AVOIDER_VISUAL_DETECTION_ID to the orange filter
#error Please define ORANGE_AVOIDER_VISUAL_DETECTION_ID to be COLOR_OBJECT_DETECTION1_ID or COLOR_OBJECT_DETECTION2_ID in your airframe
#endif
static abi_event color_detection_ev;
static void color_detection_cb(uint8_t _attribute_((unused)) sender_id,
int16_t _attribute((unused)) pixel_x, int16_t __attribute_((unused)) pixel_y,
int16_t _attribute((unused)) pixel_width, int16_t __attribute_((unused)) pixel_height,
int32_t quality, int16_t _attribute_((unused)) extra)
{
color_count = quality;
}

#ifndef FLOOR_VISUAL_DETECTION_ID
#error This module requires two color filters, as such you have to define FLOOR_VISUAL_DETECTION_ID to the orange filter
#error Please define FLOOR_VISUAL_DETECTION_ID to be COLOR_OBJECT_DETECTION1_ID or COLOR_OBJECT_DETECTION2_ID in your airframe
#endif
static abi_event floor_detection_ev;
static void floor_detection_cb(uint8_t _attribute_((unused)) sender_id,
int16_t _attribute_((unused)) pixel_x, int16_t pixel_y,
        int16_t _attribute((unused)) pixel_width, int16_t __attribute_((unused)) pixel_height,
int32_t quality, int16_t _attribute_((unused)) extra)
{
floor_count = quality;
floor_centroid = pixel_y;
}

/*
 * Initialisation function
 */
void orange_avoider_guided_init(void)
{

    // bind our colorfilter callbacks to receive the color filter outputs
    AbiBindMsgVISUAL_DETECTION(ORANGE_AVOIDER_VISUAL_DETECTION_ID, &color_detection_ev, color_detection_cb);
    AbiBindMsgVISUAL_DETECTION(FLOOR_VISUAL_DETECTION_ID, &floor_detection_ev, floor_detection_cb);
}

/*
 * Function that checks it is safe to move forwards, and then sets a forward velocity setpoint or changes the heading
 */
void orange_avoider_guided_periodic(void)
{
    // Only run the module if we are in the correct flight mode
    if (guidance_h.mode != GUIDANCE_H_MODE_GUIDED) {
        navigation_state = SAFE;
        obstacle_free_confidence = 0;
        return;
    }

    // Take the absolute difference between left divergence and right divergence
    absdiff = fabs(result->div_size_left - result->div_size_right);

    // compute current color thresholds
    int32_t color_count_threshold = oag_color_count_frac * front_camera.output_size.w * front_camera.output_size.h;
    int32_t floor_count_threshold = oag_floor_count_frac * front_camera.output_size.w * front_camera.output_size.h;
    float floor_centroid_frac = floor_centroid / (float)front_camera.output_size.h / 2.f;

    VERBOSE_PRINT("Color_count: %d  threshold: %d state: %d \n", color_count, color_count_threshold, navigation_state);
    VERBOSE_PRINT("Floor count: %d, threshold: %d\n", floor_count, floor_count_threshold);
    VERBOSE_PRINT("Floor centroid: %f\n", floor_centroid_frac);

    // update our safe confidence using color threshold
    if(color_count < color_count_threshold){
        obstacle_free_confidence++;
    } else {
        obstacle_free_confidence -= 2;  // be more cautious with positive obstacle detections
    }

    // bound obstacle_free_confidence
    Bound(obstacle_free_confidence, 0, max_trajectory_confidence);

    float speed_sp = oag_max_speeds

    //Flow threshold is a function of speed. The flow constant is multiplied by the airspeed of the drone + a minimum flow threshold for when the airspeed is zero
    flow_threshold= flow_threshold_const * airspeed_f()+flow_threshold_min;

    switch (navigation_state){
        case SAFE:
            // Condition for out of bounds detection
            if (floor_count < floor_count_threshold || fabsf(floor_centroid_frac) > 0.12){
                navigation_state = OUT_OF_BOUNDS;
            }
                //Condition for near obstacle detection using focus of Expansion
            else if (result->focus_of_expansion_x == 0) {
                navigation_state = OBSTACLE_FOUND;
            }
                // Conditions for turning left: left divergence is higher than right divergence and the absolute value
                // of the difference between left divergence and right divergence is bigger than the flow threshold
            else if (result->div_size_left > result->div_size_right && absdiff > flow_threshold) {
                navigation_state = AVOID_LEFT_OBJECT;
            }
                // Conditions for turning right: right divergence is higher than left divergence and the absolute value
                // of the difference between left divergence and right divergence is bigger than the flow threshold
            else if (result->div_size_left < result->div_size_right && absdiff > flow_threshold){
                navigation_state = AVOID_RIGHT_OBJECT;
            }
                // If there are no obstacles in the way and it is not out of bounds the drone is given a forward velocity
            else {
                guidance_h_set_guided_body_vel(speed_sp, 0);
            }

            break;
        case OBSTACLE_FOUND:
            // stop
            guidance_h_set_guided_body_vel(0, 0);

            //Check on which side the divergence is bigger so we know which direction we should turn to
            if (result->div_size_left > result->div_size_right){
                // Change heading rate to turn left
                guidance_h_set_guided_heading_rate(-oag_heading_rate);
                navigation_state = SAFE;
            }

            else {
                // Change heading rate to turn right
                guidance_h_set_guided_heading_rate(oag_heading_rate)
                navigation_state = SAFE;
            }

            break;

        case SEARCH_FOR_SAFE_HEADING:
            guidance_h_set_guided_heading_rate(avoidance_heading_direction * oag_heading_rate);

            // make sure we have a couple of good readings before declaring the way safe
            if (obstacle_free_confidence >= 2){
                guidance_h_set_guided_heading(stateGetNedToBodyEulers_f()->psi);
                navigation_state = SAFE;
            }
            break;

        case OUT_OF_BOUNDS:
            // stop
            guidance_h_set_guided_body_vel(0, 0);

            // start turn back into arena
            guidance_h_set_guided_heading_rate(avoidance_heading_direction * RadOfDeg(15));

            navigation_state = REENTER_ARENA;

            break;
        case REENTER_ARENA:
            // force floor center to opposite side of turn to head back into arena
            if (floor_count >= floor_count_threshold && avoidance_heading_direction * floor_centroid_frac >= 0.f){
                // return to heading mode
                guidance_h_set_guided_heading(stateGetNedToBodyEulers_f()->psi);

                // reset safe counter
                obstacle_free_confidence = 0;

                // ensure direction is safe before continuing
                navigation_state = SAFE;
            }
            break;

        case AVOID_RIGHT_OBJECT:
            // stop
            guidance_h_set_guided_body_vel(0, 0);

            // Positive heading rate to make a clockwise turn
            guidance_h_set_guided_heading_rate(oag_heading_rate);

            navigation_state = SAFE;

            break;
        case AVOID_LEFT_OBJECT:
            // stop
            guidance_h_set_guided_body_vel(0, 0);

            // Negative heading rate to make a counter-clockwise turn
            guidance_h_set_guided_heading_rate(-oag_heading_rate);

            navigation_state = SAFE;

            break;
        default:
            break;
    }
    return;
}

