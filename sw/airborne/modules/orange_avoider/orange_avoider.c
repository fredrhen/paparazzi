/*
 * Copyright (C) Roland Meertens
 *
 * This file is part of paparazzi
 *
 */
/**
 * @file "modules/orange_avoider/orange_avoider.c"
 * @author Roland Meertens
 * Example on how to use the colours detected to avoid orange pole in the cyberzoo
 * This module is an example module for the course AE4317 Autonomous Flight of Micro Air Vehicles at the TU Delft.
 * This module is used in combination with a color filter (cv_detect_color_object) and the navigation mode of the autopilot.
 * The avoidance strategy is to simply count the total number of orange pixels. When above a certain percentage threshold,
 * (given by color_count_frac) we assume that there is an obstacle and we turn.
 *
 * The color filter settings are set using the cv_detect_color_object. This module can run multiple filters simultaneously
 * so you have to define which filter to use with the ORANGE_AVOIDER_VISUAL_DETECTION_ID setting.
 */

#include "modules/computer_vision/opticflow/inter_thread_data.h"
#include "modules/orange_avoider/orange_avoider.h"
#include "firmwares/rotorcraft/navigation.h"
#include "generated/airframe.h"
#include "state.h"
#include "modules/core/abi.h"
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define NAV_C // needed to get the nav functions like Inside...
#include "generated/flight_plan.h"

#define ORANGE_AVOIDER_VERBOSE FALSE

#define PRINT(string,...) fprintf(stderr, "[orange_avoider->%s()] " string,__FUNCTION__ , ##__VA_ARGS__)
#if ORANGE_AVOIDER_VERBOSE
#define VERBOSE_PRINT PRINT
#else
#define VERBOSE_PRINT(...)
#endif

static uint8_t moveWaypointForward(uint8_t waypoint, float distanceMeters);
static uint8_t calculateForwards(struct EnuCoor_i *new_coor, float distanceMeters);
static uint8_t moveWaypoint(uint8_t waypoint, struct EnuCoor_i *new_coor);
static uint8_t increase_nav_heading(float incrementDegrees);
static uint8_t chooseRandomIncrementAvoidance(void);

enum navigation_state_t {
  SAFE,
  OBSTACLE_FOUND,
  OUT_OF_BOUNDS,
  AVOID_RIGHT_OBJECT,
  AVOID_LEFT_OBJECT,
  AVOID_CORNERS

};

// define settings
float oa_color_count_frac = 0.18f;

// define and initialise global variables
enum navigation_state_t navigation_state = SEARCH_FOR_SAFE_HEADING;
int32_t color_count = 0;                // orange color count from color filter for obstacle detection
int16_t obstacle_free_confidence = 0;   // a measure of how certain we are that the way ahead is safe.
float heading_increment = 135.f;          // heading angle increment [deg]
float maxDistance = 2.25;               // max waypoint displacement [m]
struct opticflow_result_t *result;
float flow_threshold = 0.005;
float absdiff;

const int16_t max_trajectory_confidence = 5; // number of consecutive negative object detections to be sure we are obstacle free

/*
 * This next section defines an ABI messaging event (http://wiki.paparazziuav.org/wiki/ABI), necessary
 * any time data calculated in another module needs to be accessed. Including the file where this external
 * data is defined is not enough, since modules are executed parallel to each other, at different frequencies,
 * in different threads. The ABI event is triggered every time new data is sent out, and as such the function
 * defined in this file does not need to be explicitly called, only bound in the init function
 */
#ifndef ORANGE_AVOIDER_VISUAL_DETECTION_ID
#define ORANGE_AVOIDER_VISUAL_DETECTION_ID ABI_BROADCAST
#endif
static abi_event color_detection_ev;
static void color_detection_cb(uint8_t __attribute__((unused)) sender_id,
                               int16_t __attribute__((unused)) pixel_x, int16_t __attribute__((unused)) pixel_y,
                               int16_t __attribute__((unused)) pixel_width, int16_t __attribute__((unused)) pixel_height,
                               int32_t quality, int16_t __attribute__((unused)) extra)
{
  color_count = quality;
}

static abi_event optical_flow_result;
static void optical_flow_cb(uint8_t __attribute__((unused)) sender_id,
                               uint32_t __attribute__((unused)) stamp, int32_t __attribute__((unused)) datatype,
                               uint32_t __attribute__((unused)) size, uint8_t *data)
{
  memcpy(result, data, sizeof(*result)); // Makes a copy of the struct, Removes the issue of memory problems. Not as efficient as pointers
}

/*
 * Initialisation function, setting the colour filter, random seed and heading_increment
 */
void orange_avoider_init(void)
{
  // Initialise random values
  srand(time(NULL));
  chooseRandomIncrementAvoidance();

  // bind our colorfilter callbacks to receive the color filter outputs
  AbiBindMsgVISUAL_DETECTION(ORANGE_AVOIDER_VISUAL_DETECTION_ID, &color_detection_ev, color_detection_cb);

  AbiBindMsgPAYLOAD_DATA(2, &optical_flow_result, optical_flow_cb);

  result = malloc(sizeof(struct opticflow_result_t));
}

/*
 * Function that checks it is safe to move forwards, and then moves a waypoint forward or changes the heading
 */
void orange_avoider_periodic(void)
{
  // only evaluate our state machine if we are flying
  if(!autopilot_in_flight()){
    return;
  }

  absdiff = fabs(result->div_size_left - result->div_size_right);

  printf("Left: ");
  printf("%f  ,", result->div_size_left);

  printf("Right: ");
  printf("%f \n", result->div_size_right);

  printf("Difference:");
  printf("%f \n", fabs(result->div_size_left - result->div_size_right));


  // compute current color thresholds
  int32_t color_count_threshold = oa_color_count_frac * front_camera.output_size.w * front_camera.output_size.h;

  VERBOSE_PRINT("Color_count: %d  threshold: %d state: %d \n", color_count, color_count_threshold, navigation_state);

  // update our safe confidence using color threshold
  if(color_count < color_count_threshold){
    obstacle_free_confidence++;
  } else {
    obstacle_free_confidence -= 2;  // be more cautious with positive obstacle detections
  }

  // bound obstacle_free_confidence
  Bound(obstacle_free_confidence, 0, max_trajectory_confidence);

  float moveDistance = fminf(maxDistance, 0.2f * obstacle_free_confidence);

  switch (navigation_state) {
      case SAFE:
          // Move waypoint forward
          moveWaypointForward(WP_TRAJECTORY, 1.5f * moveDistance);
	  // Detects if waypoint is out of bounds
          if (!InsideObstacleZone(WaypointX(WP_TRAJECTORY), WaypointY(WP_TRAJECTORY))) {
              navigation_state = OUT_OF_BOUNDS;
          }
	  // Detects if FOE is in the center of the camera
          else if (result->focus_of_expansion_x == 0 && result->focus_of_expansion_y == 0 ) {
              navigation_state = OBSTACLE_FOUND;
          }
	  // Detects if there is more divergence on the left hand side
          else if (result->div_size_left > result->div_size_right && absdiff > flow_threshold){
              navigation_state = AVOID_LEFT_OBJECT;
          }
	  // Detects if there is more divergence on the right hand side
          else if (result->div_size_left < result->div_size_right && absdiff > flow_threshold){
              navigation_state = AVOID_RIGHT_OBJECT;
          }
	  // Detects if waypoint is out of bounds and there is divergence on either side of the drone
          else if (!InsideObstacleZone(WaypointX(WP_TRAJECTORY), WaypointY(WP_TRAJECTORY)) && result->div_size_left > 1 || !InsideObstacleZone(WaypointX(WP_TRAJECTORY), WaypointY(WP_TRAJECTORY)) && result->div_size_right > 1){
        	  navigation_state = AVOID_CORNERS;
          }
	  // Move towards waypoint
          else {
              moveWaypointForward(WP_GOAL, 1.0f * moveDistance);
	  }
      break;
    case OBSTACLE_FOUND:
  	
        // stops the drone
	waypoint_move_here_2d(WP_GOAL);
	waypoint_move_here_2d(WP_TRAJECTORY);	  
	 
	// determines if there is more divergence to the left
        if (result->div_size_left < result->div_size_right){
            // turn left
            increase_nav_heading(-1 * 5 * heading_increment);
	    // return to default state
            navigation_state = SAFE;
        }
	
	// determines if there is more divergence to the right
        else {
            // turn right
            increase_nav_heading(1 * 5 * heading_increment);
            // return to default state
            navigation_state = SAFE;
        }

      break;
    case AVOID_CORNERS:
       
       // stop
	waypoint_move_here_2d(WP_GOAL);
	waypoint_move_here_2d(WP_TRAJECTORY);
		  
	// determines if there is more divergence to the left
    	if (result->div_size_left > result->div_size_right && absdiff > 1){
		
		// turn left by a huge angle
		increase_nav_heading(-6 * heading_increment);
    		// return to default state
            	navigation_state = SAFE;
    	}
	
	// determines if there is more divergence to the right
    	else if (result->div_size_left > result->div_size_right && absdiff > 1){
			
		// turn right by a huge angle
		increase_nav_heading(6 * heading_increment);
		// return to default state
            	navigation_state = SAFE;
    	}
    	break;
    case OUT_OF_BOUNDS:
		  
      // slow down
      moveWaypointForward(WP_GOAL, .25 * moveDistance);
		  
      // turn away from out of bounds
      increase_nav_heading(7 * heading_increment);
		  
      // project a waypoint in front of the drone
      moveWaypointForward(WP_TRAJECTORY, 1.5f);
      
      // determines if the new path is indeed within bounds
      if (InsideObstacleZone(WaypointX(WP_TRAJECTORY),WaypointY(WP_TRAJECTORY))){
	      
        // add offset to head back into arena
        increase_nav_heading(heading_increment);
	      
        navigation_state = SAFE;
      }
      break;
    case AVOID_RIGHT_OBJECT:
        // stop
        waypoint_move_here_2d(WP_GOAL);
        waypoint_move_here_2d(WP_TRAJECTORY);

        // turn left
        increase_nav_heading(-1.5 * heading_increment);
		  
	// return to default state
        navigation_state = SAFE;

        break;
    case AVOID_LEFT_OBJECT:
        // stop
        waypoint_move_here_2d(WP_GOAL);
        waypoint_move_here_2d(WP_TRAJECTORY);

        // turn right
        increase_nav_heading(1.5 * heading_increment);

        // return to default state
        navigation_state = SAFE;

        break;

    default:
      break;
  }
  return;
}

/*
 * Increases the NAV heading. Assumes heading is an INT32_ANGLE. It is bound in this function.
 */
uint8_t increase_nav_heading(float incrementDegrees)
{
  float new_heading = stateGetNedToBodyEulers_f()->psi + RadOfDeg(incrementDegrees);

  // normalize heading to [-pi, pi]
  FLOAT_ANGLE_NORMALIZE(new_heading);

  // set heading, declared in firmwares/rotorcraft/navigation.h
  // for performance reasons the navigation variables are stored and processed in Binary Fixed-Point format
  nav_heading = ANGLE_BFP_OF_REAL(new_heading);

  VERBOSE_PRINT("Increasing heading to %f\n", DegOfRad(new_heading));
  return false;
}

/*
 * Calculates coordinates of distance forward and sets waypoint 'waypoint' to those coordinates
 */
uint8_t moveWaypointForward(uint8_t waypoint, float distanceMeters)
{
  struct EnuCoor_i new_coor;
  calculateForwards(&new_coor, distanceMeters);
  moveWaypoint(waypoint, &new_coor);
  return false;
}

/*
 * Calculates coordinates of a distance of 'distanceMeters' forward w.r.t. current position and heading
 */
uint8_t calculateForwards(struct EnuCoor_i *new_coor, float distanceMeters)
{
  float heading  = stateGetNedToBodyEulers_f()->psi;

  // Now determine where to place the waypoint you want to go to
  new_coor->x = stateGetPositionEnu_i()->x + POS_BFP_OF_REAL(sinf(heading) * (distanceMeters));
  new_coor->y = stateGetPositionEnu_i()->y + POS_BFP_OF_REAL(cosf(heading) * (distanceMeters));
  VERBOSE_PRINT("Calculated %f m forward position. x: %f  y: %f based on pos(%f, %f) and heading(%f)\n", distanceMeters,	
                POS_FLOAT_OF_BFP(new_coor->x), POS_FLOAT_OF_BFP(new_coor->y),
                stateGetPositionEnu_f()->x, stateGetPositionEnu_f()->y, DegOfRad(heading));
  return false;
}

/*
 * Sets waypoint 'waypoint' to the coordinates of 'new_coor'
 */
uint8_t moveWaypoint(uint8_t waypoint, struct EnuCoor_i *new_coor)
{
  VERBOSE_PRINT("Moving waypoint %d to x:%f y:%f\n", waypoint, POS_FLOAT_OF_BFP(new_coor->x),
                POS_FLOAT_OF_BFP(new_coor->y));
  waypoint_move_xy_i(waypoint, new_coor->x, new_coor->y);
  return false;
}

/*
 * Sets the variable 'heading_increment' randomly positive/negative
 */
uint8_t chooseRandomIncrementAvoidance(void)
{
  // Randomly choose CW or CCW avoiding direction
  if (rand() % 2 == 0) {
    heading_increment = 5.f;
    VERBOSE_PRINT("Set avoidance increment to: %f\n", heading_increment);
  } else {
    heading_increment = -5.f;
    VERBOSE_PRINT("Set avoidance increment to: %f\n", heading_increment);
  }
  return false;
}

