#ifndef CONVEX_PARTITION_H
#define CONVEX_PARTITION_H

#include <stddef.h>

/**
 * Convex Partition Optimization - C Interface
 * 
 * Data structures and function signatures for high-performance 
 * convex polygon partition optimization.
 */

// Maximum sizes for static allocation - can be adjusted as needed
#define MAX_VERTICES 1000
#define MAX_POLYGONS 100
#define MAX_POLYGON_SIZE 20

/**
 * Vertex Cloud Structure
 * 
 * Represents a collection of vertices organized by type:
 * - Corner vertices (fixed at square corners)
 * - Edge vertices (on square boundaries)  
 * - Inner vertices (inside square)
 */
typedef struct {
    int n_inner;        // Number of inner vertices
    int n_edge[4];      // Edge vertex counts [x=0, y=0, x=1, y=1]
    int n_corner;       // Corner vertex count (always 4)
    int n;              // Total vertices = n_corner + sum(n_edge) + n_inner
    double x[MAX_VERTICES];  // X coordinates
    double y[MAX_VERTICES];  // Y coordinates
} vertex_cloud;

/**
 * Polygon Structure
 * 
 * Represents a single polygon as indices into a vertex cloud
 */
typedef struct {
    int s;              // Number of vertices in polygon
    int vertices[MAX_POLYGON_SIZE];  // Indices into vertex cloud
} polygon;

/**
 * Partition Structure
 * 
 * Represents a complete partition of the unit square
 */
typedef struct {
    int n_poly;         // Number of polygons in partition
    int s_max;          // Maximum vertices per polygon
    int s[MAX_POLYGONS]; // Size of each polygon
    int pi[MAX_POLYGONS][MAX_POLYGON_SIZE]; // Polygon vertex indices
} partition;

/**
 * Gradient Structure
 * 
 * Gradient vectors for polygon loss function w.r.t. vertex positions
 */
typedef struct {
    double dxy[MAX_POLYGON_SIZE][2];  // Gradient [vertex][x,y]
} gradient;

// Function declarations

/**
 * Loss Functions
 */
double loss_vc(const vertex_cloud* cloud, polygon* out_worst_polygon, partition* out_partition);
double loss_part(const partition* part, polygon* out_worst_polygon);
double loss_poly(const polygon* poly, const vertex_cloud* cloud);

/**
 * Gradient Functions
 */
double dloss_poly(const polygon* poly, const vertex_cloud* cloud, gradient* grad);

/**
 * Utility Functions
 */
void init_vertex_cloud(vertex_cloud* cloud);
void init_partition(partition* part);
void init_polygon(polygon* poly);

/**
 * Validation Functions
 */
int validate_vertex_cloud(const vertex_cloud* cloud);
int validate_partition(const partition* part, const vertex_cloud* cloud);
int validate_polygon(const polygon* poly, const vertex_cloud* cloud);

#endif // CONVEX_PARTITION_H