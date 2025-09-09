#include "convex_partition.h"
#include <stdio.h>
#include <assert.h>
#include <math.h>

/**
 * Test suite for C convex partition functions
 */

void test_init_functions() {
    printf("Testing initialization functions...\n");
    
    vertex_cloud cloud;
    partition part;
    polygon poly;
    
    // Test vertex cloud initialization
    init_vertex_cloud(&cloud);
    assert(cloud.n_corner == 4);
    assert(cloud.n == 4);
    assert(cloud.x[0] == 0.0 && cloud.y[0] == 0.0);
    assert(cloud.x[2] == 1.0 && cloud.y[2] == 1.0);
    
    // Test partition initialization  
    init_partition(&part);
    assert(part.n_poly == 1);
    assert(part.s[0] == 4);
    assert(part.pi[0][0] == 0);
    assert(part.pi[0][3] == 3);
    
    // Test polygon initialization
    init_polygon(&poly);
    assert(poly.s == 4);
    assert(poly.vertices[0] == 0);
    assert(poly.vertices[3] == 3);
    
    printf("✅ Initialization tests passed\n");
}

void test_validation_functions() {
    printf("Testing validation functions...\n");
    
    vertex_cloud cloud;
    partition part;
    polygon poly;
    
    init_vertex_cloud(&cloud);
    init_partition(&part);  
    init_polygon(&poly);
    
    // Valid cases
    assert(validate_vertex_cloud(&cloud) == 1);
    assert(validate_partition(&part, &cloud) == 1);
    assert(validate_polygon(&poly, &cloud) == 1);
    
    // Invalid cases
    assert(validate_vertex_cloud(NULL) == 0);
    assert(validate_partition(NULL, &cloud) == 0);
    assert(validate_polygon(NULL, &cloud) == 0);
    
    // Invalid vertex count
    cloud.n_corner = 3;  // Should be 4
    assert(validate_vertex_cloud(&cloud) == 0);
    
    printf("✅ Validation tests passed\n");
}

void test_loss_functions() {
    printf("Testing loss functions...\n");
    
    vertex_cloud cloud;
    partition part;
    polygon poly;
    gradient grad;
    
    init_vertex_cloud(&cloud);
    init_partition(&part);
    init_polygon(&poly);
    
    // Test polygon loss
    double poly_loss = loss_poly(&poly, &cloud);
    assert(poly_loss > 0.0);
    printf("  Polygon loss: %.6f\n", poly_loss);
    
    // Test partition loss
    polygon worst_poly;
    double part_loss = loss_part(&part, &worst_poly);
    assert(part_loss > 0.0);
    printf("  Partition loss: %.6f\n", part_loss);
    
    // Test vertex cloud loss
    double vc_loss = loss_vc(&cloud, &worst_poly, &part);
    assert(vc_loss > 0.0);
    printf("  Vertex cloud loss: %.6f\n", vc_loss);
    
    // Test gradient calculation
    double grad_loss = dloss_poly(&poly, &cloud, &grad);
    assert(grad_loss > 0.0);
    printf("  Gradient loss: %.6f\n", grad_loss);
    
    printf("✅ Loss function tests passed\n");
}

void test_data_structures() {
    printf("Testing data structure sizes and memory layout...\n");
    
    printf("  sizeof(vertex_cloud): %zu bytes\n", sizeof(vertex_cloud));
    printf("  sizeof(partition): %zu bytes\n", sizeof(partition));
    printf("  sizeof(polygon): %zu bytes\n", sizeof(polygon));  
    printf("  sizeof(gradient): %zu bytes\n", sizeof(gradient));
    
    printf("  MAX_VERTICES: %d\n", MAX_VERTICES);
    printf("  MAX_POLYGONS: %d\n", MAX_POLYGONS);
    printf("  MAX_POLYGON_SIZE: %d\n", MAX_POLYGON_SIZE);
    
    printf("✅ Data structure tests passed\n");
}

void test_edge_cases() {
    printf("Testing edge cases...\n");
    
    vertex_cloud cloud;
    polygon poly;
    
    init_vertex_cloud(&cloud);
    init_polygon(&poly);
    
    // Test with invalid polygon size
    poly.s = 2;  // Too few vertices
    assert(validate_polygon(&poly, &cloud) == 0);
    
    // Test with out-of-bounds vertex index
    poly.s = 4;
    poly.vertices[0] = -1;  // Invalid index
    assert(validate_polygon(&poly, &cloud) == 0);
    
    poly.vertices[0] = cloud.n;  // Out of bounds
    assert(validate_polygon(&poly, &cloud) == 0);
    
    printf("✅ Edge case tests passed\n");
}

int main() {
    printf("🧪 Running C Extension Tests\n");
    printf("=============================\n");
    
    test_init_functions();
    test_validation_functions();
    test_loss_functions();
    test_data_structures();
    test_edge_cases();
    
    printf("\n🎉 All C tests passed successfully!\n");
    return 0;
}