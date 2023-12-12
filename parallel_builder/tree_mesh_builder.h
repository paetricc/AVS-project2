/**
 * @file    tree_mesh_builder.h
 *
 * @author  TOMÁŠ BÁRTŮ <xbartu11@stud.fit.vutbr.cz>
 *
 * @brief   Parallel Marching Cubes implementation using OpenMP tasks + octree early elimination
 *
 * @date    12-12-2023
 **/

#ifndef TREE_MESH_BUILDER_H
#define TREE_MESH_BUILDER_H

#include "base_mesh_builder.h"

#define SQRT3 1.73205
#define CUT_OFF 3

class TreeMeshBuilder : public BaseMeshBuilder
{
public:
    TreeMeshBuilder(unsigned gridEdgeSize);

protected:
    unsigned marchCubes(const ParametricScalarField &field);
    float evaluateFieldAt(const Vec3_t<float> &pos, const ParametricScalarField &field);
    void emitTriangle(const Triangle_t &triangle);
    const Triangle_t *getTrianglesArray() const { return mTriangles.data(); }
    bool isBlockEmpty(const Vec3_t<float> currentPosition, const ParametricScalarField &field, float gridSize, float GridResolution);
    void computeCenterPosition(const Vec3_t<float> &currentPosition, Vec3_t<float> &centerPosition, const float gridSize);
    static void computeNewPosition(const Vec3_t<float> &currentPosition, Vec3_t<float> &newPosition, const float gridSize, const int i);
    unsigned int octree(const Vec3_t<float> &currentPosition, float gridSize, const ParametricScalarField &field);

    std::vector<Triangle_t> mTriangles;
};

#endif // TREE_MESH_BUILDER_H